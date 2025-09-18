import numpy as np
import tensorflow as tf

import xarray as xr
import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


################################# MODEL ##########################################

def convlstm_block(x, filters=64, kernel_size=(3,3), dropout=0, return_sequences=True):
    conv1 = layers.ConvLSTM2D(filters, kernel_size, padding="same",
                              return_sequences=True,
                              recurrent_dropout=dropout)(x)
    bn1 = layers.BatchNormalization()(conv1)
    act1 = layers.ReLU()(bn1)

    conv2 = layers.ConvLSTM2D(filters, kernel_size, padding="same",
                              return_sequences=return_sequences,
                              recurrent_dropout=dropout)(act1)
    bn2 = layers.BatchNormalization()(conv2)
    act2 = layers.ReLU()(bn2)

    return act2


def create_UNET_convlstm(inputs_shape=(4,160,160,3), num_classes=1,
                         depth=4, base_filters=64, dropout_rate=0.2):

    inputs = keras.Input(shape=inputs_shape)

    T, h, w, c = inputs_shape
    divisor = 2 ** depth

    # Check divisibility
    if h % divisor != 0 or w % divisor != 0:
        raise ValueError(
            f"Input height and width must be divisible by {divisor} (2^{depth}). "
            f"Got: {(h, w)}"
        )

    # Encoder
    encoders = []
    x = inputs
    for i in range(depth):
        filters = base_filters * (2**i)
        x = convlstm_block(x, filters, return_sequences=True)
        encoders.append(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)

    # Bottleneck
    x = convlstm_block(x, base_filters * (2**depth), return_sequences=True)
    x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)

    # Decoder
    for i in reversed(range(depth)):
        filters = base_filters * (2**i)
        x = layers.TimeDistributed(
            layers.Conv2DTranspose(filters, kernel_size=(2,2), strides=2, padding="same")
        )(x)
        x = layers.Concatenate()([x, encoders[i]])
        x = convlstm_block(x, filters, return_sequences=True)

    # Collapse time at the end
    x = convlstm_block(x, base_filters, return_sequences=False)

    # Output
    outputs = layers.Conv2D(num_classes, kernel_size=(1,1), padding="same", activation=None)(x)

    return keras.Model(inputs, outputs)


################################# HELPER FUNCTIONS ##########################################


def decode_index(index, what="year"):
    if what == "run":
        return index // 10**8
    elif what == "year":
        return (index // 10**4) % 10**4
    elif what == "month":
        return (index // 10**2) % 100
    elif what == "day":
        return index % 100

def add_days_from_index(index, ndays):
    """add days from an index encoded runs*10**8 + years*10**4 + months*10**2 + days"""
    days = decode_index(index, "day")
    months = decode_index(index, "month")
    years = decode_index(index, "year")
    runs = decode_index(index, "run")

    if isinstance(index, np.int64):
        dates = np.datetime64(f"{years:04d}-{months:02d}-{days:02d}")
        new_dates = dates + np.timedelta64(ndays, "D")
        # extract back to year, month, day
        years_new = new_dates.astype(object).year
        months_new = new_dates.astype(object).month
        days_new = new_dates.astype(object).day

    else:
        dates = np.array(
            [
                np.datetime64(f"{y:04d}-{m:02d}-{d:02d}")
                for y, m, d in zip(years, months, days)
            ]
        )
        new_dates = dates + np.timedelta64(ndays, "D")
        # extract back to year, month, day
        years_new = np.array([d.astype(object).year for d in new_dates])
        months_new = np.array([d.astype(object).month for d in new_dates])
        days_new = np.array([d.astype(object).day for d in new_dates])

    # re-encode
    new_index = runs * 10**8 + years_new * 10**4 + months_new * 10**2 + days_new
    return new_index


################################# DATASET ##########################################


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, psl_file, zg_file, pr10_file, tas_file, batch_size=32,std_ref_psl=None,std_ref_zg=None,std_ref_pr10=None,std_red_tas=None):
        self.zg_data = xr.open_dataset(zg_file)['zg']
        self.psl_data = xr.open_dataset(psl_file)['psl']
        self.tas_data = xr.open_dataset(tas_file)['tas']
        self.tas_input_data = xr.open_dataset(tas_file)['tas']
        self.pr10_data = xr.open_dataset(pr10_file)['pr10']
        self.tas_std = std_red_tas if std_red_tas is not None else self.tas_data.std(dim='time')
        self.psl_std = std_ref_psl if std_ref_psl is not None else self.psl_data.std(dim='time')
        self.zg_std = std_ref_zg if std_ref_zg is not None else self.zg_data.std(dim='time')
        self.pr10_std = std_ref_pr10 if std_ref_pr10 is not None else self.pr10_data.std(dim='time')
        
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.psl_data) / self.batch_size))

    def __getitem__(self, idx):
        psl = self.psl_data[idx * self.batch_size:(idx + 1) * self.batch_size].values
        psl = psl / self.psl_std.values
        zg = self.zg_data[idx * self.batch_size:(idx + 1) * self.batch_size].values
        zg = zg / self.zg_std.values
        pr10 = self.pr10_data[idx * self.batch_size:(idx + 1) * self.batch_size].values
        pr10 = pr10 / self.pr10_std.values
        tas_input = self.tas_input_data[idx * self.batch_size:(idx + 1) * self.batch_size].values
        tas_input = tas_input / self.tas_std.values
        tas = self.tas_data[idx * self.batch_size:(idx + 1) * self.batch_size].values

        # Add channel dimension
        in_tensor = np.stack([zg,psl,pr10,tas_input], axis=-1)
        out_tensor = np.expand_dims(tas, axis=-1)
      
        return in_tensor, out_tensor
        
class TemporalDataset(CustomDataset):
    def __init__(self, *args, lag=1, seq_len=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.lag=lag
        self.times = np.array(self.tas_data.time.values)

        valid_starts = []
        for i in range(len(self.times) - seq_len - lag +1):
            expected = [add_days_from_index(self.times[i], d) for d in np.arange(seq_len)]
            if np.array_equal(self.times[i:i+seq_len], expected):
                valid_starts.append(i)

        self.valid_starts = np.array(valid_starts, dtype=int)

    def __len__(self):
        return int(np.ceil(len(self.valid_starts) / self.batch_size))

    def __getitem__(self, idx):
        batch_starts = self.valid_starts[idx*self.batch_size:(idx+1)*self.batch_size]

        seqs = []
        targets = []

        for s in batch_starts:
            sl = slice(s, s+self.seq_len)
            psl = self.psl_data[sl].values / self.psl_std.values
            zg = self.zg_data[sl].values / self.zg_std.values
            pr10 = self.pr10_data[sl].values / self.pr10_std.values
            tas_input = self.tas_input_data[sl].values / self.tas_std.values

            seq = np.stack([zg, psl, pr10, tas_input], axis=-1)
            seqs.append(seq)

            tar = self.tas_data[s+self.seq_len+self.lag-1].values
            targets.append(np.expand_dims(tar, axis=-1))

        X = np.array(seqs)
        y = np.array(targets)

        return X, y

if __name__ == "__main__":

    zg_file_train = "/homedata/pchevali/clean_data_ipsl/training_val_sets/zg_anom_ipsl_train.nc"
    psl_file_train = "/homedata/pchevali/clean_data_ipsl/training_val_sets/psl_anom_ipsl_train.nc"
    pr10_file_train = "/homedata/pchevali/clean_data_ipsl/training_val_sets/pr10_anom_ipsl_train.nc"
    tas_file_train = "/homedata/pchevali/clean_data_ipsl/training_val_sets/tas_anom_ipsl_train.nc"

    zg_file_val = "/homedata/pchevali/clean_data_ipsl/training_val_sets/zg_anom_ipsl_val.nc"
    psl_file_val = "/homedata/pchevali/clean_data_ipsl/training_val_sets/psl_anom_ipsl_val.nc"
    pr10_file_val = "/homedata/pchevali/clean_data_ipsl/training_val_sets/pr10_anom_ipsl_val.nc"
    tas_file_val = "/homedata/pchevali/clean_data_ipsl/training_val_sets/tas_anom_ipsl_val.nc"

    dataset_train = TemporalDataset(psl_file_train, zg_file_train, pr10_file_train, tas_file_train,
                                    batch_size=64, seq_len=2, lag=1)
    dataset_val = TemporalDataset(psl_file_val, zg_file_val, pr10_file_val, tas_file_val,
                                  batch_size=64, seq_len=2, lag=1,
                                  std_ref_psl=dataset_train.psl_std,
                                  std_ref_zg=dataset_train.zg_std,
                                  std_ref_pr10=dataset_train.pr10_std,
                                  std_red_tas=dataset_train.tas_std)

    ################################# MODEL TRAINING ##########################################
    
    
    input_shape=(2,48,32,4)
    num_classes=1
    depth=4
    n_filters=64
    model=create_UNET_convlstm(inputs_shape=input_shape,num_classes=num_classes,depth=depth,base_filters=n_filters)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), loss="MSE"
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint("models/psl_zg_pr10_tas_to_tas_ipsl_lag1_lstm.keras", save_best_only=True),
    ]
    
    epochs = 75
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=callbacks,
        verbose=2,
    )
    
    
    ################################# MODEL EXPORT ##########################################
    
    
    out_predict=np.squeeze(model.predict(dataset_val))
    out_true = np.concatenate([y for _, y in dataset_val], axis=0)
    out_true = np.squeeze(out_true)
    
    time = dataset_val.tas_data.time.values[:out_true.shape[0]]
    lat  = dataset_val.tas_data.lat.values
    lon  = dataset_val.tas_data.lon.values
    
    ds = xr.Dataset(
        {
            "out_true": (["time", "lat", "lon"], out_true),
            "out_predict": (["time", "lat", "lon"], out_predict),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        }
    )
    
    ds.to_netcdf("models/psl_zg_pr10_tas_to_tas_ipsl_lag1_lstm.nc")
    model.save("models/psl_zg_pr10_tas_to_tas_ipsl_lag1_lstm.keras")
    np.save('models/psl_zg_pr10_tas_to_tas_ipsl_lag1_lstm.npy',history.history)
