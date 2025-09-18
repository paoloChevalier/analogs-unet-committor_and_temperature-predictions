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

def conv_block(inputs,filters=64):
    
    conv1 = layers.Conv2D(filters, kernel_size = (3,3), padding = "same")(inputs)
    batch_norm1 = layers.BatchNormalization()(conv1)
    act1 = layers.ReLU()(batch_norm1)
    
    conv2 = layers.Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
    batch_norm2 = layers.BatchNormalization()(conv2)
    act2 = layers.ReLU()(batch_norm2)
    
    return act2

def create_UNET(inputs_shape=(160,160,3),num_classes=1, depth=4, base_filters=64, dropout_rate=0.2):
    inputs = keras.Input(shape=inputs_shape)

    h, w, c = inputs_shape
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
        x = conv_block(x, filters)
        encoders.append(x)
        x = layers.MaxPooling2D((2,2))(x)
    
    # Bottleneck
    x = conv_block(x, base_filters * (2**depth))
    x = layers.Dropout(dropout_rate)(x)
    
    # Decoder
    for i in reversed(range(depth)):
        filters = base_filters * (2**i)
        x = layers.Conv2DTranspose(filters, kernel_size=(2,2), strides=2, padding='same')(x)
        x = layers.Concatenate()([x, encoders[i]])
        x = conv_block(x, filters)
    
    # Output
    outputs = layers.Conv2D(num_classes, kernel_size=(1,1), padding='same', activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    return model

################################# CUSTOM LOSS ##########################################

def france_weighted_binary_crossentropy(france_mask_tensor):
    """Create loss function that only computes over France region"""
    def loss_fn(y_true, y_pred):
        # Reshape France mask to match BCE loss shape
        france_mask = tf.reshape(france_mask_tensor, [1, *france_mask_tensor.shape])
        
        # Compute BCE loss and apply France mask
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        masked_bce = bce * france_mask
        
        # Return mean loss over France region only
        return tf.reduce_sum(masked_bce) / (tf.reduce_sum(france_mask) + 1e-7)
    
    return loss_fn


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
    def __init__(self, psl_file, zg_file, pr10_file, tas_file, batch_size=32,lag=0,std_ref_psl=None,std_ref_zg=None,std_ref_pr10=None,std_red_tas=None,q_ref=None):
        self.zg_data = xr.open_dataset(zg_file)['zg']
        self.psl_data = xr.open_dataset(psl_file)['psl']
        self.tas_data = xr.open_dataset(tas_file)['tas']
        self.tas_input_data = xr.open_dataset(tas_file)['tas']
        self.pr10_data = xr.open_dataset(pr10_file)['pr10']
        self.tas_std = std_red_tas if std_red_tas is not None else self.tas_data.std(dim='time')
        self.psl_std = std_ref_psl if std_ref_psl is not None else self.psl_data.std(dim='time')
        self.zg_std = std_ref_zg if std_ref_zg is not None else self.zg_data.std(dim='time')
        self.pr10_std = std_ref_pr10 if std_ref_pr10 is not None else self.pr10_data.std(dim='time')
        self.q= q_ref if q_ref is not None else self.tas_data.sel(lat=48.87,lon=2.33,method="nearest").quantile(0.98).item()
        
        self.batch_size = batch_size

        # Shift tas backwards in time by lag days
        lagged_time = add_days_from_index(self.tas_data.time.values, -lag)
        self.tas_data = self.tas_data.assign_coords(time=lagged_time)

        # Intersect available times
        common_time = np.intersect1d(
            self.psl_data.time.values,
            self.tas_data.time.values
        )

        # Restrict all datasets
        self.psl_data = self.psl_data.sel(time=common_time)
        self.zg_data = self.zg_data.sel(time=common_time)
        self.pr10_data = self.pr10_data.sel(time=common_time)
        self.tas_data  = (self.tas_data.sel(time=common_time)>self.q)*1
        self.tas_input_data  = self.tas_input_data.sel(time=common_time)

        # Define France bounding box (approximate coordinates)
        self.france_lat_min, self.france_lat_max = 40, 52
        self.france_lon_min, self.france_lon_max = -5.0, 10
        
        # Create France mask
        self.france_mask = self._create_france_mask()

    def _create_france_mask(self):
        """Create a binary mask for France region"""
        lat = self.tas_data.lat.values
        lon = self.tas_data.lon.values
        
        # Create 2D grid of coordinates
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create mask for France
        france_mask = ((lat_grid >= self.france_lat_min) & 
                       (lat_grid <= self.france_lat_max) & 
                       (lon_grid >= self.france_lon_min) & 
                       (lon_grid <= self.france_lon_max))
        
        return france_mask.astype(np.float32)

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
        output = self.tas_data[idx * self.batch_size:(idx + 1) * self.batch_size].values

        # Add channel dimension
        in_tensor = np.stack([zg,psl,pr10,tas_input], axis=-1)
        out_tensor = np.expand_dims(output, axis=-1)
      
        return in_tensor, out_tensor


if __name__ == "__main__":

    zg_file_train = "/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_train.nc"
    psl_file_train = "/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_train.nc"
    pr10_file_train = "/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_train.nc"
    tas_file_train = "/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_train.nc"

    zg_file_val = "/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_val.nc"
    psl_file_val = "/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_val.nc"
    pr10_file_val = "/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_val.nc"
    tas_file_val = "/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_val.nc"

    dataset_train = CustomDataset(psl_file_train, zg_file_train, pr10_file_train, tas_file_train, batch_size=64, lag=9)
    dataset_val = CustomDataset(
        psl_file_val, zg_file_val, pr10_file_val, tas_file_val, batch_size=64, lag=9,
        std_ref_psl=dataset_train.psl_std,
        std_ref_zg=dataset_train.zg_std,
        std_ref_pr10=dataset_train.pr10_std,
        std_red_tas=dataset_train.tas_std,
        q_ref=dataset_train.q
    )


    ################################# MODEL TRAINING ##########################################
    
    
    input_shape=(48,32,4)
    num_classes=1
    depth=3
    n_filters=64
    model=create_UNET(inputs_shape=input_shape,num_classes=num_classes,depth=depth,base_filters=n_filters)
    
    # Use custom loss that only computes over France
    custom_loss = france_weighted_binary_crossentropy(dataset_train.france_mask)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), 
        loss=custom_loss, 
        metrics=["accuracy"]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint("models/psl_zg_pr10_tas_to_tas_era5_lag9_COMMITTOR.keras", save_best_only=True),
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
    
    # Get predictions and remove values outside France
    out_predict = np.squeeze(model.predict(dataset_val))
    out_predict = out_predict * dataset_val.france_mask  # Remove non-France predictions
    
    out_true = np.concatenate([y for _, y in dataset_val], axis=0)
    out_true = np.squeeze(out_true)
    out_true = out_true * dataset_val.france_mask  # Also mask true values for consistency

    
    time = dataset_val.tas_data.time.values[:out_true.shape[0]]
    lat = dataset_val.tas_data.lat.values
    lon = dataset_val.tas_data.lon.values
    
    ds = xr.Dataset(
        {
            "out_true": (["time", "lat", "lon"], out_true),
            "out_predict": (["time", "lat", "lon"], out_predict),
        },
        coords={"time": time, "lat": lat, "lon": lon}
    )
    
    ds.to_netcdf("models/psl_zg_pr10_tas_to_tas_era5_lag9_COMMITTOR.nc")
    model.save("models/psl_zg_pr10_tas_to_tas_era5_lag9_COMMITTOR.keras")
    np.save('models/psl_zg_pr10_tas_to_tas_era5_lag9_COMMITTOR.npy',history.history)
