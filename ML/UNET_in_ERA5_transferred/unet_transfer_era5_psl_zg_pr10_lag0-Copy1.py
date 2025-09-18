import numpy as np
import tensorflow as tf
import xarray as xr
import keras
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from unet_ipsl_psl_zg_pr10_to_tas_lag0 import create_UNET, CustomDataset  # import first script

# Load pretrained model

pretrained_model_path = "models/psl_zg_pr10_to_tas_ipsl_lag0.keras"
base_model = keras.models.load_model(pretrained_model_path, compile=False)

# Freeze some layers

# for layer in base_model.layers:
#     # Encodeur jusqu’à conv2d_7
#     if layer.name in ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5"]:
#         layer.trainable = False
#     else:
#         layer.trainable = True


# Recompile with new optimizer/loss

base_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),  # smaller LR for transfer learning
    loss="MSE"
)

# New dataset (same preprocessing!)

zg_file_train = "/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_train.nc"
psl_file_train = "/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_train.nc"
pr10_file_train = "/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_train.nc"
tas_file_train = "/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_train.nc"

zg_file_val = "/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_val.nc"
psl_file_val = "/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_val.nc"
pr10_file_val = "/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_val.nc"
tas_file_val = "/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_val.nc"

# Train and val datasets
dataset_train = CustomDataset(
    psl_file_train, zg_file_train, pr10_file_train, tas_file_train, batch_size=32, lag=0
)
dataset_val = CustomDataset(
    psl_file_val, zg_file_val, pr10_file_val, tas_file_val,
    batch_size=32, lag=0,
    std_ref_psl=dataset_train.psl_std,
    std_ref_zg=dataset_train.zg_std,
    std_ref_pr10=dataset_train.pr10_std
)

# Training

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    keras.callbacks.ModelCheckpoint("models/psl_zg_pr10_to_tas_transfer_era5_lag0.keras", save_best_only=True),
]

history = base_model.fit(
    dataset_train,
    epochs=50,
    validation_data=dataset_val,
    callbacks=callbacks,
    verbose=2,
)


# Save outputs

out_predict = np.squeeze(base_model.predict(dataset_val))
out_true = np.concatenate([y for _, y in dataset_val], axis=0)
out_true = np.squeeze(out_true)

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
ds.to_netcdf("models/psl_zg_pr10_tas_to_tas_era5_lag2_COMMITTOR.nc")
base_model.save("models/psl_zg_pr10_to_tas_transfer_era5_lag0.keras")
np.save("models/psl_zg_pr10_to_tas_transfer_era5_lag0.npy", history.history)
