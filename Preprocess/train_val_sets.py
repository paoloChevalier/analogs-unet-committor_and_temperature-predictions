#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr


# Load ERA5 data

# In[ ]:


tas_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_tas_daily_eu.nc"
tas_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_tas_anom_daily_eu.nc"
tas_era5 = xr.open_dataset(tas_path_era5)
tas_anom_era5 = xr.open_dataset(tas_anom_path_era5)


# In[ ]:


zg_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_zg_daily_eu.nc"
zg_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_zg_anom_daily_eu.nc"
zg_era5 = xr.open_dataset(zg_path_era5)
zg_anom_era5 = xr.open_dataset(zg_anom_path_era5)


# In[ ]:


psl_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_psl_daily_eu.nc"
psl_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_psl_anom_daily_eu.nc"
psl_era5 = xr.open_dataset(psl_path_era5)
psl_anom_era5 = xr.open_dataset(psl_anom_path_era5)


# In[ ]:


pr90_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr90_daily_eu.nc"
pr90_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr90_anom_daily_eu.nc"
pr90_era5 = xr.open_dataset(pr90_path_era5)
pr90_anom_era5 = xr.open_dataset(pr90_anom_path_era5)


# In[ ]:


pr30_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr30_daily_eu.nc"
pr30_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr30_anom_daily_eu.nc"
pr30_era5 = xr.open_dataset(pr30_path_era5)
pr30_anom_era5 = xr.open_dataset(pr30_anom_path_era5)


# In[ ]:


pr10_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr10_daily_eu.nc"
pr10_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr10_anom_daily_eu.nc"
pr10_era5 = xr.open_dataset(pr10_path_era5)
pr10_anom_era5 = xr.open_dataset(pr10_anom_path_era5)


# In[ ]:


mrso_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_mrso_daily_eu.nc"
mrso_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_mrso_anom_daily_eu.nc"
mrso_era5 = xr.open_dataset(mrso_path_era5)
mrso_anom_era5 = xr.open_dataset(mrso_anom_path_era5)


# Load IPSL-CM6A-LR data

# In[ ]:


tas_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/tas_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
tas_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/tas_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
tas_ipsl = xr.open_dataset(tas_path_ipsl)
tas_anom_ipsl = xr.open_dataset(tas_anom_path_ipsl)


# In[ ]:


zg_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/zg_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
zg_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/zg_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
zg_ipsl = xr.open_dataset(zg_path_ipsl)
zg_anom_ipsl = xr.open_dataset(zg_anom_path_ipsl)


# In[ ]:


psl_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/psl_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
psl_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/psl_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
psl_ipsl = xr.open_dataset(psl_path_ipsl)
psl_anom_ipsl = xr.open_dataset(psl_anom_path_ipsl)


# In[ ]:


mrso_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/mrso_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
mrso_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/mrso_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
mrso_ipsl = xr.open_dataset(mrso_path_ipsl)
mrso_anom_ipsl = xr.open_dataset(mrso_anom_path_ipsl)


# In[ ]:


pr10_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr10_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr10_ipsl = xr.open_dataset(pr10_path_ipsl)
pr10_anom_ipsl = xr.open_dataset(pr10_anom_path_ipsl)


# In[ ]:


pr30_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr30_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr30_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr30_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr30_ipsl = xr.open_dataset(pr30_path_ipsl)
pr30_anom_ipsl = xr.open_dataset(pr30_anom_path_ipsl)


# In[ ]:


pr90_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr90_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr90_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr90_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr90_ipsl = xr.open_dataset(pr90_path_ipsl)
pr90_anom_ipsl = xr.open_dataset(pr90_anom_path_ipsl)


# dealing with the dimensions ( %16)

# In[ ]:


for i in range(100):
    if i%(2**4)==0:
        print(i)


# In[ ]:


lat = tas_anom_ipsl.lat.values
lon = tas_anom_ipsl.lon.values
print("lat : ", len(lat), "\nlon : ", len(lon))


# In[ ]:


new_lat = np.linspace(lat.min(), lat.max(), 48)
new_lon = np.linspace(lon.min(), lon.max(), 32)


# In[ ]:


tas_anom_ipsl_interp = tas_anom_ipsl.interp(lat=new_lat, lon=new_lon, method="linear")
zg_anom_ipsl_interp = zg_anom_ipsl.interp(lat=new_lat, lon=new_lon, method="linear")
psl_anom_ipsl_interp = psl_anom_ipsl.interp(lat=new_lat, lon=new_lon, method="linear")
pr10_anom_ipsl_interp = pr10_anom_ipsl.interp(lat=new_lat, lon=new_lon, method="linear")


# In[ ]:


train_split_era5=int(0.8*len(tas_era5.time.values))
train_split_ipsl=int(0.8*33)*8645


# ipsl

# In[ ]:


tas_anom_ipsl_train=tas_anom_ipsl_interp.isel(time=slice(0,train_split_ipsl))
tas_anom_ipsl_val=tas_anom_ipsl_interp.isel(time=slice(train_split_ipsl,None))
zg_anom_ipsl_train=zg_anom_ipsl_interp.isel(time=slice(0,train_split_ipsl))
zg_anom_ipsl_val=zg_anom_ipsl_interp.isel(time=slice(train_split_ipsl,None))
psl_anom_ipsl_train=psl_anom_ipsl_interp.isel(time=slice(0,train_split_ipsl))
psl_anom_ipsl_val=psl_anom_ipsl_interp.isel(time=slice(train_split_ipsl,None))
pr10_anom_ipsl_train=pr10_anom_ipsl_interp.isel(time=slice(0,train_split_ipsl))
pr10_anom_ipsl_val=pr10_anom_ipsl_interp.isel(time=slice(train_split_ipsl,None))


# In[ ]:


tas_anom_ipsl_train.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/tas_anom_ipsl_train.nc")
tas_anom_ipsl_val.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/tas_anom_ipsl_val.nc")
zg_anom_ipsl_train.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/zg_anom_ipsl_train.nc")
zg_anom_ipsl_val.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/zg_anom_ipsl_val.nc")
psl_anom_ipsl_train.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/psl_anom_ipsl_train.nc")
psl_anom_ipsl_val.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/psl_anom_ipsl_val.nc")
pr10_anom_ipsl_train.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/pr10_anom_ipsl_train.nc")
pr10_anom_ipsl_val.to_netcdf("/homedata/pchevali/clean_data_ipsl/training_val_sets/pr10_anom_ipsl_val.nc")


# era

# In[ ]:


tas_anom_era5_interp = tas_anom_era5.interp(lat=tas_anom_ipsl_interp.lat,lon=tas_anom_ipsl_interp.lon)
zg_anom_era5_interp = zg_anom_era5.interp(lat=tas_anom_ipsl_interp.lat,lon=tas_anom_ipsl_interp.lon)
psl_anom_era5_interp = psl_anom_era5.interp(lat=tas_anom_ipsl_interp.lat,lon=tas_anom_ipsl_interp.lon)
pr10_anom_era5_interp = pr10_anom_era5.interp(lat=tas_anom_ipsl_interp.lat,lon=tas_anom_ipsl_interp.lon)


# In[ ]:


tas_anom_era5_train=tas_anom_era5_interp.isel(time=slice(0,train_split_era5))
tas_anom_era5_val=tas_anom_era5_interp.isel(time=slice(train_split_era5,None))
zg_anom_era5_train=zg_anom_era5_interp.isel(time=slice(0,train_split_era5))
zg_anom_era5_val=zg_anom_era5_interp.isel(time=slice(train_split_era5,None))
psl_anom_era5_train=psl_anom_era5_interp.isel(time=slice(0,train_split_era5))
psl_anom_era5_val=psl_anom_era5_interp.isel(time=slice(train_split_era5,None))
pr10_anom_era5_train=pr10_anom_era5_interp.isel(time=slice(0,train_split_era5))
pr10_anom_era5_val=pr10_anom_era5_interp.isel(time=slice(train_split_era5,None))


# In[ ]:


tas_anom_era5_train.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_train.nc")
tas_anom_era5_val.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/tas_anom_era5_val.nc")
zg_anom_era5_train.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_train.nc")
zg_anom_era5_val.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/zg_anom_era5_val.nc")
psl_anom_era5_train.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_train.nc")
psl_anom_era5_val.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/psl_anom_era5_val.nc")
pr10_anom_era5_train.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_train.nc")
pr10_anom_era5_val.to_netcdf("/homedata/pchevali/ERA5/training_val_sets/pr10_anom_era5_val.nc")

