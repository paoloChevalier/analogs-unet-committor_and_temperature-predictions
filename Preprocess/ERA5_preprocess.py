#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic libraries
import numpy as np
import xarray as xr
import pandas as pd

# In[2]:


def compute_anom(data):
    """Compute anomalies.

    The function computes the anomalies from a given dataset by
    substracting the climatological value of the variable (mean for
    summer or by month).

    Args:
        data: A dataset

    Returns:
        data_anom: The dataset which contains the anomalies.
    """
    data_climato = data.groupby("time.dayofyear").mean("time")
    data_anom = data.groupby("time.dayofyear") - data_climato
    return data_anom.drop_vars("dayofyear")


# In[3]:


def fix_lat_lon(data):
    """Tidy latitude and longitude data.

    The function fixes issues from descending values and 0-360 range.

    Args:
        data: A dataset

    Returns:
        data: The fixed dataset.
    """
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby("lon")
    data = data.assign_coords(lat=data.lat).sortby("lat")
    return data


# In[4]:


def jja(data):
    """Extract jja months. (and 15 days from may and september)

    The function extracts the jja months from a given dataset.

    Args:
        data: A dataset

    Returns:
        data: The dataset with only the jja months.
    """
    mask = (
        data.time.dt.month.isin([6, 7, 8])
        | ((data.time.dt.month == 5) & (data.time.dt.day >= 11))
        | ((data.time.dt.month == 9) & (data.time.dt.day <= 20))
    )
    data = data.sel(time=mask)
    return data


# # tas

# In[12]:


t2m_era5_path = "/homedata/pchevali/ERA5/raw/era5_t2m_daily_eu.nc"
t2m_era5 = xr.open_dataset(t2m_era5_path, drop_variables=["time_bnds"])
t2m_era5 = t2m_era5.sel(time=t2m_era5.time.dt.year < 2025)
t2m_era5 = jja(t2m_era5)
t2m_era5 = fix_lat_lon(t2m_era5)
t2m_era5["time"] = t2m_era5.time + pd.Timedelta(minutes=30)
t2m_era5 = t2m_era5.rename({"t2m": "tas"})


# In[13]:


t2m_era5_anom = compute_anom(t2m_era5)


# In[14]:


index = (
    t2m_era5.time.dt.year.values * 10**4
    + t2m_era5.time.dt.month.values * 10**2
    + t2m_era5.time.dt.day.values
)
t2m_era5 = t2m_era5.reset_index("time").assign_coords(time=index)
t2m_era5_anom = t2m_era5_anom.reset_index("time").assign_coords(time=index)


# In[15]:


t2m_era5.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_tas_daily_eu.nc")
t2m_era5_anom.to_netcdf(
    "/homedata/pchevali/ERA5/preprocessed/era5_tas_anom_daily_eu.nc"
)


# In[16]:


del t2m_era5, t2m_era5_anom


# # zg

# In[8]:


z500_era5_path = "/homedata/pchevali/ERA5/raw/era5_z500_daily_eu.nc"
z500_era5 = xr.open_dataset(z500_era5_path, drop_variables=["time_bnds"])
z500_era5 = z500_era5.sel(plev=5e04).drop_vars("plev")
z500_era5 = z500_era5.sel(time=z500_era5.time.dt.year < 2025)
z500_era5 = jja(z500_era5)
z500_era5 = fix_lat_lon(z500_era5)
z500_era5["time"] = z500_era5.time + pd.Timedelta(minutes=30)
z500_era5 = z500_era5.rename({"z500": "zg"})


# In[9]:


z500_era5_anom = compute_anom(z500_era5)


# In[10]:


index = (
    z500_era5.time.dt.year.values * 10**4
    + z500_era5.time.dt.month.values * 10**2
    + z500_era5.time.dt.day.values
)
z500_era5 = z500_era5.reset_index("time").assign_coords(time=index)
z500_era5_anom = z500_era5_anom.reset_index("time").assign_coords(time=index)


# In[11]:


z500_era5.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_zg_daily_eu.nc")
z500_era5_anom.to_netcdf(
    "/homedata/pchevali/ERA5/preprocessed/era5_zg_anom_daily_eu.nc"
)


# In[12]:


del z500_era5, z500_era5_anom


# # psl

# In[13]:


msl_era5_path = "/homedata/pchevali/ERA5/raw/era5_msl_daily_eu.nc"
msl_era5 = xr.open_dataset(msl_era5_path, drop_variables=["time_bnds"])
msl_era5 = msl_era5.sel(time=msl_era5.time.dt.year < 2025)
msl_era5 = jja(msl_era5)
msl_era5 = fix_lat_lon(msl_era5)
msl_era5["time"] = msl_era5.time + pd.Timedelta(minutes=30)
msl_era5 = msl_era5.rename({"msl": "psl"})


# In[14]:


msl_era5_anom = compute_anom(msl_era5)


# In[15]:


index = (
    msl_era5.time.dt.year.values * 10**4
    + msl_era5.time.dt.month.values * 10**2
    + msl_era5.time.dt.day.values
)
msl_era5 = msl_era5.reset_index("time").assign_coords(time=index)
msl_era5_anom = msl_era5_anom.reset_index("time").assign_coords(time=index)


# In[16]:


msl_era5.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_psl_daily_eu.nc")
msl_era5_anom.to_netcdf(
    "/homedata/pchevali/ERA5/preprocessed/era5_psl_anom_daily_eu.nc"
)


# In[17]:


del msl_era5, msl_era5_anom


# # mrso

# In[18]:


mrso1_path = "/homedata/pchevali/ERA5/raw/era5_mrso_daily_eu1.nc"
mrso1 = xr.open_dataset(mrso1_path)
mrso2_path = "/homedata/pchevali/ERA5/raw/era5_mrso_daily_eu2.nc"
mrso2 = xr.open_dataset(mrso2_path)
mrso3_path = "/homedata/pchevali/ERA5/raw/era5_mrso_daily_eu3.nc"
mrso3 = xr.open_dataset(mrso3_path)
mrso4_path = "/homedata/pchevali/ERA5/raw/era5_mrso_daily_eu4.nc"
mrso4 = xr.open_dataset(mrso4_path)


# In[19]:


mrso1 = mrso1.rename(
    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
).drop_vars(["expver", "number"])
mrso2 = mrso2.rename(
    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
).drop_vars(["expver", "number"])
mrso3 = mrso3.rename(
    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
).drop_vars(["expver", "number"])
mrso4 = mrso4.rename(
    {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
).drop_vars(["expver", "number"])


# In[20]:


mrso_val = (
    1000
    * (
        mrso1.swvl1 * 0.07
        + mrso2.swvl2 * 0.21
        + mrso3.swvl3 * 0.71
        + mrso4.swvl4 * 1.89
    )
    * (200 / 289)
)


# In[21]:


mrso = mrso1.copy()
mrso["mrso"] = mrso_val
mrso = mrso.drop_vars(["swvl1"])
mrso = jja(mrso)
mrso = mrso.sel(time=mrso.time.dt.year < 2025)
mrso = fix_lat_lon(mrso)


# In[22]:


mrso_anom = compute_anom(mrso)


# In[23]:


index = (
    mrso.time.dt.year.values * 10**4
    + mrso.time.dt.month.values * 10**2
    + mrso.time.dt.day.values
)
mrso = mrso.reset_index("time").assign_coords(time=index).fillna(0)
mrso_anom = mrso_anom.reset_index("time").assign_coords(time=index).fillna(0)


# In[24]:


mrso.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_mrso_daily_eu.nc")
mrso_anom.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_mrso_anom_daily_eu.nc")


# In[25]:


del mrso, mrso_anom, mrso1, mrso2, mrso3, mrso4, mrso_val


# # pr (10,30,90)

# In[5]:


tp_path = "/homedata/pchevali/ERA5/raw/era5_tp_daily_eu.nc"
tp = xr.open_dataset(tp_path)
tp = fix_lat_lon(tp).sel(time=tp.time.dt.year < 2025)
tp.close()


# In[6]:


pr10 = tp.copy()
pr10["pr10"] = tp.tp.rolling(time=10).sum()
pr10["pr10"].attrs["units"] = "mm"
pr10["pr10"].attrs["long_name"] = "10 days precipitation sum"
pr10 = pr10.drop_vars("tp")
pr10 = jja(pr10)
pr10_anom = compute_anom(pr10)
index = (
    pr10.time.dt.year.values * 10**4
    + pr10.time.dt.month.values * 10**2
    + pr10.time.dt.day.values
)
pr10 = pr10.reset_index("time").assign_coords(time=index).fillna(0)
pr10_anom = pr10_anom.reset_index("time").assign_coords(time=index).fillna(0)
pr10.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr10_daily_eu.nc")
pr10_anom.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr10_anom_daily_eu.nc")


# In[7]:


del pr10, pr10_anom, index


# In[8]:


pr30 = tp.copy()
pr30["pr30"] = tp.tp.rolling(time=30).sum()
pr30["pr30"].attrs["units"] = "mm"
pr30["pr30"].attrs["long_name"] = "30 days precipitation sum"
pr30 = pr30.drop_vars("tp")
pr30 = jja(pr30)
pr30_anom = compute_anom(pr30)
index = (
    pr30.time.dt.year.values * 10**4
    + pr30.time.dt.month.values * 10**2
    + pr30.time.dt.day.values
)
pr30 = pr30.reset_index("time").assign_coords(time=index).fillna(0)
pr30_anom = pr30_anom.reset_index("time").assign_coords(time=index).fillna(0)
pr30.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr30_daily_eu.nc")
pr30_anom.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr30_anom_daily_eu.nc")


# In[9]:


del pr30, pr30_anom, index


# In[10]:


pr90 = tp.copy()
pr90["pr90"] = tp.tp.rolling(time=90).sum()
pr90["pr90"].attrs["units"] = "mm"
pr90["pr90"].attrs["long_name"] = "90 days precipitation sum"
pr90 = pr90.drop_vars("tp")
pr90 = jja(pr90)
pr90_anom = compute_anom(pr90)
index = (
    pr90.time.dt.year.values * 10**4
    + pr90.time.dt.month.values * 10**2
    + pr90.time.dt.day.values
)
pr90 = pr90.reset_index("time").assign_coords(time=index).fillna(0)
pr90_anom = pr90_anom.reset_index("time").assign_coords(time=index).fillna(0)
pr90.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr90_daily_eu.nc")
pr90_anom.to_netcdf("/homedata/pchevali/ERA5/preprocessed/era5_pr90_anom_daily_eu.nc")


# In[11]:


del tp, pr90, pr90_anom, index

