#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic libraries
import numpy as np
import xarray as xr

# Images and file libraries
import glob


# In[ ]:


def compute_anom_multiple_runs(data):
    """Compute anomalies.

    The function computes the anomalies from a given dataset by
    substracting the climatological value of the variable (mean for
    summer or by month).

    Args:
        data: A dataset

    Returns:
        data_anom: The dataset which contains the anomalies.
    """
    dayofyear=data.time_run.time.dt.dayofyear
    data_climato = data.groupby(dayofyear).mean("time_run")
    data_anom = data.groupby(dayofyear) - data_climato
    return data_anom.drop_vars("dayofyear")


# In[ ]:


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


# In[ ]:


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

# In[ ]:


tas_path="/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r*i1p1f1/day/tas/gr/latest/tas_day_IPSL-CM6A-LR_historical_r*i1p1f1_gr_18500101-20141231.nc"
path_list=glob.glob(tas_path)
tas = xr.open_mfdataset(path_list,combine="nested", concat_dim="run",drop_variables=["time_bounds","height"],parallel=True)
tas = tas.sel(time=tas.time.dt.year>=1950)
tas = fix_lat_lon(tas)
tas = tas.sel(lat=slice(30, 75), lon=slice(-30, 40))
tas = jja(tas)
runs=[f.split("_")[4] for f in path_list]
tas['run'] = runs

tas=tas.stack(time_run=("run","time")).transpose("time_run","lat","lon")
tas.load()
tas_anom=compute_anom_multiple_runs(tas)

run_numbers = np.array([int(s.split('i')[0][1:]) for s in tas.run.values])
index=run_numbers*10**8+tas.time.dt.year.values*10**4+tas.time.dt.month.values*10**2+tas.time.dt.day.values

tas_anom = tas_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
tas_anom=tas_anom.rename({"time_run":"time"})

tas = tas.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
tas=tas.rename({"time_run":"time"})


# In[ ]:


tas.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/tas_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
tas_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/tas_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del tas
del tas_anom


# # zg

# In[ ]:


zg_path="/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r*i1p1f1/day/zg/gr/latest/zg_day_IPSL-CM6A-LR_historical_r*i1p1f1_gr_19500101-20141231.nc"
path_list=glob.glob(zg_path)
zg = xr.open_mfdataset(path_list,combine="nested", concat_dim="run",drop_variables=["time_bounds","axis_nbounds"])
zg = zg.sel(time=zg.time.dt.year>=1950,plev=50000).drop_vars("plev")
zg = fix_lat_lon(zg)
zg = zg.sel(lat=slice(30, 75), lon=slice(-30, 40))
zg = jja(zg)
runs=[f.split("_")[4] for f in path_list]
zg['run'] = runs

zg=zg.stack(time_run=("run","time")).transpose("time_run","lat","lon")
zg.load()
zg_anom=compute_anom_multiple_runs(zg)

run_numbers = np.array([int(s.split('i')[0][1:]) for s in zg.run.values])
index=run_numbers*10**8+zg.time.dt.year.values*10**4+zg.time.dt.month.values*10**2+zg.time.dt.day.values

zg_anom = zg_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
zg_anom=zg_anom.rename({"time_run":"time"})

zg = zg.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
zg=zg.rename({"time_run":"time"})


# In[ ]:


zg.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/zg_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
zg_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/zg_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del zg
del zg_anom


# # psl

# In[ ]:


psl_path="/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r*i1p1f1/day/psl/gr/latest/psl_day_IPSL-CM6A-LR_historical_r*i1p1f1_gr_18500101-20141231.nc"
path_list=glob.glob(psl_path)
psl = xr.open_mfdataset(path_list,combine="nested", concat_dim="run",drop_variables=["time_bounds","axis_nbounds"])
psl = psl.sel(time=psl.time.dt.year>=1950)
psl = fix_lat_lon(psl)
psl = psl.sel(lat=slice(30, 75), lon=slice(-30, 40))
psl = jja(psl)
runs=[f.split("_")[4] for f in path_list]
psl['run'] = runs

psl=psl.stack(time_run=("run","time")).transpose("time_run","lat","lon")
psl.load()
psl_anom=compute_anom_multiple_runs(psl)

run_numbers = np.array([int(s.split('i')[0][1:]) for s in psl.run.values])
index=run_numbers*10**8+psl.time.dt.year.values*10**4+psl.time.dt.month.values*10**2+psl.time.dt.day.values

psl_anom = psl_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
psl_anom=psl_anom.rename({"time_run":"time"})

psl = psl.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
psl=psl.rename({"time_run":"time"})


# In[ ]:


psl.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/psl_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
psl_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/psl_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del psl
del psl_anom


# # mrso

# In[ ]:


mrso_path="/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r*i1p1f1/day/mrso/gr/latest/mrso_day_IPSL-CM6A-LR_historical_r*i1p1f1_gr_18500101-20141231.nc"
path_list=glob.glob(mrso_path)
mrso = xr.open_mfdataset(path_list,combine="nested", concat_dim="run",drop_variables=["time_bounds","axis_nbounds"])
mrso = mrso.sel(time=mrso.time.dt.year>=1950)
mrso = fix_lat_lon(mrso)
mrso = mrso.sel(lat=slice(30, 75), lon=slice(-30, 40))
mrso = jja(mrso)
runs=[f.split("_")[4] for f in path_list]
mrso['run'] = runs

mrso=mrso.stack(time_run=("run","time")).transpose("time_run","lat","lon")
mrso.load()
mrso_anom=compute_anom_multiple_runs(mrso)

run_numbers = np.array([int(s.split('i')[0][1:]) for s in mrso.run.values])
index=run_numbers*10**8+mrso.time.dt.year.values*10**4+mrso.time.dt.month.values*10**2+mrso.time.dt.day.values

mrso_anom = mrso_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
mrso_anom=mrso_anom.rename({"time_run":"time"}).fillna(0)

mrso = mrso.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
mrso=mrso.rename({"time_run":"time"}).fillna(0)


# In[ ]:


mrso.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/mrso_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
mrso_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/mrso_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del mrso
del mrso_anom


# # pr(10,30,90)

# Computing it

# In[ ]:


pr_path="/bdd/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r*i1p1f1/day/pr/gr/latest/pr_day_IPSL-CM6A-LR_historical_r*i1p1f1_gr_18500101-20141231.nc"
path_list=glob.glob(pr_path)
pr = xr.open_mfdataset(path_list,combine="nested", concat_dim="run",drop_variables=["time_bounds"],chunks={"time": 250})
pr = fix_lat_lon(pr)
pr = pr.sel(lat=slice(30, 75), lon=slice(-30, 40),time=pr.time.dt.year>=1950)
runs=[f.split("_")[4] for f in path_list]
pr['run'] = runs


# In[ ]:


pr["pr_mm_day"] = pr.pr * 86400
pr["pr_mm_day"].attrs["units"] = "mm"
pr["pr_mm_day"].attrs["long_name"] = "Total precipitation"
pr = pr.drop_vars("pr").rename({"pr_mm_day": "pr"})
pr.load()


# In[ ]:


pr10=pr.copy()
pr10["pr10"] = pr10.pr.rolling(time=10).sum()
pr10["pr10"].attrs["units"] = "mm"
pr10["pr10"].attrs["long_name"] = "10 days precipitation sum"
pr10 = pr10.drop_vars("pr")
pr10 = jja(pr10)
pr10=pr10.stack(time_run=("run","time")).transpose("time_run","lat","lon")
pr10_anom=compute_anom_multiple_runs(pr10)
run_numbers = np.array([int(s.split('i')[0][1:]) for s in pr10.run.values])
index=run_numbers*10**8+pr10.time.dt.year.values*10**4+pr10.time.dt.month.values*10**2+pr10.time.dt.day.values

pr10_anom = pr10_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr10_anom=pr10_anom.rename({"time_run":"time"})
pr10 = pr10.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr10=pr10.rename({"time_run":"time"})

pr10.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
pr10_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del pr10,pr10_anom


# In[ ]:


pr30=pr.copy()
pr30["pr30"] = pr30.pr.rolling(time=30).sum()
pr30["pr30"].attrs["units"] = "mm"
pr30["pr30"].attrs["long_name"] = "10 days precipitation sum"
pr30 = pr30.drop_vars("pr")
pr30 = jja(pr30)
pr30=pr30.stack(time_run=("run","time")).transpose("time_run","lat","lon")
pr30_anom=compute_anom_multiple_runs(pr30)
run_numbers = np.array([int(s.split('i')[0][1:]) for s in pr30.run.values])
index=run_numbers*10**8+pr30.time.dt.year.values*10**4+pr30.time.dt.month.values*10**2+pr30.time.dt.day.values

pr30_anom = pr30_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr30_anom=pr30_anom.rename({"time_run":"time"})
pr30 = pr30.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr30=pr30.rename({"time_run":"time"})

pr30.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr30_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
pr30_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr30_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del pr30,pr30_anom


# In[ ]:


pr90=pr.copy()
pr90["pr90"] = pr90.pr.rolling(time=90).sum()
pr90["pr90"].attrs["units"] = "mm"
pr90["pr90"].attrs["long_name"] = "10 days precipitation sum"
pr90 = pr90.drop_vars("pr")
pr90 = jja(pr90)
pr90=pr90.stack(time_run=("run","time")).transpose("time_run","lat","lon")
pr90_anom=compute_anom_multiple_runs(pr90)
run_numbers = np.array([int(s.split('i')[0][1:]) for s in pr90.run.values])
index=run_numbers*10**8+pr90.time.dt.year.values*10**4+pr90.time.dt.month.values*10**2+pr90.time.dt.day.values

pr90_anom = pr90_anom.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr90_anom=pr90_anom.rename({"time_run":"time"})
pr90 = pr90.reset_index("time_run").assign_coords(
    time_run=index
).drop_vars(["run","time"])
pr90=pr90.rename({"time_run":"time"})

pr90.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr90_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")
pr90_anom.to_netcdf("/homedata/pchevali/clean_data_ipsl/preprocessed/pr90_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc")


# In[ ]:


del pr90,pr90_anom,pr

