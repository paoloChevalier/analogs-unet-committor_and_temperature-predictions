#!/usr/bin/env python
# coding: utf-8

# Basic libraries
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
import dask.array as da

# Some helper functions


def decode_index(index, what="year"):
    """decodes an index encoded runs*10**8 + years*10**4 + months*10**2 + days to a date"""
    if what == "run":
        return index // 10**8
    elif what == "year":
        return (index // 10**4) % 10**4
    elif what == "month":
        return (index // 10**2) % 100
    elif what == "day":
        return index % 100


def index_to_date(index):
    """transforms an index encoded runs*10**8 + years*10**4 + months*10**2 + days to a date"""
    years = decode_index(index, "year")
    months = decode_index(index, "month")
    days = decode_index(index, "day")
    runs = decode_index(index, "run")

    # Ensure arrays for vectorization
    scalar_input = np.isscalar(index)
    years = np.atleast_1d(years)
    months = np.atleast_1d(months)
    days = np.atleast_1d(days)
    runs = np.atleast_1d(runs)

    dates = pd.to_datetime(
        dict(year=years.astype(int), month=months.astype(int), day=days.astype(int)),
        errors="raise",
    )

    # If input was scalar → return scalar
    if scalar_input:
        return (dates[0], runs[0])
    return dates, runs


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


# # Load IPSL-CM6A-LR data


tas_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/tas_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
tas_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/tas_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
tas_ipsl = xr.open_dataset(tas_path_ipsl)
tas_anom_ipsl = xr.open_dataset(tas_anom_path_ipsl)


zg_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/zg_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
zg_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/zg_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
zg_ipsl = xr.open_dataset(zg_path_ipsl)
zg_anom_ipsl = xr.open_dataset(zg_anom_path_ipsl)


psl_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/psl_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
psl_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/psl_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
psl_ipsl = xr.open_dataset(psl_path_ipsl)
psl_anom_ipsl = xr.open_dataset(psl_anom_path_ipsl)


pr10_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr10_anom_path_ipsl = "/homedata/pchevali/clean_data_ipsl/preprocessed/pr10_anom_day_IPSL-CM6A-LR_historical_allruns_19500101-20141231_30W_40E_30N_75N.nc"
pr10_ipsl = xr.open_dataset(pr10_path_ipsl)
pr10_anom_ipsl = xr.open_dataset(pr10_anom_path_ipsl)


# # Load ERA5 data


tas_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_tas_daily_eu.nc"
tas_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_tas_anom_daily_eu.nc"
tas_era5 = xr.open_dataset(tas_path_era5)
tas_anom_era5 = xr.open_dataset(tas_anom_path_era5)


zg_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_zg_daily_eu.nc"
zg_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_zg_anom_daily_eu.nc"
zg_era5 = xr.open_dataset(zg_path_era5)
zg_anom_era5 = xr.open_dataset(zg_anom_path_era5)


psl_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_psl_daily_eu.nc"
psl_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_psl_anom_daily_eu.nc"
psl_era5 = xr.open_dataset(psl_path_era5)
psl_anom_era5 = xr.open_dataset(psl_anom_path_era5)


pr10_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr10_daily_eu.nc"
pr10_anom_path_era5 = "/homedata/pchevali/ERA5/preprocessed/era5_pr10_anom_daily_eu.nc"
pr10_era5 = xr.open_dataset(pr10_path_era5)
pr10_anom_era5 = xr.open_dataset(pr10_anom_path_era5)


tas_era5 = tas_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
tas_anom_era5 = tas_anom_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
zg_era5 = zg_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
zg_anom_era5 = zg_anom_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
psl_era5 = psl_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
psl_anom_era5 = psl_anom_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
pr10_era5 = pr10_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)
pr10_anom_era5 = pr10_anom_era5.interp(lat=tas_ipsl.lat, lon=tas_ipsl.lon)


# # Merge data


zg_anom_merged = xr.concat([zg_anom_era5, zg_anom_ipsl], dim="time")
psl_anom_merged = xr.concat([psl_anom_era5, psl_anom_ipsl], dim="time")
pr10_anom_merged = xr.concat([pr10_anom_era5, pr10_anom_ipsl], dim="time")
tas_anom_merged = xr.concat([tas_anom_era5, tas_anom_ipsl], dim="time")



# # Helper functions

# ## Compute RMSE

# ### Functions


def compute_pairwise_rmse_fast_matmul(data1, data2):
    """Computes the pairwise rms between every times of two datasets

    The function computes the pairwise rms between every times of two datasets,
    can take a in memory array or dask backed array lazily.

    Args:
        data1: A np.array[time_run,space]
        data2: A np.array[time_run,space]

    Returns:
        A xarray.DataSet containing everything.
    """
    return np.sqrt(
        np.maximum(
            (data1**2).mean(axis=1)[:, None]
            + (data2**2).mean(axis=1)[None, :]
            - 2 * (data1 @ data2.T) / data1.shape[1],
            0,
        )
    )  # max(0,values) because matrix multiplication is fast but can be unprecise leading to <0 results


# ## Compute Analogs


def compute_analogs(
    analogs_in, analogs_of, lat1, lat2, lon1, lon2, chunk_size, corr=False, topk=20
):
    """Computes and exports the analogs

    The function computes the analogs of all the dates in analogs_of
    in analogs_in and puts them in a xarray.DataSet
    data1 and data2 MUST HAVE SAME GRID
    (faster but unstable on ram (hence the choice of chunk size is important))

    Args:
        analogs_in: The field on which to compute analogs in
        analogs_of: The field on which to compute analogs of
        lat1: The lowest value for latitude
        lat2: The highest value for latitude
        lon1: The lowest value for latitude
        lon2: The highest value for latitude
        corr: Boolean indicating whether to compute the correlations
        topk: The number of analogs you want for each day

    Returns:
        A xarray.DataSet containing everything.
    """
    analogs_of_cut = analogs_of.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).stack(
        space=["lat", "lon"]
    )
    analogs_in_cut = analogs_in.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).stack(
        space=["lat", "lon"]
    )

    times_in = analogs_in.time.values
    times_of = analogs_of.time.values

    pairwise_rmse = compute_pairwise_rmse_fast_matmul(
        analogs_in_cut.chunk({"time": chunk_size, "space": 150}).data,
        analogs_of_cut.chunk({"time": chunk_size, "space": 150}).data,
    ).compute()

    n_in, n_of = pairwise_rmse.shape

    dates_in, runs_in = index_to_date(times_in)
    dates_of, runs_of = index_to_date(times_of)

    in_years = dates_in.dt.year.values
    of_years = dates_of.dt.year.values
    in_doys = dates_in.dt.dayofyear.values
    of_doys = dates_of.dt.dayofyear.values

    rmse_results = np.empty((n_of, topk))  # array where we'll put the rmse
    date_results = np.empty(
        (n_of, topk), dtype=np.int64
    )  # array where we'll put the indexs

    # Loop over all date in the analogs_of set
    for j in range(n_of):
        # Mask: exclude ±5 days if same year OR different run
        mask = (
            (np.abs(in_doys - of_doys[j]) > 5)
            | (in_years != of_years[j])
            | (runs_in != runs_of[j])
        )

        # put all the forbidden values as inf
        vals = np.where(mask, pairwise_rmse[:, j], np.inf)

        # Select topk smallest (argpartition is like a sort but it only gives you the topk
        # biggest values and then the rest isn't sorted, the complexity is smaller than
        # that of a regular sort, )
        topk_idx = np.argpartition(vals, topk)[:topk]
        topk_sorted = topk_idx[np.argsort(vals[topk_idx])]

        rmse_results[j, :] = vals[topk_sorted]
        date_results[j, :] = times_in[topk_sorted]

    if corr:
        corr_results = np.empty((n_of, topk))
        for j in range(n_of):
            target_date = times_of[j]
            target_field = analogs_of.sel(
                lat=slice(lat1, lat2), lon=slice(lon1, lon2)
            ).sel(time=target_date)
            analogs_dates = date_results[j, :]
            analog_field = analogs_in.sel(
                lat=slice(lat1, lat2), lon=slice(lon1, lon2)
            ).sel(time=analogs_dates)
            corr_results[j, :] = xs.spearman_r(
                target_field, analog_field, dim=["lat", "lon"]
            ).values

        ds = xr.Dataset(
            {
                "rmse": (["time", "analog_number"], rmse_results),
                "analog_date": (["time", "analog_number"], date_results),
                "rank_corr": (["time", "analog_number"], corr_results),
            },
            coords={"time": times_of, "analog_number": np.arange(1, topk + 1)},
        )
    else:
        ds = xr.Dataset(
            {
                "rmse": (["time", "analog_number"], rmse_results),
                "analog_date": (["time", "analog_number"], date_results),
            },
            coords={"time": times_of, "analog_number": np.arange(1, topk + 1)},
        )
    return ds


def compute_analogs_weighted(
    analogs_in,
    analogs_of,
    weights,
    lats1,
    lats2,
    lons1,
    lons2,
    chunk_size,
    topk=20,
    normalise="normal"
):
    """Computes and exports the analogs for different fields

    The function computes the analogs of all the dates in analogs_of
    in analogs_in and puts them in a xarray.DataSet
    data1 and data2 MUST HAVE SAME GRIDS
    (fast but unstable on ram (hence the choice of chunk size is important))

    Args:
        analogs_in: The fields on which to compute analogs in
        analogs_of: The fields on which to compute analogs of
        lat1: The lowest value for latitude
        lat2: The highest value for latitude
        lon1: The lowest value for latitude
        lon2: The highest value for latitude
        weights: NP ARRAY the weights for each field in the same order as input
        topk: The number of analogs you want for each day

    Returns:
        A xarray.DataSet containing everything.
    """
    analogs_of_cut = [
        field_of.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).stack(
            space=["lat", "lon"]
        )
        for field_of, lat1, lat2, lon1, lon2 in zip(
            analogs_of, lats1, lats2, lons1, lons2
        )
    ]

    analogs_in_cut = [
        field_in.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).stack(
            space=["lat", "lon"]
        )
        for field_in, lat1, lat2, lon1, lon2 in zip(
            analogs_in, lats1, lats2, lons1, lons2
        )
    ]

    if normalise=="normal":
        analogs_of_cut = [
            ((field_of - field_of.mean(dim="space")) / field_of.std(dim="space"))
            for field_of in analogs_of_cut
        ]
    
        analogs_in_cut = [
            ((field_in - field_in.mean(dim="space")) / field_in.std(dim="space"))
            for field_in in analogs_in_cut
        ]

    elif normalise=="rank":
        analogs_of_cut = [
            field_of.rank(dim="time",pct=True)
            for field_of in analogs_of_cut
        ]
    
        analogs_in_cut = [
            field_in.rank(dim="time",pct=True)
            for field_in in analogs_in_cut
        ]
    else:
        print("???")
        return

    times_in = analogs_in[0].time.values
    times_of = analogs_of[0].time.values

    pairwise_rmse = da.array(
        [
            compute_pairwise_rmse_fast_matmul(
                target.chunk({"time": chunk_size, "space": 250}).data,
                source.chunk({"time": chunk_size, "space": 250}).data,
            )
            for target, source in zip(analogs_in_cut, analogs_of_cut)
        ]
    )

    pairwise_rmse = da.sqrt(
        da.sum(pairwise_rmse * da.sqrt(weights[:, None, None]), axis=0)
    ).compute()

    n_in, n_of = pairwise_rmse.shape

    dates_in, runs_in = index_to_date(times_in)
    dates_of, runs_of = index_to_date(times_of)

    in_years = dates_in.dt.year.values
    of_years = dates_of.dt.year.values
    in_doys = dates_in.dt.dayofyear.values
    of_doys = dates_of.dt.dayofyear.values

    rmse_results = np.empty((n_of, topk))  # array where we'll put the rmse
    date_results = np.empty(
        (n_of, topk), dtype=np.int64
    )  # array where we'll put the indexs

    # Loop over all date in the analogs_of set
    for j in range(n_of):
        # Mask: exclude ±5 days if same year OR different run
        mask = (
            (np.abs(in_doys - of_doys[j]) > 5)
            | (in_years != of_years[j])
            | (runs_in != runs_of[j])
        )

        # put all the forbidden values as inf
        vals = np.where(mask, pairwise_rmse[:, j], np.inf)

        # Select topk smallest (argpartition is like a sort but it only gives you the topk
        # biggest values and then the rest isn't sorted, the complexity is smaller than
        # that of a regular sort, )
        topk_idx = np.argpartition(vals, topk)[:topk]
        topk_sorted = topk_idx[np.argsort(vals[topk_idx])]

        rmse_results[j, :] = vals[topk_sorted]
        date_results[j, :] = times_in[topk_sorted]

    ds = xr.Dataset(
        {
            "rmse": (["time", "analog_number"], rmse_results),
            "analog_date": (["time", "analog_number"], date_results),
        },
        coords={"time": times_of, "analog_number": np.arange(1, topk + 1)},
    )
    return ds


# # One variable

# ## ERA5 into ERA5


# lat1, lat2, lon1, lon2 = 35, 75, -30, 40

# analogs_zg_era5_to_era5 = compute_analogs(
#     zg_anom_era5.zg, zg_anom_era5.zg, lat1, lat2, lon1, lon2, -1,corr=True
# )

# analogs_psl_era5_to_era5 = compute_analogs(
#     psl_anom_era5.psl, psl_anom_era5.psl, lat1, lat2, lon1, lon2, -1, corr=True
# )

# analogs_zg_era5_to_era5.to_netcdf("/homedata/pchevali/analogs/analogs_zg_era5_to_era5.nc")
# analogs_psl_era5_to_era5.to_netcdf("/homedata/pchevali/analogs/analogs_psl_era5_to_era5.nc")


# ## ERA5 into IPSL-CM6A-LR and ERA5


# lat1, lat2, lon1, lon2 = 35, 75, -30, 40

# analogs_zg_era5_to_all = compute_analogs(
#     zg_anom_merged.zg, zg_anom_era5.zg, lat1, lat2, lon1, lon2, 10000,corr=True
# )

# analogs_psl_era5_to_all = compute_analogs(
#     psl_anom_merged.psl, psl_anom_era5.psl, lat1, lat2, lon1, lon2, 10000, corr=True
# )

# analogs_zg_era5_to_all.to_netcdf("/homedata/pchevali/analogs/analogs_zg_era5_to_all.nc")
# analogs_psl_era5_to_all.to_netcdf("/homedata/pchevali/analogs/analogs_psl_era5_to_all.nc")


# # Multiple variables


# era5_fields = [
#     zg_anom_era5.zg,
#     psl_anom_era5.psl,
#     pr10_anom_era5.pr10,
# ]
# merged_fields = [
#     zg_anom_merged.zg,
#     psl_anom_merged.psl,
#     pr10_anom_merged.pr10,
# ]
# W = np.array(
#     [
#         [0.8, 0.1, 0.1],
#         [0.1, 0.8, 0.1],
#         [0.1, 0.1, 0.8],
#         [0.7, 0.2, 0.1],
#         [0.7, 0.1, 0.2],
#         [0.2, 0.7, 0.1],
#         [0.1, 0.7, 0.2],
#         [0.2, 0.1, 0.7],
#         [0.1, 0.2, 0.7],
#         [0.6, 0.3, 0.1],
#         [0.6, 0.1, 0.3],
#         [0.3, 0.6, 0.1],
#         [0.1, 0.6, 0.3],
#         [0.3, 0.1, 0.6],
#         [0.1, 0.3, 0.6],
#         [0.5, 0.4, 0.1],
#         [0.5, 0.1, 0.4],
#         [0.4, 0.5, 0.1],
#         [0.1, 0.5, 0.4],
#         [0.4, 0.1, 0.5],
#         [0.1, 0.4, 0.5],
#         [0.45, 0.45, 0.1],
#         [0.45, 0.1, 0.45],
#         [0.1, 0.45, 0.45],
#         [0.55, 0.35, 0.1],
#         [0.55, 0.1, 0.35],
#         [0.35, 0.55, 0.1],
#         [0.1, 0.55, 0.35],
#         [0.35, 0.1, 0.55],
#         [0.1, 0.35, 0.55],
#         [0.75, 0.15, 0.1],
#         [0.75, 0.1, 0.15],
#         [0.15, 0.75, 0.1],
#         [0.1, 0.75, 0.15],
#         [0.15, 0.1, 0.75],
#         [0.1, 0.15, 0.75],
#         [0.6, 0.25, 0.15],
#         [0.6, 0.15, 0.25],
#         [0.25, 0.6, 0.15],
#         [0.15, 0.6, 0.25],
#         [0.25, 0.15, 0.6],
#         [0.15, 0.25, 0.6],
#         [0.7, 0.15, 0.15],
#         [0.15, 0.7, 0.15],
#         [0.15, 0.15, 0.7],
#         [0.8, 0.05, 0.15],
#         [0.8, 0.15, 0.05],
#         [0.05, 0.8, 0.15],
#         [0.15, 0.8, 0.05],
#         [0.05, 0.15, 0.8],
#     ]
# )
# w = np.array([0.45, 0.45, 0.1])
# lats1 = [30, 30, 42.5]
# lats2 = [75, 75, 52.5]
# lons1 = [-30, -30, 1.5]
# lons2 = [40, 40, 6]


# ## ERA5 into ERA5

# for i, w in enumerate(W, start=1):
#     result = compute_analogs_weighted(
#         era5_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         -1,
#     )
#     result.to_netcdf(
#         f"/homedata/pchevali/analogs/analogs_weighted_era5_to_era5_w{i}.nc"
#     )



# ## ERA5 into IPSL-CM6A-LR and ERA5

# for i, w in enumerate(W, start=1):
#     result = compute_analogs_weighted(
#         merged_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         15000,
#     )
#     result.to_netcdf(
#         f"/homedata/pchevali/analogs/analogs_weighted_era5_to_all_w{i}.nc"
#     )

# # Multiple variables with tas

# ## in ERA5

# ## in both

era5_fields = [
    zg_anom_era5.zg,
    psl_anom_era5.psl,
    tas_anom_era5.tas,
    pr10_anom_era5.pr10,
]
merged_fields = [
    zg_anom_merged.zg,
    psl_anom_merged.psl,
    tas_anom_merged.tas,
    pr10_anom_merged.pr10,
]

w = np.array([0.45, 0.45, 0.1, 0.1])
lats1 = [30, 30, 42.5, 42.5]
lats2 = [75, 75, 52.5, 52.5]
lons1 = [-30, -30, 1.5, 1.5]
lons2 = [40, 40, 6, 6]

# w = np.array([0.64, 0.28, 0.05, 0.08])
# lats1 = [30, 30, 42.5, 42.5]
# lats2 = [65, 65, 52.5, 52.5]
# lons1 = [-30, -30, 1.5, 1.5]
# lons2 = [30, 30, 6, 6]

# analogs_weighted_with_tas_era5_to_era5=compute_analogs_weighted(
#         era5_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         15000,
#         normalise="rank"
#     )

# analogs_weighted_with_tas_era5_to_era5.to_netcdf(
#     "/homedata/pchevali/analogs/analogs_weighted_with_tas_era5_to_era5_ranked.nc"
# )

# analogs_weighted_with_tas_era5_to_all=compute_analogs_weighted(
#         merged_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         15000,
#         normalise="rank"
#     )

# analogs_weighted_with_tas_era5_to_all.to_netcdf(
#     "/homedata/pchevali/analogs/analogs_weighted_with_tas_era5_to_all_ranked.nc"
# )

# analogs_weighted_with_tas_era5_to_era5=compute_analogs_weighted(
#         era5_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         15000,
#         normalise="normal"
#     )

# analogs_weighted_with_tas_era5_to_era5.to_netcdf(
#     "/homedata/pchevali/analogs/analogs_weighted_with_tas_era5_to_era5.nc"
# )

# analogs_weighted_with_tas_era5_to_all=compute_analogs_weighted(
#         merged_fields,
#         era5_fields,
#         w,
#         lats1,
#         lats2,
#         lons1,
#         lons2,
#         15000,
#         normalise="normal"
#     )

# analogs_weighted_with_tas_era5_to_all.to_netcdf(
#     "/homedata/pchevali/analogs/analogs_weighted_with_tas_era5_to_all.nc"
# )

analogs_weighted_with_tas_all_to_all_ranked=compute_analogs_weighted(
        merged_fields,
        merged_fields,
        w,
        lats1,
        lats2,
        lons1,
        lons2,
        10000,
        normalise="rank"
    )

analogs_weighted_with_tas_all_to_all.to_netcdf(
    "/homedata/pchevali/analogs/analogs_weighted_with_tas_all_to_all_ranked.nc"
)