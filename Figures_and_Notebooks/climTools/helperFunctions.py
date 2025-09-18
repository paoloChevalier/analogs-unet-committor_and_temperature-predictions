import glob
import cartopy.crs as ccrs
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from PIL import Image
from tqdm.notebook import tqdm

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

    if isinstance(index, np.int64) or isinstance(index, int):
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


def get_indep_maxima(data, n, days_in_between, lat=None, lon=None):
    if lat is not None:
        data_at_one_point = data.sel(lat=lat, lon=lon, method="nearest").sel(
            time=np.isin(decode_index(data.time, "month"), [6, 7, 8])
        )
        series = data_at_one_point.to_series()
    else:
        series = data.sel(
            time=np.isin(decode_index(data.time, "month"), [6, 7, 8])
        ).to_series()
    # Sort the series
    series = series.sort_values(ascending=False)

    res = pd.Series(
        dtype=series.dtype,
    )

    if series.index[0] // 10**8 != 0:
        for ids, val in series.items():
            date, run = index_to_date(ids)
            # Obvious first maxima case, it has to be selected
            if res.empty:
                res.at[ids] = val
            else:
                # Check the date difference with all previously selected
                if all(
                    (abs((date - index_to_date(prev_id)[0]).days) >= days_in_between)
                    or (run != index_to_date(prev_id)[1])
                    for prev_id in res.index
                ):
                    res.at[ids] = val
            if res.size == n:
                break
    else:
        for ids, val in series.items():
            date = index_to_date(ids)[0]
            # Obvious first maxima case, it has to be selected
            if res.empty:
                res.at[ids] = val
            else:
                # Check the date difference with all previously selected
                if all(
                    abs((date - index_to_date(prev_id)[0]).days) >= days_in_between
                    for prev_id in res.index
                ):
                    res.at[ids] = val
            if res.size == n:
                break
    return res


lon_corners_idf = [1.5, 3.5, 3.5, 1.5, 1.5]
lat_corners_idf = [48.0, 48.0, 49.5, 49.5, 48.0]


def plot_composite(
    field1,
    cmap_legend,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    times,
    field2=None,
    title=None,
    vmin=None,
    vmax=None,
    save=None,
    rev=False,
):
    """Plot a composite map.

    The function plots the average on given time dates of one or two fields,
    the first as filled contours on a map and the second as contours.

    Args:
        field1: A xr.DataArray for the first field
        cmap_legend: The legend for the colourmap
        lat_min: The lowest value for latitude in the extent of the plot
        lat_max: The highest value for latitude in the extent of the plot
        lon_min: The lowest value for latitude in the extent of the plot
        lon_max: The highest value for latitude in the extent of the plot
        times: The times over which the average will be computed
        field2=None: (if not None) A xr.DataArray for the second field
        title=None:  (if not None) The plot title
        vmin=None: (if not None) The lowest value for the colourmap range
        vmax=None: (if not None) The highest value for the colourmap range
        save=None: (if not None) The path to which save the figure
        rev=False: (if not None) Whether the colourmap has to be reversed

    Returns:
        Nothing.
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    if rev is True:
        field1.sel(
            time=field1.time.isin(times),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        ).mean("time").plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=30,
            cmap="RdBu",
            cbar_kwargs={"label": cmap_legend},
            vmin=vmin,
            vmax=vmax,
        )
    else:
        field1.sel(
            time=field1.time.isin(times),
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        ).mean("time").plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=30,
            cmap="RdBu_r",
            cbar_kwargs={"label": cmap_legend},
            vmin=vmin,
            vmax=vmax,
        )

    if field2 is not None:
        contours = (
            field2.sel(
                time=field2.time.isin(times),
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )
            .mean("time")
            .plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=9, colors="g")
        )
        ax.clabel(
            contours,
            fmt="%.0f",
            fontsize=8,
            colors="g",
            inline=True,
            use_clabeltext=True,
        )

    ax.coastlines()

    gl = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.plot(
        lon_corners_idf,
        lat_corners_idf,
        color="lime",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    plt.title(title)

    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_one_date(
    field1,
    cmap_legend,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    time,
    field2=None,
    title=None,
    vmin=None,
    vmax=None,
    save=None,
    rev=False,
):
    """Plot a map for a single date.

    The function plots one or two fields on a given date, the first
    as filled contours on a map and the second as contours.

    Args:
        field1: A xr.DataArray for the first field
        cmap_legend: The legend for the colourmap
        lat_min: The lowest value for latitude in the extent of the plot
        lat_max: The highest value for latitude in the extent of the plot
        lon_min: The lowest value for latitude in the extent of the plot
        lon_max: The highest value for latitude in the extent of the plot
        time: The time on which to plot
        field2=None: (if not None) A xr.DataArray for the second field
        title=None:  (if not None) The plot title
        vmin=None: (if not None) The lowest value for the colourmap range
        vmax=None: (if not None) The highest value for the colourmap range
        save=None: (if not None) The path to which save the figure
        rev=False: (if not None) Whether the colourmap has to be reversed

    Returns:
        Nothing.
    """
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    if rev is True:
        field1.sel(
            time=time, lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=30,
            cmap="RdBu",
            cbar_kwargs={"label": cmap_legend},
            vmin=vmin,
            vmax=vmax,
        )
    else:
        field1.sel(
            time=time, lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            levels=30,
            cmap="RdBu_r",
            cbar_kwargs={"label": cmap_legend},
            vmin=vmin,
            vmax=vmax,
        )

    if field2 is not None:
        contours = field2.sel(
            time=time, lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        ).plot.contour(ax=ax, transform=ccrs.PlateCarree(), levels=8, colors="g")
        ax.clabel(
            contours,
            fmt="%.0f",
            fontsize=8,
            colors="g",
            inline=True,
            use_clabeltext=True,
        )

    ax.coastlines()

    gl = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.plot(
        lon_corners_idf,
        lat_corners_idf,
        color="lime",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    plt.title(title)

    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_one_date_3fields(
    field1,
    field3,
    field4,
    cmap_legend1,
    cmap_legend2,
    cmap_legend3,
    lat_min1,
    lat_max1,
    lon_min1,
    lon_max1,
    lat_min2,
    lat_max2,
    lon_min2,
    lon_max2,
    lat_min3,
    lat_max3,
    lon_min3,
    lon_max3,
    times,
    field2=None,
    title=None,
    vmin1=None,
    vmax1=None,
    vmin2=None,
    vmax2=None,
    vmin3=None,
    vmax3=None,
    save=None,
    plot=True,
):
    """Plot 3 maps for a single date.

    The function plots three or four fields on a given date, three
    as filled contours on maps and one as contours over the first.

    Args:
        field1: A xr.DataArray for the first field
        field3: A xr.DataArray for the third field
        field4: A xr.DataArray for the fourth field
        cmap_legend1: The legend for the first colourmap
        cmap_legend2: The legend for the second colourmap
        cmap_legend3: The legend for the third colourmap
        lat_min1: The lowest value for latitude in the extent of the first plot
        lat_max1: The highest value for latitude in the extent of the first plot
        lon_min1: The lowest value for latitude in the extent of the first plot
        lon_max1: The highest value for latitude in the extent of the first plot
        lat_min2: The lowest value for latitude in the extent of the second plot
        lat_max2: The highest value for latitude in the extent of the second plot
        lon_min2: The lowest value for latitude in the extent of the second plot
        lon_max2: The highest value for latitude in the extent of the second plot
        lat_min3: The lowest value for latitude in the extent of the third plot
        lat_max3: The highest value for latitude in the extent of the third plot
        lon_min3: The lowest value for latitude in the extent of the third plot
        lon_max3: The highest value for latitude in the extent of the third plot
        times: The times over which the average will be computed
        field2=None: (if not None) A xr.DataArray for the second field
        title=None:  (if not None) The plot title
        vmin1=None: (if not None) The lowest value for the colourmap range
        vmax1=None: (if not None) The highest value for the colourmap range
        vmin2=None: (if not None) The lowest value for the colourmap range
        vmax2=None: (if not None) The highest value for the colourmap range
        vmin3=None: (if not None) The lowest value for the colourmap range
        vmax3=None: (if not None) The highest value for the colourmap range
        save=None: (if not None) The path to which save the figure
        plot=True: Tells whether the user wants to plot and not just save the images

    Returns:
        Nothing.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(19, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax1.set_extent([lon_min1, lon_max1, lat_min1, lat_max1], crs=ccrs.PlateCarree())
    field1.sel(
        time=times, lat=slice(lat_min1, lat_max1), lon=slice(lon_min1, lon_max1)
    ).plot.contourf(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        levels=30,
        cmap="RdBu_r",
        cbar_kwargs={"label": cmap_legend1},
        vmin=vmin1,
        vmax=vmax1,
        add_labels=False,
        extend="both",
    )
    if field2 is not None:
        contours = field2.sel(
            time=times, lat=slice(lat_min1, lat_max1), lon=slice(lon_min1, lon_max1)
        ).plot.contour(
            ax=ax1, transform=ccrs.PlateCarree(), levels=8, colors="g", add_labels=False
        )
        ax1.clabel(
            contours,
            fmt="%.0f",
            fontsize=8,
            colors="g",
            inline=True,
            use_clabeltext=True,
        )
    ax1.coastlines()
    gl = ax1.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax1.plot(
        lon_corners_idf,
        lat_corners_idf,
        color="lime",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
    )

    ax2.set_extent([lon_min2, lon_max2, lat_min2, lat_max2], crs=ccrs.PlateCarree())
    field3.sel(
        time=times, lat=slice(lat_min2, lat_max2), lon=slice(lon_min2, lon_max2)
    ).plot.contourf(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        levels=30,
        cmap="RdBu",
        cbar_kwargs={"label": cmap_legend2},
        vmin=vmin2,
        vmax=vmax2,
        add_labels=False,
        extend="both",
    )
    ax2.coastlines()
    gl = ax2.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax2.plot(
        lon_corners_idf,
        lat_corners_idf,
        color="lime",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    ax3.set_extent([lon_min3, lon_max3, lat_min3, lat_max3], crs=ccrs.PlateCarree())
    field4.sel(
        time=times, lat=slice(lat_min3, lat_max3), lon=slice(lon_min3, lon_max3)
    ).plot.contourf(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        levels=30,
        cmap="RdBu_r",
        cbar_kwargs={"label": cmap_legend3},
        vmin=vmin3,
        vmax=vmax3,
        add_labels=False,
        extend="both",
    )
    ax3.coastlines()
    gl = ax3.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    ax3.plot(
        lon_corners_idf,
        lat_corners_idf,
        color="lime",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    plt.suptitle(title)
    if save is not None:
        plt.savefig(save, dpi=175)
        plt.close(fig)
    if plot is True:
        plt.show()


def plot_multiple_n(
    field,
    cmap_legend,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    tasmax,
    title=None,
    vmin=None,
    vmax=None,
    save=None,
    rev=False,
):
    """Plot a field anomaly composite maps for multiple n.

    The function plots the selected field anomalies composite maps
    for different amounts (n=500,100,50,5] of extremes (lowering the threshold in a way).

    Args:
        field: A xr.DataArray for the field
        cmap_legend: The legend for the colourmap
        lat_min: The lowest value for latitude in the extent of the plot
        lat_max: The highest value for latitude in the extent of the plot
        lon_min: The lowest value for latitude in the extent of the plot
        lon_max: The highest value for latitude in the extent of the plot
        tasmax: The xr.DataArray of the temperatures
        title=None:  (if not None) The plot title
        vmin=None: (if not None) The lowest value for the colourmap range
        vmax=None: (if not None) The highest value for the colourmap range
        save=None: (if not None) The path to which save the figure
        rev=False: (if not None) Whether the colourmap has to be reversed

    Returns:
        Nothing.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = axes.flatten()
    ns = [500, 100, 50, 5]

    for n_index, (n, ax) in enumerate(zip(ns, axes)):
        extremes = get_indep_maxima(
            data=tasmax, lat=48.88, lon=2.33, n=n, days_in_between=5
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        if rev is True:
            contourf = (
                field.sel(
                    time=field.time.isin(extremes.index),
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                )
                .mean("time")
                .plot.contourf(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=30,
                    cmap="RdBu",
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False,
                )
            )
        else:
            contourf = (
                field.sel(
                    time=field.time.isin(extremes.index),
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max),
                )
                .mean("time")
                .plot.contourf(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    levels=30,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False,
                )
            )

        ax.coastlines()
        gl = ax.gridlines(
            draw_labels=True,
            crs=ccrs.PlateCarree(),
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

        ax.plot(
            lon_corners_idf,
            lat_corners_idf,
            color="lime",
            linewidth=1,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(f"n = {n:.0f}")

    # add a single colorbar for all maps
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(contourf, cax=cbar_ax, orientation="vertical")
    cbar.set_label(cmap_legend)

    plt.suptitle(title)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_corr_evolution(
    fields, extremes, lats1, lats2, lons1, lons2, labels, colours, tau_range
):
    taus = range(tau_range)
    dims = range(len(lons1))
    correlations = [
        [
            compute_mean_rank_corr_between_events(
                fields[dim],
                times=add_days_from_index(extremes.index, -tau),
                lat1=lats1[dim],
                lat2=lats2[dim],
                lon1=lons1[dim],
                lon2=lons2[dim],
            )
            for tau in taus
        ]
        for dim in dims
    ]

    for dim in dims:
        plt.plot(correlations[dim], "o-", label=labels[dim], color=colours[dim])

    # plt.title(
    #     "Evolution of the correlation for anomalies \n as we get further away before the extremes"
    # )

    plt.xlabel(r"$\tau$")
    plt.ylabel("Correlation")
    plt.axhline(y=0, color="r", linestyle="--", linewidth=0.5)
    plt.xticks(range(tau_range))
    plt.grid()
    plt.legend()
    plt.show()


def plot_rms_evolution(
    fields, extremes, lats1, lats2, lons1, lons2, labels, colours, tau_range
):
    taus = range(tau_range)
    dims = range(len(lons1))
    stds = [
        fields[i]
        .sel(lat=slice(lats1[i], lats2[i]), lon=slice(lons1[i], lons2[i]))
        .std()
        .item()
        for i in dims
    ]
    rms = [
        [
            compute_mean_rms_between_events(
                fields[dim],
                times=add_days_from_index(extremes.index, -tau),
                lat1=lats1[dim],
                lat2=lats2[dim],
                lon1=lons1[dim],
                lon2=lons2[dim],
            )
            for tau in taus
        ]
        for dim in dims
    ]
    random_days = np.random.choice(fields[0].time.values, size=200, replace=False)
    rms_global = [
        compute_mean_rms_between_events(
            fields[dim],
            times=random_days,
            lat1=lats1[dim],
            lat2=lats2[dim],
            lon1=lons1[dim],
            lon2=lons2[dim],
        )
        for dim in dims
    ]
    plt.figure(figsize=(10, 6))
    for dim in dims:
        plt.plot(
            np.array(rms[dim]) / stds[dim], "o-", label=labels[dim], color=colours[dim]
        )
        plt.axhline(
            y=rms_global[dim] / stds[dim],
            color=colours[dim],
            linestyle="--",
            linewidth=1,
            label="Global " + labels[dim],
        )

    # plt.title(
    #     "Evolution of the RMS/std for anomalies \n as we get further away before the extremes"
    # )
    plt.xlabel(r"$\tau$")
    plt.ylabel("RMS/std")
    plt.xticks(range(tau_range))
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()


def reduce_res(data, desired_res):
    """Reduce the grid resolution.

    The function reduces the grid resolution of a dataset
    by computing the number of grid points to take the mean
    over and calling xr.coarsen().

    Args:
        data: A dataset
        desired_res: The desired resolution in °

    Returns:
        data: The dataset with the new resolution.
    """
    input_grid_resolution = data.lon[1] - data.lon[0]
    weight = int(desired_res / input_grid_resolution)
    data = data.coarsen(lat=weight, lon=weight, boundary="trim").mean()
    return data


def extract_number(filename):
    """Extract the number from a filename path.

    The function extracts the number of a file of the form
    some_path/something_i.extension.

    Args:
        file: A string containing the path

    Returns:
        number: The extracted number.
    """
    # Get file name
    name = filename.split("/")[-1]
    # Get number.png part
    number_part = name.split("_")[-1]
    # Get number
    number = number_part.split(".")[0]
    return int(number)


def create_gif(folder_path, output_path, duration=2000):
    """Creates a gif.

    The function creates a gif from a series of images.

    Args:
        folder_path: A string containing the path where to find
        the images
        output_path: A string containing the path where to export
        the gif

    Returns:
        Nothing.
    """
    filenames = glob.glob(f"{folder_path}/*.png")
    filenames.sort(key=extract_number)

    # load images
    images = [Image.open(fname) for fname in filenames]

    # set durations so last frame stays longer
    durations = [duration] * (len(images) - 1) + [duration * 10]

    # save
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def compute_pairwise_rank_corr(field):
    """Computes pairwise spatial correlation between events (spearman)

    The function flattens the field to use the already implemented
    corr method from pandas.

    Args:
        field: A xr.DataArray for the field

    Returns:
        Nothing.
    """
    field_flat = field.stack(space=("lat", "lon"))
    pairwise_corr_matrix = field_flat.to_pandas().T.corr("spearman")
    return pairwise_corr_matrix


def compute_mean_rank_corr_between_events(field, times, lat1, lat2, lon1, lon2):
    """Computes mean spatial correlation between events (spearman)

    The function calls the compute_pairwise_spearman_corr
    and returns the average correlation for the selected times.

    Args:
        field: A xr.DataArray for the field
        times: The times on which to compute
        lat1: The lowest value for latitude
        lat2: The highest value for latitude
        lon1: The lowest value for latitude
        lon2: The highest value for latitude

    Returns:
        Nothing.
    """
    field = field.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2), time=times)
    corr_matrix = compute_pairwise_rank_corr(field)
    return corr_matrix.values[~np.eye(corr_matrix.shape[0],dtype=bool)].mean()


def compute_pairwise_linear_corr(field):
    """Computes pairwise spatial correlation between events (pearson)

    The function flattens the field to use the already implemented
    corr method from pandas.

    Args:
        field: A xr.DataArray for the field

    Returns:
        Nothing.
    """
    field_flat = field.stack(space=("lat", "lon"))
    pairwise_corr_matrix = field_flat.to_pandas().T.corr("pearson")
    return pairwise_corr_matrix


def compute_mean_linear_corr_between_events(field, times, lat1, lat2, lon1, lon2):
    """Computes mean spatial correlation between events (pearson)

    The function calls the compute_pairwise_spearman_corr
    and returns the average correlation for the selected times.

    Args:
        field: A xr.DataArray for the field
        times: The times on which to compute
        lat1: The lowest value for latitude
        lat2: The highest value for latitude
        lon1: The lowest value for latitude
        lon2: The highest value for latitude

    Returns:
        Nothing.
    """
    field = field.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2), time=times)
    corr_matrix = compute_pairwise_linear_corr(field)
    return corr_matrix[~np.eye(corr_matrix.shape[0])].mean()


def compute_pairwise_rmse_fast_matmul(data1, data2):
    """Computes the pairwise rms between every times of two datasets

    The function computes the pairwise rms between every times of two datasets,
    can take a in memory array or dask backed array lazily.

    Args:
        data1: A np.array[time,space]
        data2: A np.array[time,space]

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



def compute_mean_rms_between_events(field, times, lat1, lat2, lon1, lon2):
    field = field.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2), time=times).stack(space=("lat", "lon"))
    pairwise_rmse = compute_pairwise_rmse_fast_matmul(field.data, field.data)
    return pairwise_rmse[~np.eye(pairwise_rmse.shape[0],dtype=bool)].mean()


def compute_mean_rms_between_events_dask(
    field, times, lat1, lat2, lon1, lon2, chunk_size=300
):
    field_da = field.sel(
        time=times, lat=slice(lat1, lat2), lon=slice(lon1, lon2)
    ).chunk({"time": chunks_size})
    pairwise_rmse = compute_pairwise_rmse_fast2(field_da, field_da)
    mask = da.triu(da.ones(pairwise_rmse.shape), k=1).astype(bool)
    return pairwise_rmse[mask].mean()

def CCC(data1, data2):
    """Computes Lin's correlation coefficient.

    The function computes Lin's concordance correlation coefficient
    between two datarray.

    Args:
        data1: The first array (fixed date)
        data2: The second array (fixed date)

    Returns:
        The CCC.
    """
    corr = xs.pearson_r(data1, data2).values
    std1 = data1.std().values
    std2 = data2.std().values
    mds = (data1.mean() - data2.mean()).values ** 2
    lin = (2 * corr * std1 * std2) / (mds + std1**2 + std2**2)
    return lin