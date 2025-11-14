import xarray
import numpy as np
import cftime
import numba as nb
import datetime
from hdp.utils import get_version, add_history
from tqdm.auto import tqdm
import dask.array as da


@nb.njit
def index_heatwaves(
    hot_days_ts: np.ndarray, min_duration: int, max_break: int, max_subs: int
) -> np.ndarray:
    """
    Identifies the heatwaves in the timeseries using the specified heatwave definition

    :param hot_days_ts: Integer array of ones and zeros where ones indicates a hot day
    :type hot_days_ts: np.ndarray
    :param min_duration: The minimum number of hot days to constitute a heatwave event, including after breaks
    :type min_duration: int
    :param max_break: The maximum number of days between hot days within one heatwave event
    :type max_break: int
    :param max_subs: the maximum number of subsequent events allowed to be apart of the initial consecutive hot days
    :type max_subs: int
    :return: Timeseries where nonzero integers indicate heatwave indices for each timestep
    :rtype: np.ndarray
    """
    ts = np.zeros(hot_days_ts.size + 2, dtype=nb.int64)
    for i in range(0, hot_days_ts.size):
        if hot_days_ts[i]:
            ts[i + 1] = 1
    diff_ts = np.diff(ts)
    diff_indices = np.where(diff_ts != 0)[0]

    in_heatwave = False
    current_hw_index = 0
    sub_events = 0
    hw_indices = np.zeros(diff_ts.size, dtype=nb.int64)

    for i in range(diff_indices.size - 1):
        index = diff_indices[i]
        next_index = diff_indices[i + 1]

        if (
            diff_ts[index] == 1
            and next_index - index >= min_duration
            and not in_heatwave
        ):
            current_hw_index += 1
            in_heatwave = True
            hw_indices[index:next_index] = current_hw_index
        elif diff_ts[index] == -1 and next_index - index > max_break:
            in_heatwave = False
        elif diff_ts[index] == 1 and in_heatwave and sub_events < max_subs:
            sub_events += 1
            hw_indices[index:next_index] = current_hw_index
        elif diff_ts[index] == 1 and in_heatwave and sub_events >= max_subs:
            if next_index - index >= min_duration:
                current_hw_index += 1
                hw_indices[index:next_index] = current_hw_index
            else:
                in_heatwave = False
            sub_events = 0

    return hw_indices[0 : hw_indices.size - 1]


@nb.njit
def heatwave_number(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Measures the number of heatwaves (by event, not days) in each season of a given heatwave index time series.
    Heatwave metric, commonly abbreviated as HWN.

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Number of heatwaves per heatwave season
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        uniques = np.unique(hw_ts[end_points[0] : end_points[1]])
        output[y] = uniques[uniques != 0].size

    return output


@nb.njit
def heatwave_frequency(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Measures the number of heatwave days in each season of a given heatwave index time series.
    Heatwave metric, commonly abbreviated as HWF.

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Number of heatwave days per heatwave season.
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        output[y] = np.sum(hw_ts[end_points[0] : end_points[1]] > 0, dtype=nb.int64)
    return output


@nb.njit
def heatwave_duration(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Measures the length of the longest heatwave in each season of a given heatwave index time series.
    Heatwave metric, commonly abbreviated as HWD.

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Length of longest heatwave per heatwave season.
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.int64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        hw_ts_slice = hw_ts[end_points[0] : end_points[1]]
        unique_indices = np.unique(hw_ts_slice)

        if unique_indices.size == 1:
            output[y] = 0
        else:
            unique_indices = unique_indices[1:]

        hw_lengths = np.zeros(unique_indices.size, dtype=nb.int64)
        for index, value in enumerate(unique_indices):
            if value != 0:
                for day in hw_ts_slice:
                    if day == value:
                        hw_lengths[index] += 1

        output[y] = np.max(hw_lengths)
    return output


@nb.njit
def heatwave_average(hw_ts: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Measures the average length of all heatwaves in each season of a given heatwave index time series.
    Heatwave metric, commonly abbreviated as HWA.

    :param hw_ts: Integer timeseries of indexed heatwave days.
    :type hw_ts: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Average heatwave length per heatwave season.
    :rtype: np.ndarray
    """
    output = np.zeros(season_ranges.shape[0], dtype=nb.float64)
    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        hw_ts_slice = hw_ts[end_points[0] : end_points[1]]
        unique_indices = np.unique(hw_ts_slice)

        if unique_indices.size == 1:
            output[y] = 0
        else:
            unique_indices = unique_indices[1:]

        hw_lengths = np.zeros(unique_indices.size, dtype=nb.int64)
        for index, value in enumerate(unique_indices):
            if value != 0:
                for day in hw_ts_slice:
                    if day == value:
                        hw_lengths[index] += 1

        output[y] = np.mean(hw_lengths)
    return output


@nb.njit
def heatwave_avi(hw_ts_intensity: np.ndarray, season_ranges: np.ndarray) -> np.ndarray:
    """
    Measures the average max temperature of heatwave days in each season of a given heatwave index time series.
    Average Heatwave Intensity, commonly abbreviated as AVI.

    :param hw_ts_intensity: Timeseries of heat measurement which is zero on non-heatwave days.
    :type hw_ts_intensity: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :return: Average Heatwave Intensity per heatwave season.
    :rtype: np.ndarray
    """
    # output of size n_years
    output = np.zeros(season_ranges.shape[0], dtype=nb.float64)

    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        hw_ts_slice = hw_ts_intensity[end_points[0] : end_points[1]]

        # unique_indices = np.unique(hw_ts_slice)
        # if unique_indices.size == 1:  # if there are no heatwave days
        if np.isnan(hw_ts_slice).all():
            output[y] = 0
        else:
            # ignores the masked values
            output[y] = np.nanmean(hw_ts_slice)
    return output


@nb.njit
def heatwave_ava(
    hw_ts_intensity: np.ndarray,
    season_ranges: np.ndarray,
    threshold: np.ndarray,
    doy_map: np.ndarray,
) -> np.ndarray:
    """
    Measures the average intensity anomaly of heatwave days in each season of a given heatwave index time series. Anomalies are taken with respect to a threshold.
    Average Heatwave Anomaly, commonly abbreviated as AVA.

    :param hw_ts_intensity: Timeseries of heat measurement which is zero is non-heatwave days.
    :type hw_ts_intensity: np.ndarray
    :param season_ranges: Range of array indices, corresponding to heatwave season, in indexed heatwave day timeseries to count.
    :type season_ranges: np.ndarray
    :param threshold: Threshold for extreme heat.
    :type threshold: np.ndarray
    :param doy_map: Mapping from index in times array to day of year value.
    :type doy_map: np.ndarray
    :return: Average Heatwave Intensity per heatwave season.
    :rtype: np.ndarray
    """
    # output of size n_years
    output = np.zeros(season_ranges.shape[0], dtype=np.float64)

    for y in range(season_ranges.shape[0]):
        end_points = season_ranges[y]
        hw_ts_slice = hw_ts_intensity[end_points[0] : end_points[1]]

        # find the corresponding doy for hw_ts_slice
        doy_slice = doy_map[end_points[0] : end_points[1]]
        # threshold_slice = threshold[doy_slice[0] : doy_slice[-1] + 1]
        threshold_slice = np.array([threshold[d] for d in doy_slice])

        # unique_indices = np.unique(hw_ts_slice)
        # if unique_indices.size == 1:  # if there are no heatwave days
        if np.isnan(hw_ts_slice).all():
            output[y] = 0
        else:
            hw_ts_anom_slice = hw_ts_slice - threshold_slice
            # ignores the masked values
            output[y] = np.nanmean(hw_ts_anom_slice)
    return output


def get_range_indices(times: np.ndarray, start: tuple, end: tuple) -> np.ndarray:
    """
    Calculates the range of time indices to define each heatwave season for a given time series.
    This function is agnostic to the calendar type, but will not yield accurate results if the month and day exceed the calendar's definition.
    The ranges of indices are then used to slice the time series into the heatwave seasons, this is faster than iterating through CFTime objects.

    :param times: Array of CFTime objects corresponding to the time series to define the seasons over.
    :type times: np.ndarray
    :param start: Tuple of starting month integer and day integer to compare times array against.
    :type start: tuple
    :param end: Tuple of ending month integer and day integer to compare times array against.
    :type end: tuple
    :return: Range of indices in times array for each season.
    :rtype: np.ndarray
    """
    num_years = times[-1].year - times[0].year + 1
    ranges = np.zeros((num_years, 2), dtype=int) - 1

    n = 0
    looking_for_start = True
    for t in range(times.shape[0]):
        if looking_for_start:
            if times[t].month == start[0] and times[t].day == start[1]:
                looking_for_start = False
                ranges[n, 0] = t
        else:
            if times[t].month == end[0] and times[t].day == end[1]:
                looking_for_start = True
                ranges[n, 1] = t
                n += 1

    if not looking_for_start:
        ranges[-1, -1] = times.shape[0]

    return ranges


def compute_hemisphere_ranges(
    measure: xarray.DataArray, start=(5, 1), end=(10, 1)
) -> xarray.DataArray:
    """
    Computes the heatwave season ranges by time index (not the timestamp, rather the index corresponding to the timestamp) for each grid cell based on whether it is in the Northern Hemisphere (boreal summer, May 1st to October 1st) or Southern Hemisphere (austral summer, November 1st to March 1st).

    :param measure: DataArray containing 'lat' and 'lon' variables corresponding to grid.
    :type measure: xarray.DataArray
    :param start: a (month, day) tuple for boreal season
    :param end: same as stary
    :return: Generates seasonal ranges by hemisphere for an arbitrary 'lat'-'lon' grid.
    :rtype: xarray.DataArray
    """

    north_ranges = get_range_indices(measure.time.values, start, end)
    south_start = ((start[0] + 6 - 1) % 12 + 1, 1)
    south_end = ((end[0] + 6 - 1) % 12 + 1, 1)
    south_ranges = get_range_indices(
        measure.time.values,
        south_start,
        south_end,
    )

    # find "complete years" ------------------
    # this chunk was in the original function
    # i think it's bugged out and odens't doing anything
    # because north_ranges.size should be len(north_ranges)...
    slice_start = 0  # will store the index of the first complete year
    # slice_end = north_ranges.size # this was the og code, broken I think
    slice_end = len(north_ranges)  # will the index of the last complete year
    start_indentified = False  # flag for whether we've found the first complete year

    for year_index, n_end_points in enumerate(north_ranges):
        # combine north and south
        end_points = np.concatenate([n_end_points, south_ranges[year_index]])

        # if the first year isn't complete, then skip this year and try again
        if -1 in end_points and not start_indentified:
            slice_start = year_index
            continue
        # if the first year is complete, then we've found the start
        elif not start_indentified:
            start_indentified = True

        # if we've found the start and the last year isn't complete, then
        # end at the year before (i.e, the last complete year)
        if start_indentified and -1 in end_points:
            # slice_end = year_index # this was the old code, but I replaced this functionality below
            slice_end = year_index + 1
            break

    # if southern hemipshere spans multiple years (e.g. DJF)
    # then we will use the year from last month (i.e. December 1999 -> Feb 2000 will be DJF for year 2000)
    # which means... that we won't have data for the the first year of data
    # e.g. if 1999 was the first year of data, then we don't have data for DJF year 1999 (bc we don't have Dec 1998)

    # if southern hemisphere spans multiple years, offset  by 1 year
    if south_end < south_start:
        slice_start_north = slice_start + 1
        slice_start_south = slice_start

        slice_end_north = slice_end
        slice_end_south = slice_end - 1
    else:
        slice_start_north = slice_start
        slice_start_south = slice_start
        slice_end_north = slice_end
        slice_end_south = slice_end

    # # we know that the last year might be short, so let's fix that explicitly
    # is_last_year_short_south = num_days_per_year_south[-1] < num_days_per_year_south[-2]
    # is_last_year_short_north = num_days_per_year_north[-1] < num_days_per_year_north[-2]
    # if is_last_year_short_north or is_last_year_short_south:
    #     slice_end = slice_end - 1

    # subset to complete years
    north_ranges = north_ranges[slice_start_north:slice_end_north]
    south_ranges = south_ranges[slice_start_south:slice_end_south]

    # check that all years have the same number of days
    num_days_per_year_north = [endpoint[1] - endpoint[0] for endpoint in north_ranges]
    num_days_per_year_south = [endpoint[1] - endpoint[0] for endpoint in south_ranges]
    assert len(set(num_days_per_year_north)) == 1, (
        f"not all years in the northern hemisphere have the same number of days. here are the possible number of days: {set(num_days_per_year_north)}"
    )
    assert len(set(num_days_per_year_south)) == 1, (
        f"not all years in the southern hemisphere have the same number of days. here are the possible number of days: {set(num_days_per_year_south)}"
    )

    years = np.arange(measure.time.values[0].year, measure.time.values[-1].year + 1, 1)
    years = years[slice_start_north:slice_end_north]

    ranges = (
        np.zeros(
            (north_ranges.shape[0], 2, measure.lat.size, measure.lon.size), dtype=int
        )
        - 1
    )

    for i in range(measure.lat.size):
        for j in range(measure.lon.size):
            if measure.lat.values[i] < 0:
                ranges[:, :, i, j] = south_ranges
            else:
                ranges[:, :, i, j] = north_ranges

    return xarray.DataArray(
        data=ranges,
        dims=["year", "end_points", "lat", "lon"],
        coords={
            "year": years,
            "end_points": ["start", "finish"],
            "lat": measure.lat.values,
            "lon": measure.lon.values,
        },
    )


def get_range_indices_doy(measure, start: int, end: int) -> np.ndarray:
    """
    same as get_range_indices, but based on day of year start and end, instead of (month, day)
    :param measure: DataArray containing 'lat' and 'lon' variables corresponding to grid.
    :param start: int between 1 and 365. the day of year that we want to start with.
    """

    doys = measure.time.dt.dayofyear.values
    times = measure.time.values
    num_years = times[-1].year - times[0].year + 1
    ranges = np.zeros((num_years, 2), dtype=int) - 1

    n = 0  # n indexes over years
    looking_for_start = True
    for t in range(doys.shape[0]):  # for each day
        if looking_for_start:
            if doys[t] == start:  # if first day, then
                looking_for_start = False
                ranges[n, 0] = t  # set start for year to this day
        else:  # if we already know the start
            if doys[t] == end:  # if last day
                looking_for_start = True
                ranges[n, 1] = t  # set end for year n to this day
                n += 1

    if not looking_for_start:
        ranges[-1, -1] = times.shape[0]

    return ranges


def compute_hemisphere_ranges_doy(
    measure: xarray.DataArray, doy_mask: xarray.DataArray
) -> xarray.DataArray:
    """
    replcement for compute_hemisphere_ranges for doy-masks instead of (month, day) start and end.
    :param doy_mask: a (dayofyear, lat) mask that is 1 for summer days and 0 otherwise.
    """

    assert measure.lat.size == doy_mask.lat.size, (
        "measure and mask don't have same lat size"
    )
    assert np.isclose(measure.lat.values, doy_mask.lat.values).all(), (
        "measure and mask don't match lats"
    )

    # ranges has dimensions (year, [start, end], lat, lon)
    ranges_array = (
        np.zeros(
            (
                # time is -1 bc no austral summer in first year.
                np.unique(measure.time.dt.year).size - 1,
                2,
                measure.lat.size,
                measure.lon.size,
            ),
            dtype=int,
        )
        - 1
    )
    for i in range(measure.lat.size):
        # indices of doys that make up summer
        lat_i_mask = doy_mask.isel(lat=i).values
        # doys that make up summer
        doy_lat_mask = doy_mask.dayofyear.where(lat_i_mask == 1).values

        ## add logic for austral summers, which wrap around.
        # unique doys that make up summer, without NA
        doys = doy_lat_mask[~np.isnan(doy_lat_mask)]
        doy_jumps = np.diff(doys)  # check for jumps
        # if there's a jump (i.e. NAs in the middle of the year)
        if np.max(doy_jumps) > 1:
            # the year starts right after all the nans
            first_doy = doys[np.argmax(doy_jumps) + 1]
            last_doy = doys[np.argmax(doy_jumps)]
        else:
            # in boreal summer, the doys are consecutive
            first_doy = doys[0]
            last_doy = doys[-1]

        ranges_lat = get_range_indices_doy(measure, first_doy, last_doy)

        slice_start = 0
        slice_end = ranges_lat.size
        start_indentified = False
        for year_index, end_points in enumerate(ranges_lat):  # for each year
            if (
                -1 in end_points and not start_indentified
            ):  # if we haven't found the start yet
                slice_start = year_index  # slice_start indexes years
                continue
            elif not start_indentified:
                start_indentified = True

            if start_indentified and -1 in end_points:
                slice_end = year_index
                break

        ranges_lat = ranges_lat[
            slice_start:slice_end
        ]  # get the corresponding time index slices

        # remove the last year if it doesn't have 90 days. This can happen if we're using austral summer
        # i.e. if DJF represent a year,  then we record this as the summer of Jan/Feb. So the last calendar year will miss out.
        last_year_duration = ranges_lat[-1][1] - ranges_lat[-1][0]

        # if fewer than 90 days in the last year, remove last year
        if last_year_duration != 90:
            ranges_lat = ranges_lat[:-1, :]
        else:  # if boreal summer
            # remove first year (to keep alignment with nh)
            ranges_lat = ranges_lat[1:, :]

        for j in range(measure.lon.size):
            ranges_array[:, :, i, j] = ranges_lat

    # # remove the last year if it doesn't have 90 days. This can happen if we're using austral summer
    # # i.e. if DJF represent a year,  then we record this as the summer of Jan/Feb. So the last calendar year will miss out.
    # last_year_durations = ranges_array[-1, 1, :, 0] - ranges_array[-1, 0, :, 0]

    # # if any lats have fewer than 90 days in the last year
    # if (last_year_durations != 90).any():
    #     # remove last year of data
    #     ranges_array = ranges_array[:-1, :, :, :]

    # check that all years are 90 days
    assert (
        np.array(
            [
                ranges_array[t, 1, :, 0] - ranges_array[t, 0, :, 0]
                for t in np.arange(ranges_array.shape[0])
            ]
        )
        == 90
    ).all(), "not all seasons are 90 days long"

    # mark the year as the year of the *endpoint*, not the start point -------------------------------------
    # this matter for austral summer. i.e. if DJF, then the year corresponding to February should be recorded.
    smallest_endpoint = np.max(ranges_array[0, 1, :, 0])
    largest_endpoint = np.min(ranges_array[-1, 1, :, 0])
    years = np.arange(
        measure.time.values[smallest_endpoint].year,
        measure.time.values[largest_endpoint].year + 1,
        1,
    )
    years = years[slice_start:slice_end]

    start_years = np.array(
        [measure.time.values[ind].year for ind in ranges_array[0, 1, :, 0]]
    )
    end_years = np.array(
        [measure.time.values[ind].year for ind in ranges_array[-1, 1, :, 0]]
    )
    assert np.unique(start_years).size == 1, (
        "check start years, esp hemisphere weirdness"
    )
    assert np.unique(end_years).size == 1, "check end years, esp hemisphere weirdness"

    ranges_xr = xarray.DataArray(
        data=ranges_array,
        dims=["year", "end_points", "lat", "lon"],
        coords={
            "year": years,
            "end_points": ["start", "finish"],
            "lat": measure.lat.values,
            "lon": measure.lon.values,
        },
    )

    # return value
    return ranges_xr


def build_doy_map(times: np.ndarray) -> np.ndarray:
    """
    Maps the time series index (key) to its respective day of the year (value).

    :param measure: Array of CFTime objects.
    :type measure: np.ndarray
    :return: Day of year for each timestep.
    :rtype: np.ndarray
    """
    doy_map = np.zeros(times.size, dtype=int) - 1
    for time_index, time in enumerate(times):
        doy_map[time_index] = time.dayofyr - 1
    return doy_map


@nb.njit
def indicate_hot_days(
    measure: np.ndarray, threshold: np.ndarray, doy_map: np.ndarray
) -> np.ndarray:
    """
    Determines whether each time step in a heat measure exceeds the threshold.

    :param measure: Heat measure to compare against the threshold.
    :type measure: np.ndarray
    :param threshold: Threshold for extreme heat.
    :type threshold: np.ndarray
    :param doy_map: Mapping from index in times array to day of year value.
    :type doy_map: np.ndarray
    :return: Boolean array of days in measure that exceed the threshold.
    :rtype: np.ndarray
    """
    output = np.zeros(measure.shape, dtype=nb.boolean)
    for t in range(measure.size):
        doy = doy_map[t]
        if measure[t] > threshold[doy]:
            output[t] = True
        else:
            output[t] = False
    return output


@nb.njit
def compute_heatwave_metrics(
    measure: np.ndarray,
    threshold: np.ndarray,
    doy_map: np.ndarray,
    min_duration: int,
    max_break: int,
    max_subs: int,
    season_ranges: np.ndarray,
) -> np.ndarray:
    """
    Computes HWN, HWF, HWD, and HWA metrics for a given measure, threshold, and definition. Additional parameters can be used to fine tune the analysis.
    This is the Numba-compiled function that is parallelized with Dask and formatted into a more user-friendly format by hdp.metrics.compute_individual_metrics

    :param measure: Heat measure to compare against the threshold.
    :type measure: np.ndarray
    :param threshold: Threshold for extreme heat.
    :type threshold: np.ndarray
    :param doy_map: Mapping from index in times array to day of year value.
    :type doy_map: np.ndarray
    :param min_duration: Minimum number of days that exceed the threshold to constitute a heatwave.
    :type min_duration: int
    :param max_break: Maximum number of days below the threshold that can exist in a heatwave.
    :type max_break: int
    :param max_subs: Maximum number of subsequent heatwaves.
    :type max_subs: int
    :param season_ranges:
    :type season_ranges: np.ndarray
    :return: Array of equal number of dimensions to measure and an additional dimension containing each heatwave metric: HWN, HWF, HWD, HWA
    :rtype: np.ndarray
    """
    hot_days_ts = indicate_hot_days(measure, threshold, doy_map)
    hw_ts = index_heatwaves(hot_days_ts, min_duration, max_break, max_subs)
    hwf = heatwave_frequency(hw_ts, season_ranges)
    hwn = heatwave_number(hw_ts, season_ranges)
    hwd = heatwave_duration(hw_ts, season_ranges)
    hwa = heatwave_average(hw_ts, season_ranges)

    # add intensity metrics
    # hw_ts_intensity = np.ma.masked_where(hw_ts == 0, measure) # numba doesn't support ma
    hw_ts_intensity = np.where(hw_ts == 0, np.nan, measure)
    hw_avi = heatwave_avi(hw_ts_intensity, season_ranges)
    hw_ava = heatwave_ava(hw_ts_intensity, season_ranges, threshold, doy_map)

    output = np.zeros((6,) + hwf.shape, dtype=nb.float64)
    output[0] = hwf
    output[1] = hwn
    output[2] = hwd
    output[3] = hwa

    output[4] = hw_avi
    output[5] = hw_ava
    return output


def compute_heatwave_metrics_wrapper(
    measure, threshold, doy_map, hw_definitions, season_ranges
):
    # measure will be a "chunk" from xr.map_blocks. Subset to the lat/lon of this chunk.
    # i think the .sel operators aren't thread safe for dask, so make sure to use single threaded.
    threshold = threshold.sel(lat=measure.lat, lon=measure.lon)
    season_ranges = season_ranges.sel(lat=measure.lat, lon=measure.lon)
    def_coords = xarray.DataArray(
        [f"{hw_def[0]}-{hw_def[1]}-{hw_def[2]}" for hw_def in hw_definitions],
        dims=["definition"],
    )
    perc_coords = xarray.DataArray(threshold.percentile.values, dims=["percentile"])

    perc_datasets = []
    for perc in threshold.percentile.values:
        def_datasets = []
        for hw_def in hw_definitions:
            metric_data = xarray.apply_ufunc(
                compute_heatwave_metrics,
                measure,
                threshold.sel(percentile=perc),
                doy_map,
                hw_def[0],
                hw_def[1],
                hw_def[2],
                season_ranges,
                vectorize=True,  # nh temp change
                dask="parallelized",  # nh temp change
                # dask="allowed",
                input_core_dims=[
                    ["time"],
                    ["doy"],
                    ["time"],
                    [],
                    [],
                    [],
                    ["year", "end_points"],
                ],
                output_core_dims=[["metric", "year"]],
                output_dtypes=[float],
                dask_gufunc_kwargs=dict(output_sizes=dict(metric=6)),
            )
            def_datasets.append(metric_data)
        perc_datasets.append(xarray.concat(def_datasets, dim=def_coords))
    return xarray.concat(perc_datasets, dim=perc_coords)


def compute_individual_metrics(
    measure: xarray.DataArray,
    threshold: xarray.DataArray,
    hw_definitions: list,
    use_doy=False,
    doy_mask=None,
    include_threshold: bool = True,
    check_variables: bool = True,
    start=(5, 1),
    end=(10, 1),
) -> xarray.Dataset:
    """
    Computes HWN, HWF, HWD, and HWA heatwave metrics for an individual parameter configuration of measure, threshold, and definition.

    Heatwave definitions are described by tuples of three integers in the following order:
    1. Minimum number of days exceeding the threshold to define the start of a heatwave
    2. Maximum number of days following the start of a heatwave that do not exceed the threshold and are followed by days that do exceed the threshold. In other words, the maximum "break" that a heatwave can have between the initial hot days and some number of hot days afterwards.
    3. Maximum number of breaks in a heatwave (can also be thought of as the maximum number of secondary events after the initial hot days).

    :param measure: Formatted HDP measure DataArray
    :type measure: xarray.DataArray
    :param threshold: Formatted HDP threshold compatable with the given measure
    :type threshold: xarray.DataArray
    :param hw_definitions: Heatwave definitions to calculate metrics for. See the function description for how to generate definitions.
    :type hw_definitions: list[tuple]
    :param use_doy: flag for whether summer season is defined as (month,day) or dayofyear.
    :param doy_mask: ignored if use_doy is False. see compute_hemipshere_ranges_doy.
    :param include_threshold: (Optional) Whether or not to include the threshold DataArray in the aggregated output dataset. Default is True.
    :type include_threshold: bool
    :param check_variables: (Optional) Whether or not to check if measure is compatable with the threshold. Default is True.
    :type check_variables: bool
    :return: Aggregate dataset containing all of the heatwave metrics and optional datasets.
    :rtype: xarray.Dataset
    """
    if check_variables:
        assert "hdp_type" in threshold.attrs
        assert threshold.attrs["hdp_type"] == "threshold"
        assert (
            threshold.attrs["baseline_variable"] == measure.attrs["baseline_variable"]
        )
        assert threshold.attrs["baseline_calendar"] == measure.time.values[0].calendar

    combined_history = ""
    if "history" in measure.attrs:
        for entry in measure.attrs["history"].split("\n"):
            if entry != "":
                combined_history += f"(Measure) {entry}\n"
    if "history" in threshold.attrs:
        for entry in threshold.attrs["history"].split("\n"):
            if entry != "":
                combined_history += f"(Threshold) {entry}\n"

    if use_doy:
        assert doy_mask is not None, "doy_mask needed if use_doy = True"
        season_ranges = compute_hemisphere_ranges_doy(measure, doy_mask)
    else:
        season_ranges = compute_hemisphere_ranges(measure, start=start, end=end)

    times = measure.time.values
    doy_map = xarray.DataArray(data=build_doy_map(times), coords={"time": times})

    da_dims = ["percentile", "definition"]
    da_shape = [threshold.percentile.size, len(hw_definitions)]
    da_chunks = [(threshold.percentile.size), (len(hw_definitions))]

    for index, dim in enumerate(measure.dims):
        if dim != "time":
            da_dims.append(dim)
            da_shape.append(measure.shape[index])
            da_chunks.append(measure.chunks[index])

    da_dims.extend(["metric", "year"])
    da_shape.extend([6, season_ranges.year.size])
    da_chunks.extend([(6), (season_ranges.year.size)])

    da_coords = {**measure.coords}
    da_coords.pop("time", None)
    da_coords["year"] = season_ranges.year.values
    da_coords["definition"] = [
        f"{hw_def[0]}-{hw_def[1]}-{hw_def[2]}" for hw_def in hw_definitions
    ]
    da_coords["percentile"] = threshold.percentile.values

    template = xarray.DataArray(
        da.random.random(da_shape, chunks=da_chunks), dims=da_dims, coords=da_coords
    )

    metric_data = xarray.map_blocks(
        compute_heatwave_metrics_wrapper,
        obj=measure,
        args=[threshold, doy_map],
        kwargs={"hw_definitions": hw_definitions, "season_ranges": season_ranges},
        template=template,
    )

    ds = xarray.Dataset(
        dict(
            HWF=metric_data.sel(metric=0),
            HWN=metric_data.sel(metric=1),
            HWD=metric_data.sel(metric=2),
            HWA=metric_data.sel(metric=3),
            AVI=metric_data.sel(metric=4),
            AVA=metric_data.sel(metric=5),
        )
    )

    start_ts = cftime.datetime(
        ds.year[0], 1, 1, calendar=measure.time.values[0].calendar
    )
    end_ts = cftime.datetime(
        ds.year[-1], 1, 1, calendar=measure.time.values[0].calendar
    )
    ds = ds.rename(dict(year="time")).assign_coords(
        dict(time=xarray.cftime_range(start_ts, end_ts, periods=ds.year.size))
    )

    ds.attrs |= {
        "description": f"Heatwave metric dataset generated by Heatwave Diagnostics Package (HDP v{get_version()})",
        "hdp_version": get_version(),
        "hdp_type": "metric",
    }

    ds["HWF"].attrs |= {
        "units": "heatwave days",
        "long_name": "Heatwave Frequency",
        "description": "Number of days that fall within heatwave during a heatwave season",
    }
    ds["HWD"].attrs |= {
        "units": "heatwave days",
        "long_name": "Heatwave Duration",
        "description": "Length of longest heatwave during a heatwave season",
    }
    ds["HWN"].attrs |= {
        "units": "heatwave events",
        "long_name": "Heatwave Number",
        "description": "Number of distinct heatwaves during a heatwave season",
    }
    ds["HWA"].attrs |= {
        "units": "heatwave events",
        "long_name": "Heatwave Average",
        "description": "Average length of heatwaves during a heatwave season",
    }
    ds["AVI"].attrs |= {
        "units": "avg intensity",
        "long_name": "Heatwave Average Intensity",
        "description": "Average daily intensity of heatwaves days during a heatwave season",
    }
    ds["AVA"].attrs |= {
        "units": "avg anomalies",
        "long_name": "Heatwave Average Intensity Anomalies",
        "description": "Average daily intensity anomalies of heatwaves during a heatwave season",
    }
    ds["percentile"].attrs |= {"range": "(0, 1)"}
    ds["definition"].attrs |= {
        "first_number": "Minimum number of consecutively hot days",
        "second_number": "Maximum number of break days after first wave",
        "third_number": "Minimum number of consecutively hot days after the break",
    }

    for variable in ds:
        ds[variable].attrs["history"] = combined_history
        add_history(ds[variable], f"Heatwave metrics generated by HDP v{get_version()}")

    return ds


def compute_group_metrics(
    measures: xarray.Dataset,
    thresholds: xarray.Dataset,
    hw_definitions: list,
    use_doy=False,
    doy_mask=None,
    include_threshold: bool = False,
    check_variables: bool = True,
    start=(5, 1),
    end=(10, 1),
) -> xarray.Dataset:
    metric_sets = []
    for measure_name in list(measures.keys()):
        measure = measures[measure_name]
        for threshold_name in list(thresholds.keys()):
            threshold = thresholds[threshold_name]
            if (
                threshold.attrs["baseline_variable"]
                == measure.attrs["baseline_variable"]
            ):
                hw_metrics = compute_individual_metrics(
                    measure,
                    threshold,
                    hw_definitions,
                    use_doy,
                    doy_mask,
                    include_threshold,
                    check_variables,
                    start,
                    end,
                )
                var_renames = {
                    name: f"{measure_name}.{threshold_name}.{name}"
                    for name in list(hw_metrics.keys())
                }
                metric_sets.append(hw_metrics.rename(var_renames))

    aggr_ds = xarray.merge(metric_sets)
    aggr_ds.attrs["variable_naming_desc"] = (
        "(heat measure).(threshold used).(heatwave metric)"
    )
    aggr_ds.attrs["variable_naming_delimeter"] = "."
    return aggr_ds


def compute_metrics_io(
    output_path: str,
    measure_path: str,
    measure_var: str,
    threshold_path: str,
    hw_definitions: list,
    use_doy=False,
    doy_mask=None,
    include_threshold: bool = False,
    override_threshold_var: str = None,
) -> None:
    """
    Computes heatwave metrics from path inputs instead of manually supplied xarray Datasets/DataArrays (automates reading from and writing to disk).
    Resulting metrics are written directly to disk instead of holding in memory.

    :param output_path: Path to write dataset(s) to, can be a zarr store (faster) or netCDF file (slower).
    :type output_path: str
    :param measure_path: Path to heat measure dataset formatted by the HDP.
    :type measure_path: str
    :param measure_var: Name of measure variable to use from specified dataset.
    :type measure_var: str
    :param threshold_path: Path to threshold dataset formatted by the HDP.
    :type threshold_path: str
    :param hw_definitions: Definitions to compute heatwave metrics over.
    :type hw_definitions: list
    :param include_threshold: (Optional) Whether or not to include the threshold dataset in the resulting output. Default is False.
    :type include_threshold: bool
    :param override_threshold_var: (Optional) Override threshold variable to use when computing metrics. If left unspecified, the format "threshold_{measure_var}" will be used.
    :type override_threshold_var: str
    :return: None
    :rtype: None
    """
    output_path = Path(output_path)
    measure_path = Path(measure_path)
    threshold_path = Path(threshold_path)
    check_variables = True

    if override_threshold_var is None:
        threshold_var = f"threshold_{measure_var}"
        check_variables = False

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Overwrite parameter set to False and file exists at '{output_path}'."
        )

    if not output_path.parent.exists():
        if overwrite:
            makedirs(output_path)
        else:
            raise FileExistsError(
                f"Overwrite parameter set to False and directory '{output_path.parent}' does not exist."
            )

    if output_path.suffix not in [".zarr", ".nc"]:
        raise ValueError(
            f"File type '{output_path.suffix}' from '{output_path}' not supported."
        )

    if measure_path.suffix == ".zarr" and measure_path.isdir():
        measure_data = xarray.open_zarr(measure_path)[measure_var]
    else:
        measure_data = xarray.open_dataset(measure_path)[measure_var]

    if threshold_path.suffix == ".zarr" and threshold_path.isdir():
        threshold_data = xarray.open_zarr(threshold_path)[threshold_var]
    else:
        threshold_data = xarray.open_dataset(threshold_path)[threshold_var]

    metric_ds = compute_individual_metrics(
        measure_data,
        threshold_data,
        hw_definitions,
        use_doy,
        doy_mask,
        include_threshold=include_threshold,
        check_variables=check_variables,
    )

    if output_path.suffix == ".zarr":
        metric_ds.to_zarr(output_path)
    else:
        metric_ds.to_netcdf(output_path)
