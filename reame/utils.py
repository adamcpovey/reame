"""Methods forked from CIS.data_io.

These mostly work around problems with masked data and trying to generate
multidimensional HyperPoints.

aod_without_prior: Calculation to remove a priori information from an ORAC
    AOD observation.
_get_MODIS_SDS_data: Fork of the CIS method.
hdf_read: Fork of the CIS method, stripping out some abstraction.
invalid: Checks if input is None, masked, NaN or infinite.
ncdf_read: Fork of the CIS method, stripping out some abstraction.
tabulate_data_points: Generates 2wvl AOD histograms given a HyperpointView.
tabulate_geo_points: Generates swath-extent histogram given a HyperpointView.
_tidy_ncdf_data: Fork of the CIS method.
"""

from datetime import datetime
from shapely.geometry import Polygon


REFERENCE_TIME = datetime(1995, 1, 1, 0, 0, 0)

def aod_without_prior(aod_data, unc_data, typ_data, ap_tau_types):
    from numpy import array, log, log10, logical_not
    from numpy.ma import getmaskarray, masked_all

    tau_var_all = unc_data / aod_data / log(10.)

    # Reject points with sufficient uncertainty to be unstable
    # when we take out the prior
    keep = (tau_var_all < 0.75).filled(False)
    # Remove negative AOD as we're taking a logarithm
    keep &= (aod_data > 0.).filled(False)
    keep &= logical_not(getmaskarray(typ_data))

    tau = log10(aod_data[keep])
    tau_var = tau_var_all[keep] * tau_var_all[keep]

    ap_tau = array([
        ap_tau_types[phs-1] for phs in typ_data[keep]
    ])
    ap_tau_var = 1.5 * 1.5

    weight = (ap_tau_var - tau_var) / (ap_tau_var * tau_var)
    value = (tau / tau_var - ap_tau / ap_tau_var) / weight

    result = masked_all(aod_data.shape)
    result[keep] = 10.**value

    return result


def _get_MODIS_SDS_data(sds, start=None, count=None, stride=None):
    """
    Reads raw data from an SD instance.

    :param sds: The specific sds instance to read
    :param start: List of indices to start reading from each dimension
    :param count: List of number of data to read from each dimension
    :param stride: List of strides to read from each dimension
    :return: A numpy array containing the raw data with missing data is replaced by NaN.
    """
    from cis.utils import create_masked_array_for_missing_data, listify
    from cis.data_io.products.MODIS import _apply_scaling_factor_MODIS
    from numpy.ma import masked_outside

    start = [] if start is None else listify(start)
    count = [] if count is None else listify(count)
    stride = [] if stride is None else listify(stride)
    _, ndim, dim_len, _, _ = sds.info()

    # Assume full read of all omitted dimensions
    while len(start) < ndim: start += [0]
    while len(count) < ndim: count += [-1]
    while len(stride) < ndim: stride += [1]

    # Allow lazy notation for "read all"
    count = [n if n >= 0 else l-x0 for x0, n, l in zip(start, count, dim_len)]

    data = sds.get(start, count, stride).squeeze()
    attributes = sds.attributes()

    # Apply Fill Value
    missing_value = attributes.get('_FillValue', None)
    if missing_value is not None:
        data = create_masked_array_for_missing_data(data, missing_value)

    # Check for valid_range
    valid_range = attributes.get('valid_range', None)
    if valid_range is not None:
        data = masked_outside(data, *valid_range)

    # Offsets and scaling.
    add_offset = attributes.get('add_offset', 0.0)
    scale_factor = attributes.get('scale_factor', 1.0)
    data = _apply_scaling_factor_MODIS(data, scale_factor, add_offset)

    return data


def hdf_read(filenames, variable, start=None, count=None, stride=None):
    """Returns variable, concatenated over a sequence of files."""
    from cis.data_io.hdf import read
    from cis.data_io.hdf_sd import get_metadata
    from cis.utils import concatenate

    sdata, _ = read(filenames, variable)
    var = sdata[variable]
    data = concatenate([
        _get_MODIS_SDS_data(i, start, count, stride) for i in var
    ])
    metadata = get_metadata(var[0])

    return data, metadata


def invalid(value):
    """Indicates if a value is a valid number."""
    from numpy import isfinite
    from numpy.ma import masked
    return value is None or value is masked or not isfinite(value)


def ncdf_read(filenames, variable, start=None, count=None, stride=None):
    """Returns variable, concatenated over a sequence of files."""
    from cis.data_io.netcdf import read, get_metadata
    from cis.utils import concatenate, listify

    data = []
    for f in listify(filenames):
        sdata = read(f, variable)
        var = sdata[variable]
        data.append(_tidy_ncdf_data(var, start, count, stride))

    metadata = get_metadata(var)

    return concatenate(data), metadata


# Reference polygons to deal with the dateline
E_HEMISPHERE = Polygon([[0, -90], [180, -90], [180, 90], [0, 90]])
W_HEMISPHERE = Polygon([[180, -90], [360, -90], [360, 90], [180, 90]])
def tabulate_data_points(points, grid):
    """Generate a 2-wavelength histogram from a HyperpointView"""
    import numpy as np

    from itertools import product
    from shapely.errors import TopologicalError
    from running_mean import WeightedMean

    def update_arrays(i, j, values, weight):
        """Updates hist, max_aod and mean_aod using pnt.val"""
        from reame.utils import invalid

        if weight == 0.:
            return

        k550, k670 = np.searchsorted(grid.aod_bins, values) - 1
        hist[i, j, k550, k670] += weight

        for l, value in enumerate(values):
            if not invalid(value):
                mean_aod[i, j, l].add_array(value, weight)
                if value > max_aod[i, j, l]:
                    max_aod[i, j, l] = value
                if value > 0.:
                    mean_log[i, j, l].add_array(np.log10(value), weight)

    # Initialise arrays
    # Joint histogram of spectral AOD
    hist = np.zeros((grid.nlat, grid.nlon, grid.naod, grid.naod), dtype=float)
    # Maximum AOD
    max_aod = np.full((grid.nlat, grid.nlon, grid.nlam), -999., dtype=float)
    # Linear mean AOD
    mean_aod = WeightedMean(
        shape=(grid.nlat, grid.nlon, grid.nlam), dtype=float, label="aod"
    )
    # Geometric mean AOD
    mean_log = WeightedMean(
        shape=(grid.nlat, grid.nlon, grid.nlam), dtype=float, label="logaod"
    )

    # Work through all data
    points.set_longitude_range(grid.lon_bins[0])
    for pnt in points.iter_non_masked_points():
        # Determine the histogram indices
        x = np.unique(np.searchsorted(grid.lat_bins, pnt.latitude, 'right')) - 1
        if np.any(x < 0) or np.any(x >= grid.nlat):
            raise ValueError("Invalid latitudes: {}".format(pnt.latitude))

        y = np.unique(np.searchsorted(grid.lon_bins, pnt.longitude, 'right')) - 1
        if np.any(y < 0) or np.any(y >= grid.nlon):
            # Catch points that exactly hit the upper limit
            if np.any(pnt.longitude == grid.lon_bins[-1]):
                y[y == grid.nlon] -= 1
            else:
                raise ValueError("Invalid longitudes: {}".format(pnt.longitude))

        if len(x) == 1 and len(y) == 1:
            # Pixel completely falls within one lat/lon grid cell
            update_arrays(x[0], y[0], pnt.val, 1.)
        elif 0 in y and grid.nlon - 1 in y:
            # Pixel crosses the prime meridian
            lon_tmp = pnt.longitude.copy()
            lon_tmp[lon_tmp > 180.] -= 360.
            tmp = Polygon(list(zip(lon_tmp, pnt.latitude)))
            try:
                edge0 = E_HEMISPHERE.intersection(tmp)
            except TopologicalError:
                print("ERROR", edge0.boundary.xy)
                continue

            lon_tmp = pnt.longitude.copy()
            lon_tmp[lon_tmp < 180.] += 360.
            tmp = Polygon(list(zip(lon_tmp, pnt.latitude)))
            try:
                edge1 = W_HEMISPHERE.intersection(tmp)
            except TopologicalError:
                print("ERROR", edge1.boundary.xy)
                continue

            area = edge0.area + edge1.area
            for edge, i, j in product((edge0, edge1), x, y):
                try:
                    sect = grid[i,j].intersection(edge)
                except TopologicalError:
                    print("ERROR", edge.boundary.xy)
                    continue
                update_arrays(i, j, pnt.val, sect.area / area)
        else:
            # Pixel lies in multiple grid cells
            edge = Polygon(list(zip(pnt.longitude, pnt.latitude)))
            area = edge.area
            for i, j in product(x, y):
                try:
                    # MOD03 files rarely contain trash locations
                    sect = grid[i,j].intersection(edge)
                except TopologicalError:
                    print("ERROR", edge.boundary.xy)
                    continue
                update_arrays(i, j, pnt.val, sect.area / area)

    return hist, max_aod, mean_aod, mean_log


# TODO: Less replication of the above routine
def tabulate_geo_points(points, grid):
    """Generate a swath extent histogram from a HyperpointView"""
    import numpy as np

    from itertools import product
    from shapely.errors import TopologicalError
    from running_mean import WeightedMean


    # Initialise arrays
    hist = np.zeros((grid.nlat, grid.nlon), dtype=float)

    # Work through all data
    points.set_longitude_range(grid.lon_bins[0])
    for pnt in points.iter_non_masked_points():
        # Determine the histogram indices
        x = np.unique(np.searchsorted(grid.lat_bins, pnt.latitude, 'right')) - 1
        if np.any(x < 0) or np.any(x >= grid.nlat):
            raise ValueError(f"Invalid latitudes: {pnt.latitude}")

        y = np.unique(np.searchsorted(grid.lon_bins, pnt.longitude, 'right')) - 1
        if np.any(y < 0) or np.any(y >= grid.nlon):
            # Catch points that exactly hit the upper limit
            if np.any(pnt.longitude == grid.lon_bins[-1]):
                y[y == grid.nlon] -= 1
            else:
                raise ValueError(f"Invalid longitudes: {pnt.longitude}")

        if len(x) == 1 and len(y) == 1:
            # Pixel completely falls within one lat/lon grid cell
            hist[x[0], y[0]] += 1.
        elif 0 in x and grid.nlon - 1 in x:
            # Pixel crosses the prime meridian
            lon_tmp = pnt.longitude.copy()
            lon_tmp[lon_tmp > 180.] -= 360.
            tmp = Polygon(list(zip(lon_tmp, pnt.latitude)))
            try:
                edge0 = E_HEMISPHERE.intersection(tmp)
            except TopologicalError:
                continue

            lon_tmp = pnt.longitude.copy()
            lon_tmp[lon_tmp < 180.] += 360.
            tmp = Polygon(list(zip(lon_tmp, pnt.latitude)))
            try:
                edge1 = W_HEMISPHERE.intersection(tmp)
            except TopologicalError:
                continue

            area = edge0.area + edge1.area
            for edge, i, j in product((edge0, edge1), x, y):
                try:
                    sect = grid[i,j].intersection(edge)
                except TopologicalError:
                    continue
                hist[i, j] += sect.area / area
        else:
            # Pixel lies in multiple grid cells
            edge = Polygon(list(zip(pnt.longitude, pnt.latitude)))
            area = edge.area
            for i, j in product(x, y):
                try:
                    # MOD03 files rarely contain trash locations
                    sect = grid[i,j].intersection(edge)
                except TopologicalError:
                    continue
                hist[i, j] += sect.area / area

    return hist


def _tidy_ncdf_data(var, start=None, count=None, stride=None):
    from cis.utils import listify
    from numpy.ma import MaskedArray

    start = [] if start is None else listify(start)
    count = [] if count is None else listify(count)
    stride = [] if stride is None else listify(stride)
    dim_len = var.shape
    ndim = len(dim_len)

    # Assume full read of all omitted dimensions
    while len(start) < ndim: start += [0]
    while len(count) < ndim: count += [-1]
    while len(stride) < ndim: stride += [1]

    sl = (
        slice(x0, n if n >= 0 else l-x0, s)
        for x0, n, s, l in zip(start, count, stride, dim_len)
    )

    data = var[sl]
    if isinstance(data, MaskedArray):
        return data
    else:
        return MaskedArray(data)
