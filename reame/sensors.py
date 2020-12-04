"""Specific BProduct instances for various datasets.

Classes:
ORAC: AATSR or SLSTR in Aerosol CCI format from ORAC.
APORAC: As above, but with the prior removed.
SU: AATSR from Swansea University.
ADV: AATSR from FMI.
DeepBlue: MODIS land algorithm.
DarkTarget: MODIS general algorithm.
MAIAC: MODIS atmospheric correction algorithm.
ATSRL1: AATSR geolocation.
MODISL1: MODIS geolocation.
"""

from abc import abstractmethod
import datetime as dt
import numpy as np

from cis.utils import concatenate
from reame.classes import BProduct


# Epoch times for various instruments
MODIS_REFERENCE_TIME = dt.datetime(1993, 1, 1, 0, 0, 0)
ATSR_REFERENCE_TIME = dt.datetime(1970, 1, 1, 0, 0, 0)


# Variables to be read for each sensor
VAR = dict(
    ORAC=["AOD550", "AOD670"], SU=["AOD550", "AOD670"], ADV=["AOD550", "AOD670"],
    APORAC=["AOD550", "AOD870"], DeepBlue=["AOD550", "AOD660"],
    DarkTarget=["AOD550", "AOD660"], MAIAC=["AOD550", "AOD470"],
)


# Filename checks
ESA_FILE = lambda name:"ESACCI-L2P" in name and name.endswith("nc")
C3S_FILE = lambda name:"C3S_312aL5-L2P" in name and name.endswith("nc")
IS_FILE = dict(
    ORAC=lambda name: (ESA_FILE(name) or C3S_FILE(name) and "-ORAC_" in name),
    SU=lambda name: ESA_FILE(name) and "-SU_" in name,
    ADV=lambda name: ESA_FILE(name) and "-ADV_" in name,
    DeepBlue=lambda name: (name.startswith(("MOD04", "MYD04"))
                           and name.endswith("hdf")),
    DarkTarget=lambda name: (name.startswith(("MOD04", "MYD04"))
                             and name.endswith("hdf")),
    MAIAC=lambda name: name.startswith("MCD19A2") and name.endswith("hdf"),
    ATSRL1=lambda name: ((name.startswith("ATS_TOA") and name.endswith("N1")) or
                         (name.startswith("AT2_TOA") and name.endswith("E1"))),
    MODISL1=lambda name: ((name.startswith("MOD03") or name.startswith("MYD03"))
                          and name.endswith("hdf")),
)
IS_FILE["APORAC"] = IS_FILE["ORAC"]


def _get_hdf_data(file_object, name, **kwargs):
    var = file_object.select(name)
    data = var.get(**kwargs)
    var.endaccess()
    return data


class CciBase(BProduct):
    """Shared methods for Aerosol CCI products."""
    def _create_bounded_coord_list(self):
        """Bootstrap bounds to the CCI coordinates."""

        def cell_corners(lat0, lon0):
            """Extrapolate on a sinusoidal grid."""
            try:
                u, v = self.sin_grid.to_sin(lat0, lon0)
            except IndexError:
                return np.ma.masked_all((2,4))
            corners = np.array([[u, u+1, u+1, u], [v, v, v+1, v+1]])
            lat, lon = self.sin_grid.to_rect(*corners)
            return lat, lon

        coords = self._create_coord_list()
        lat_data = coords[0].data
        lon_data = coords[1].data
        time_data = coords[2].data

        latlon_bounds = np.ma.array([
            cell_corners(lat, lon)
            for lat, lon in zip(lat_data.ravel(), lon_data.ravel())
        ]).reshape(lat_data.shape + (2, 4))
        latlon_bounds = np.moveaxis(latlon_bounds, 1, 0)

        unique_times = np.unique(time_data.compressed())
        try:
            deltas = unique_times[1:] - unique_times[:-1]
            delta_map = {t: d/2 for t, d in zip(unique_times[:-1], deltas)}
            delta_map[unique_times[-1]] = deltas[-1] / 2
            time_bounds = np.ma.array([
                [t-delta_map[t], t+delta_map[t]] if t is not np.ma.masked
                else [np.ma.masked, np.ma.masked]
                for t in time_data.ravel()
            ]).reshape(time_data.shape + (2,))
        except IndexError:
            time_bounds = np.ma.masked_all(time_data.shape + (2,))

        coords[0].bounds = latlon_bounds[0]
        coords[1].bounds = latlon_bounds[1]
        coords[2].bounds = time_bounds

        return coords

    def _create_coord_list(self):
        """Read file coordinates into a CIS object"""
        from cis.data_io.Coord import Coord, CoordList
        from reame.utils import ncdf_read

        try:
            lon_data, lon_metadata = ncdf_read(self.filenames, "longitude")
            lat_data, lat_metadata = ncdf_read(self.filenames, "latitude")
        except IndexError:
            lon_data, lon_metadata = ncdf_read(self.filenames, "lon")
            lat_data, lat_metadata = ncdf_read(self.filenames, "lat")

        lat = Coord(lat_data, lat_metadata, "Y")
        lat.update_shape()
        lat.update_range()
        lon = Coord(lon_data, lon_metadata, "X")
        lon.update_shape()
        lat.update_range()

        time_data, time_metadata = ncdf_read(self.filenames, "time")
        # Ensure the standard name is set
        time_metadata.standard_name = "time"
        time = Coord(time_data, time_metadata, "T")
        time.convert_TAI_time_to_std_time(ATSR_REFERENCE_TIME)
        time.update_shape()
        time.update_range()

        return CoordList([lat, lon, time])

    @property
    def sin_grid(self):
        try:
            return self._sin_grid
        except AttributeError:
            self._set_sin_grid()
            return self._sin_grid

    @property
    def shape(self):
        from netCDF4 import Dataset
        try:
            return self._shape
        except AttributeError:
            n = 0
            for f in self.filenames:
                with Dataset(f) as fu:
                    n += fu.dimensions["pixel_number"].size
            self._shape = (n,)
            return self._shape

    @abstractmethod
    def _set_sin_grid(self):
        """Establishes sinusoidal coordinate system."""
        ...

    def get_variable_names(self):
        return ["AOD550", "AOD670", "AOD870", "AOD1600"]

    @property
    def AOD550(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD550")

    @property
    def AOD670(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD670")

    @property
    def AOD870(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD870")

    @property
    def AOD1600(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD1600")

    @property
    def AOD550_uncertainty(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD550_uncertainty")

    @property
    def AOD670_uncertainty(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD670_uncertainty")

    @property
    def AOD870_uncertainty(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD870_uncertainty")

    @property
    def AOD1600_uncertainty(self):
        from reame.utils import ncdf_read
        return ncdf_read(self.filenames, "AOD1600_uncertainty")


class ORAC(CciBase):
    def _set_sin_grid(self):
        from pyorac.mappable import SinGrid
        self._sin_grid = SinGrid(n_equator=4008, offset=(0.5, 0.5))

    def _get_qcmask(self):
        from reame.utils import ncdf_read
        qcdata, _ = ncdf_read(self.filenames, "quality_flag")
        return (qcdata != 1).filled(True)


class SU(CciBase):
    def _set_sin_grid(self):
        from pyorac.mappable import SinGrid
        self._sin_grid = SinGrid(n_equator=4008, offset=(0., 0.))

    def _get_qcmask(self):
        return np.zeros(self.shape, dtype=bool)


class ADV(CciBase):
    def _set_sin_grid(self):
        from pyorac.mappable import SinGrid
        self._sin_grid = SinGrid(n_equator=3600, offset=(0.5, 0.5))

    def _get_qcmask(self):
        return np.zeros(self.shape, dtype=bool)


class APORAC(ORAC):
    def get_variable_names(self):
        return ["AOD550", "AOD870"]

    @property
    def AOD550(self):
        from reame.utils import aod_without_prior, ncdf_read

        aod_data, aod_metadata = ncdf_read(self.filenames, "AOD550")
        unc_data, _ = ncdf_read(self.filenames, "AOD550_uncertainty")
        typ_data, _ = ncdf_read(self.filenames, "aerosol_type")
        new_aod = aod_without_prior(
            aod_data, unc_data, typ_data, [-1.0] * 10
        )
        return new_aod, aod_metadata

    @property
    def AOD870(self):
        from reame.utils import aod_without_prior, ncdf_read

        aod_data, aod_metadata = ncdf_read(self.filenames, "AOD870")
        unc_data, _ = ncdf_read(self.filenames, "AOD870_uncertainty")
        typ_data, _ = ncdf_read(self.filenames, "aerosol_type")
        new_aod = aod_without_prior(
            aod_data, unc_data, typ_data,
            [-0.951, -0.705, -0.711, -0.715, -0.717,
             -0.869, -0.960, -0.865, -0.706, -0.407]
        )
        return new_aod, aod_metadata

    @property
    def AOD670(self):
        raise NotImplementedError("No uncertainty information @ 670nm")

    @property
    def AOD1600(self):
        raise NotImplementedError("No uncertainty information @ 1600nm")


class ModisBase(BProduct):
    """Shared methods for MODIS algorithms."""
    def __init__(self, filenames, mod03_path=None):

        def to_mod03_filename(f):
            """Replace 04_L2 with 03 in a filename."""
            import os.path
            from glob import glob

            fdr, name = os.path.split(f)
            parts = name.split(".")
            parts[0] = parts[0].replace("04_L2", "03")
            parts[4] = "?" * 13

            if mod03_path is not None:
                date = dt.datetime.strptime(parts[1] + parts[2], "A%Y%j%H%M")
                fdr = date.strftime(mod03_path)

            search_path = os.path.join(fdr, ".".join(parts))
            try:
                return glob(search_path)[0]
            except IndexError:
                raise FileNotFoundError("MOD03: " + search_path)

        def valid_MOD_file(f):
            """Checks a MODIS file can be used."""
            from pyhdf.SD import SD
            from pyhdf.error import HDF4Error
            try:
                file_object = SD(f)
                val = _get_hdf_data(file_object, "Longitude", start=(0,0),
                                    count=(1,1), stride=(1,1))
                file_object.end()
                return True
            except HDF4Error:
                return False

        super(ModisBase, self).__init__(filenames)
        filtered_filenames = []
        filtered_mod03_filenames = []
        for f0 in self._filenames:
            try:
                f1 = to_mod03_filename(f0)
            except FileNotFoundError:
                continue
            if valid_MOD_file(f0) and valid_MOD_file(f1):
                filtered_filenames.append(f0)
                filtered_mod03_filenames.append(f1)

        self._filenames = filtered_filenames
        self._mod03_filenames = filtered_mod03_filenames


    def _create_bounded_coord_list(self):
        """Adaptation of the CIS MODIS_L2 class version that isn't lazy."""
        from cis.time_util import convert_sec_since_to_std_time
        from pyhdf.error import HDF4Error
        from pyhdf.SD import SD

        def calc_latlon_bounds(base_data, nrows=10):
            """Interpolate 10-line MODIS scans to return pixel edges."""
            from acp_utils import rolling_window
            from itertools import product
            from scipy.interpolate import RegularGridInterpolator

            # Coordinates in file give cell centres
            nx, ny = base_data.shape
            assert nx % nrows == 0
            x0 = np.arange(0.5, nrows, 1)
            y0 = np.arange(0.5, ny, 1)

            # Aerosol pixels skip the outermost columns
            ystart = (ny % nrows) // 2
            x1 = np.array([0, nrows])
            y1 = np.arange(ystart, ny+1, nrows)

            # Iterate over 10-line chunks
            bounds = []
            for chunk in np.split(base_data, nx // nrows, 0):
                if (chunk.max() - chunk.min()) > 180.:
                    # Sodding dateline
                    chunk[chunk < 0.] += 360.
                interp = RegularGridInterpolator(
                    (x0, y0), chunk, "linear", False, None
                )
                tmp = interp(list(product(x1, y1))).reshape(2, len(y1))
                corners = rolling_window(tmp, (2,2))
                bounds.append(corners.reshape(ny // nrows, 4))

            # Ensure corners are given in sequential order
            bounds = np.ma.masked_invalid(bounds)
            bounds[..., 2:4] = bounds[..., [3,2]]

            return bounds

        lon_bounds = []
        lat_bounds = []
        for f in self._mod03_filenames:
            try:
                file_object = SD(f)
                lon_1kmdata = _get_hdf_data(file_object, "Longitude")
                lat_1kmdata = _get_hdf_data(file_object, "Latitude")
                file_object.end()
            except HDF4Error:
                raise IOError("Corrupted file " + f)

            tmp_bounds = calc_latlon_bounds(lon_1kmdata)
            tmp_bounds[tmp_bounds > 180.] -= 360.
            tmp_bounds[tmp_bounds <= -180.] += 360.
            lon_bounds.append(tmp_bounds)

            tmp_bounds = calc_latlon_bounds(lat_1kmdata)
            tmp_bounds[tmp_bounds >= 90.] = np.ma.masked
            tmp_bounds[tmp_bounds <= -90.] = np.ma.masked
            lat_bounds.append(tmp_bounds)

        coords = self._create_coord_list()

        coords[0].bounds = concatenate(lat_bounds)
        coords[1].bounds = concatenate(lon_bounds)

        unique_times = np.unique(coords[2].data.compressed())
        try:
            deltas = unique_times[1:] - unique_times[:-1]
            delta_map = {t: d/2 for t, d in zip(unique_times, deltas)}
            delta_map[unique_times[-1]] = deltas[-1] / 2
            time_bounds = np.ma.array([
                [t-delta_map[t], t+delta_map[t]] if t is not np.ma.masked else [np.ma.masked, np.ma.masked]
                for t in coords[2].data.ravel()
            ]).reshape(coords[2].data.shape + (2,))
        except IndexError:
            # File too small to have multiple time stamps; guess +-2.5min
            time_bounds = np.stack([coords[2].data - 0.00174, coords[2].data +0.00174], axis=2)
        coords[2].bounds = convert_sec_since_to_std_time(
            time_bounds, MODIS_REFERENCE_TIME
        )

        return coords

    def _create_coord_list(self):
        """Read data coordinates into CIS object"""
        from cis.data_io.Coord import Coord, CoordList
        from reame.utils import hdf_read

        lon_data, lon_metadata = hdf_read(self.filenames, "Longitude")
        lon = Coord(lon_data, lon_metadata, "X")
        lon.update_shape()
        lon.update_range()

        lat_data, lat_metadata = hdf_read(self.filenames, "Latitude")
        lat = Coord(lat_data, lat_metadata, "Y")
        lat.update_shape()
        lat.update_range()

        time_data, time_metadata = hdf_read(self.filenames, "Scan_Start_Time")
        # Ensure the standard name is set
        time_metadata.standard_name = "time"
        time = Coord(time_data, time_metadata, "T")
        time.convert_TAI_time_to_std_time(MODIS_REFERENCE_TIME)
        time.update_shape()
        time.update_range()

        return CoordList([lat, lon, time])


class DeepBlue(ModisBase):
    def get_variable_names(self):
        return ["AOD412", "AOD470", "AOD550", "AOD660"]

    @property
    def AOD550(self):
        from reame.utils import hdf_read
        return hdf_read(
            self.filenames, "Deep_Blue_Aerosol_Optical_Depth_550_Land"
        )

    @property
    def AOD412(self):
        return self._spectral_aod(0)

    @property
    def AOD470(self):
        return self._spectral_aod(1)

    @property
    def AOD660(self):
        return self._spectral_aod(2)

    def _spectral_aod(self, i):
        from reame.utils import hdf_read
        return hdf_read(
            self.filenames, "Deep_Blue_Spectral_Aerosol_Optical_Depth_Land",
            start=[i], count=[1]
        )

    def _get_qcmask(self):
        from reame.utils import hdf_read
        qcdata, _ = hdf_read(
            self.filenames, "Deep_Blue_Aerosol_Optical_Depth_550_Land_QA_Flag"
        )
        return (qcdata < 2).filled(True)


class DarkTarget(ModisBase):
    def get_variable_names(self):
        return ["AOD470", "AOD550", "AOD660", "AOD2130"]

    @property
    def land(self):
        from reame.utils import hdf_read
        try:
            return (self._lsf == 1).filled(False)
        except AttributeError:
            self._lsf, _ = hdf_read(self.filenames, "Land_sea_Flag")
            return (self._lsf == 1).filled(False)

    @property
    def AOD470(self):
        return self._spectral_aod(0)

    @property
    def AOD550(self):
        return self._spectral_aod(1)

    @property
    def AOD660(self):
        return self._spectral_aod(2)

    def _spectral_aod(self, i):
        from reame.utils import hdf_read
        land, land_metadata = hdf_read(
            self.filenames, "Corrected_Optical_Depth_Land",
            start=[i], count=[1]
        )
        sea, sea_metadata = hdf_read(
            self.filenames, "Effective_Optical_Depth_Average_Ocean",
            start=[i], count=[1]
        )

        data = np.ma.where(self.land, land, sea)
        land_metadata.standard_name = "atmosphere_absorption_optical_thickness_due_to_ambient_aerosol"
        land_metadata.long_name += "AND " + sea_metadata.long_name

        return data, land_metadata

    def _get_qcmask(self):
        from reame.utils import hdf_read
        qcdata, _ = hdf_read(self.filenames, "Land_Ocean_Quality_Flag")
        mask = np.ma.where(self.land, qcdata != 3, qcdata < 1)
        return mask.filled(True)


# TODO: Deal with time properly, as I don't care for the UKESM work
class MAIAC(BProduct):
    """Methods for MODIS MAIAC AOD product."""
    def __init__(self, filenames, mod03_path=None, grid_path=None):
        super(MAIAC, self).__init__(filenames)
        self.mod03_path = mod03_path
        self.grid_path = grid_path

    def gdal_variable_name(self, filename, variable):
        """Form list of field names to open for a given variable."""

        grid = ("grid5km" if variable in ("cosSZA", "cosVZA", "RelAZ",
                                          "Scattering_Angle", "Glint_Angle")
                else "grid1km")
        return ":".join((
            "HDF4_EOS", "EOS_GRID", '"'+filename+'"', grid, variable
        ))

    def _calculate_grid_edges(self, var_name):
        """Calculate coordinate grid from product's projection."""
        from osgeo.gdal import Open
        from osgeo import osr

        variable = Open(var_name)

        sin = osr.SpatialReference()
        sin.ImportFromWkt(variable.GetProjection())

        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)

        tx = osr.CoordinateTransformation(sin, wgs84)
        transform = np.vectorize(tx.TransformPoint)

        # +1 as we want grid cell edges
        x0, dx, _, y0, _, dy = variable.GetGeoTransform()
        x_ = x0 + dx * np.arange(variable.RasterXSize + 1)
        y_ = y0 + dy * np.arange(variable.RasterYSize + 1)
        x, y = np.meshgrid(x_, y_)
        lat_edges, lon_edges, _ = transform(x, y)

        def make_corner_list(arr):
            """Ravel a grid of cell edges into an ordered list of corners."""
            from acp_utils import rolling_window
            out = rolling_window(arr, (2,2))
            out = out.reshape(out.shape[:-2] + (4,))
            out[..., 2:4] = out[..., [3,2]]
            return out

        lat_bounds = make_corner_list(lat_edges)
        lon_bounds = make_corner_list(lon_edges)

        return lat_bounds, lon_bounds

    def _calculate_grid_centres(self, var_name):
        """Calculate coordinates of cell centres from project's projection"""
        from osgeo.gdal import Open
        from osgeo import osr

        variable = Open(var_name)

        sin = osr.SpatialReference()
        sin.ImportFromWkt(variable.GetProjection())

        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)

        tx = osr.CoordinateTransformation(sin, wgs84)
        transform = np.vectorize(tx.TransformPoint)

        x0, dx, _, y0, _, dy = variable.GetGeoTransform()
        x_ = x0 + dx * np.arange(0.5, variable.RasterXSize)
        y_ = y0 + dy * np.arange(0.5, variable.RasterYSize)
        x, y = np.meshgrid(x_, y_)
        lat_data, lon_data, _ = transform(x, y)

        return lat_data, lon_data

    def _calculate_grid_time(self, var_name, lat_data, lon_data):
        """Approximate time from a pair of corresponding MOD03 files"""
        from osgeo.gdal import Open
        from scipy.interpolate import griddata

        def fetch_MOD03_coordinates(start_time, aqua=False):
            import os.path
            from glob import glob
            from pyhdf.SD import SD
            from pyhdf.error import HDF4Error

            # Locate MOD03 file
            search_path = start_time.strftime(os.path.join(
                self.mod03_path, "MOD03.A%Y%j.%H%M.061*hdf"
            ))
            if aqua:
                # NOTE: System dependent approximation
                search_path = search_path.replace("MOD", "MYD")
            try:
                mod03_file = glob(search_path)[0]
            except IndexError:
                raise FileNotFoundError("MOD03: " + search_path)

            # Read space-time grid from that file
            try:
                file_object = SD(mod03_file)
                dims = file_object.datasets()["Longitude"][1]
                count = dims[0] // 10, dims[1] // 10
                mod_lon = _get_hdf_data(file_object, "Longitude", start=(0, 2),
                                        count=count, stride=(10, 10))
                mod_lat = _get_hdf_data(file_object, "Latitude", start=(0, 2),
                                        count=count, stride=(10, 10))
                mod_time = _get_hdf_data(file_object, "EV start time",
                                         count=count[:1])
                file_object.end()
            except HDF4Error:
                raise IOError("Corrupted file: " + mod03_file)

            return mod_lon, mod_lat, mod_time

        time_data = []
        variable = Open(var_name)
        meta = variable.GetMetadata_Dict()
        for timestamp in meta["Orbit_time_stamp"].split():
            # Parse time stamp
            start_time = dt.datetime.strptime(timestamp[:-1], "%Y%j%H%M")

            try:
                # Interpolate time from MOD03 files
                mod_lon0, mod_lat0, mod_time0 = fetch_MOD03_coordinates(
                    start_time - dt.timedelta(seconds=300), timestamp[-1] == "A"
                )
                mod_lon1, mod_lat1, mod_time1 = fetch_MOD03_coordinates(
                    start_time, timestamp[-1] == "A"
                )
                mod_lon = concatenate([mod_lon0, mod_lon1])
                mod_lat = concatenate([mod_lat0, mod_lat1])
                mod_time = concatenate([mod_time0, mod_time1])
                if (mod_lon.max() - mod_lon.min()) > 180.:
                    # Sodding dateline
                    mod_lon[mod_lon < 0.] += 360.

                # Interpolate that grid onto the sinusoidal projection
                time = griddata(
                    (mod_lon.ravel(), mod_lat.ravel()),
                    np.tile(mod_time, mod_lon.shape[1]),
                    (lon_data, lat_data), method="nearest"
                )
            except (FileNotFoundError, TypeError):
                # Just use the orbit start time
                seconds = start_time - MODIS_REFERENCE_TIME
                time = np.full(lat_data.shape, seconds.total_seconds())

            time_data.append(time)

        return concatenate(time_data)

    def _create_bounded_coord_list(self):
        from cis.time_util import convert_sec_since_to_std_time
        from os.path import basename

        coords = self._create_coord_list()

        lat_bounds_all = []
        lon_bounds_all = []
        for fname in self.filenames:
            if self.grid_path:
                granule = basename(fname).split(".")[2]
                lat_bounds, lon_bounds = self._read_grid_edges(granule)
            else:
                var_name = self.gdal_variable_name(fname, "Optical_Depth_055")
                lat_bounds, lon_bounds = self._calculate_grid_edges(var_name)

            # Workaround files containing only one day
            sh = (-1,) + lat_bounds.shape[:-1]
            keep = np.logical_not(self._read_qcmask(fname)).reshape(sh)

            for keep_slice in keep:
                lat_bounds_all.extend(lat_bounds[keep_slice])
                lon_bounds_all.extend(lon_bounds[keep_slice])

        coords[0].bounds = np.ma.array(lat_bounds_all)
        coords[1].bounds = np.ma.array(lon_bounds_all)
        # As the time stamp is approximate (multiple scans can fall in a single
        # sinusoidal cell), guess the bounds are +/- 2 scans (each being 5s).
        coords[2].bounds = convert_sec_since_to_std_time(
            np.stack([coords[2].data - 10, coords[2].data + 10], axis=-1),
            MODIS_REFERENCE_TIME
        )

        return coords

    def _create_coord_list(self):
        from cis.data_io.Coord import Coord, CoordList
        from cis.data_io.ungridded_data import Metadata
        from cis.time_util import convert_sec_since_to_std_time
        from os.path import basename

        lat_all = []
        lon_all = []
        time_all = []
        for fname in self.filenames:
            var_name = self.gdal_variable_name(fname, "Optical_Depth_055")
            if self.grid_path:
                granule = basename(fname).split(".")[2]
                lat_data, lon_data = self._read_grid_centres(granule)
            else:
                lat_data, lon_data = self._calculate_grid_centres(var_name)
            time_data = self._calculate_grid_time(var_name, lat_data, lon_data)

            # Workaround files containing only one day
            sh = (-1,) + lat_data.shape
            time_data = time_data.reshape(sh)
            keep = np.logical_not(self._read_qcmask(fname)).reshape(sh)

            for time_slice, keep_slice in zip(time_data, keep):
                lat_all.extend(lat_data[keep_slice])
                lon_all.extend(lon_data[keep_slice])
                time_all.extend(time_slice[keep_slice])

        if len(lat_all) == 0:
            raise NotImplementedError("It's empty!")

        lat = Coord(
            np.ma.array(lat_all), Metadata(
                name="lat", standard_name="latitude", units="degrees",
                range=(-90., 90.)
            ), "Y"
        )
        lat.update_shape()
        lon = Coord(
            np.ma.array(lon_all), Metadata(
                name="lon", standard_name="longitude", units="degrees",
                range=(-180., 180.)
            ), "X"
        )
        lon.update_shape()
        time = Coord(
            np.ma.array(time_all), Metadata(
                name="time", standard_name="time",
                units="Seconds since 1993-1-1 00:00:00.0 0"
            ), "T"
        )
        time.convert_TAI_time_to_std_time(MODIS_REFERENCE_TIME)
        time.update_shape()

        # Set the QC mask as we now know how many points we have
        self._qcmask = np.full(lat.shape, False)

        return CoordList([lat, lon, time])

    def _read_grid_edges(self, granule):
        from os.path import join

        fname = join(self.grid_path, granule + ".npz")
        with np.load(fname) as sv:
            return sv["lat_bound"], sv["lon_bound"]

    def _read_grid_centres(self, granule):
        from os.path import join

        fname = join(self.grid_path, granule + ".npz")
        with np.load(fname) as sv:
            return sv["lat_cent"], sv["lon_cent"]

    def get_variable_names(self):
        return ["AOD470", "AOD550"]

    @property
    def AOD470(self):
        from reame.utils import hdf_read
        aod = []
        for f in self.filenames:
            data, meta = hdf_read([f], "Optical_Depth_047")
            qc = self._read_qcmask(f)
            aod.extend(data[np.logical_not(qc)].filled(np.nan))
        return np.ma.masked_invalid(aod), meta

    @property
    def AOD550(self):
        from reame.utils import hdf_read
        aod = []
        for f in self.filenames:
            data, meta = hdf_read([f], "Optical_Depth_055")
            qc = self._read_qcmask(f)
            aod.extend(data[np.logical_not(qc)].filled(np.nan))
        return np.ma.masked_invalid(aod), meta

    def _read_qcmask(self, filename):
        from reame.utils import hdf_read
        qcdata, _ = hdf_read([filename], "AOD_QA")
        # mask_val = sum((2**i for i in (0,1,2,5,6,7,8,9,10,11)))
        mask = np.ma.bitwise_and(qcdata.astype("int64"), 4071) != 1
        return mask.filled(True)

    def _get_qcmask(self):
        raise NotImplementedError("Call _create_bounded_coord_list().")



class ATSRL1(BProduct):
    """ATSR-2/AATSR L1B files (for 'all-pixels' histograms)"""
    def _create_bounded_coord_list(self):
        from acp_utils import rolling_window
        from orbit import ATSR

        coords = self._create_coord_list()

        lat_bounds = []
        lon_bounds = []
        time_bounds = []
        for fname in self.filenames:
            prod = ATSR(fname)

            lat_c = rolling_window(prod.lat_corner, (2,2))
            lat_bounds.append(lat_c.reshape(prod.shape + (4,)))
            lon_c = rolling_window(prod.lon_corner, (2,2))
            lon_bounds.append(lon_c.reshape(prod.shape + (4,)))
            t = prod.get_time()
            b = np.stack([t, np.roll(t, -1)], axis=2)
            b[-1,:,1] = 2*t[-1,:] - t[-2,:]
            time_bounds.append(b)

        coords[0].bounds = concatenate(lat_bounds).reshape(coords[0].data.shape +(4,))
        coords[0].bounds[...,2:4] = coords[0].bounds[...,[3,2]]
        coords[1].bounds = concatenate(lon_bounds).reshape(coords[1].data.shape +(4,))
        coords[1].bounds[...,2:4] = coords[1].bounds[...,[3,2]]
        coords[2].bounds = concatenate(time_bounds)

        return coords

    def _create_coord_list(self):
        from cis.data_io.Coord import Coord, CoordList
        from cis.data_io.ungridded_data import Metadata
        from cis.time_util import cis_standard_time_unit as cstu

        # These implement a lot of what is necessary, but aren't in CIS style
        from acp_utils import rolling_window
        from orbit import ATSR

        lat_data = []
        lon_data = []
        time_data = []
        for fname in self.filenames:
            prod = ATSR(fname)

            lat_data.append(prod.lat)
            lon_data.append(prod.lon)
            time_data.append(prod.get_time())

        # TODO: Properly define metadata
        lat_meta = Metadata(standard_name="latitude", units="degrees")
        lon_meta = Metadata(standard_name="longitude", units="degrees")
        time_meta = Metadata(standard_name="time", units=cstu)

        lat = Coord(concatenate(lat_data), lat_meta, "Y")
        lat.update_shape()
        lat.update_range()
        lon = Coord(concatenate(lon_data), lon_meta, "Y")
        lon.update_shape()
        lon.update_range()
        time = Coord(concatenate(time_data), time_meta, "T")
        time.update_shape()
        time.update_range()

        return CoordList([lat, lon, time])

    @property
    def shape(self):
        from epr import Product
        try:
            return self._shape
        except AttributeError:
            n = 0
            for f in self.filenames:
                prod = Product(f)
                n += prod.get_scene_height()
            self._shape = (n, prod.get_scene_width())
            return self._shape

    def _get_qcmask(self):
        return np.zeros(self.shape, dtype=bool)

    def get_variable_names(self):
        return ["SolZen"]

    @property
    def SolZen(self):
        from orbit import ATSR
        from cis.data_io.ungridded_data import Metadata

        tmp = []
        for f in self.filenames:
            orbit = ATSR(f)

            # Get tie point grid
            sph = orbit._prod.get_sph()
            tie_field = sph.get_field("VIEW_ANGLE_TIE_POINTS")
            tie_pts = tie_field.get_elems()
            # Get tie point values
            scan_y = orbit._read_field("NADIR_VIEW_SOLAR_ANGLES_ADS",
                                       "img_scan_y")
            tie_solelv = orbit._read_field("NADIR_VIEW_SOLAR_ANGLES_ADS",
                                           "tie_pt_sol_elev")
            # Swath grid
            x = np.arange(512) - 255.5
            y = orbit._read_field("11500_12500_NM_NADIR_TOA_MDS", "img_scan_y")
            y[:-1] += 0.5 * (y[1:] - y[:-1])
            y[-1] += 0.5 * (y[-1] - y[-2])

            solelv = orbit.extrap_atsr_angle(tie_pts, scan_y, x, y, tie_solelv)
            tmp.append(90. - solelv)

        return concatenate(tmp), Metadata(standard_name="solar_zenith_angle", units="degrees")


class MODISL1(BProduct):
    """MOD/MYD L1 files (for 'all-pixels' histograms)"""
    def _create_bounded_coord_list(self):
        from cis.data_io.Coord import Coord, CoordList
        from cis.data_io.ungridded_data import Metadata
        from cis.time_util import cis_standard_time_unit as cstu

        # These implement a lot of what is necessary, but aren't in CIS style
        from acp_utils import rolling_window
        from orbit import MODIS

        lat_data = []
        lat_bounds = []
        lon_data = []
        lon_bounds = []
        time_data = []
        time_bounds = []
        for fname in self.filenames:
            prod = MODIS(fname)

            lat_data.append(prod.lat)
            lon_data.append(prod.lon)
            lat_c = rolling_window(prod.lat_corner, (2,2))
            lat_bounds.append(lat_c.reshape(prod.shape + (4,)))
            lon_c = rolling_window(prod.lon_corner, (2,2))
            lon_bounds.append(lon_c.reshape(prod.shape + (4,)))
            t = prod.get_time()
            time_data.append(t)
            b = np.stack([t, np.roll(t, -1)], axis=2)
            b[-1,:,1] = 2*t[-1,:] - t[-2,:]
            time_bounds.append(b)

        # TODO: Properly define metadata
        lat_meta = Metadata(standard_name="latitude", units="degrees")
        lon_meta = Metadata(standard_name="longitude", units="degrees")
        time_meta = Metadata(standard_name="time", units=cstu)

        lat = Coord(concatenate(lat_data), lat_meta, "Y")
        lat.update_shape()
        lat.update_range()
        lat.bounds = concatenate(lat_bounds).reshape(lat.shape +(4,))
        lat.bounds[...,2:4] = lat.bounds[...,[3,2]]
        lon = Coord(concatenate(lon_data), lon_meta, "Y")
        lon.update_shape()
        lon.update_range()
        lon.bounds = concatenate(lon_bounds).reshape(lon.shape +(4,))
        lon.bounds[...,2:4] = lon.bounds[...,[3,2]]
        time = Coord(concatenate(time_data), time_meta, "T")
        time.update_shape()
        time.update_range()
        time.bounds = concatenate(time_bounds)

        return CoordList([lat, lon, time])

    @property
    def shape(self):
        from pyhdf.SD import SD
        try:
            return self._shape
        except AttributeError:
            n = 0
            for f in self.filenames:
                prod = SD(f)
                n += prod.attributes()['Number of Scans']
            self._shape = (10*n, prod.attributes()['Max Earth Frames'])
            return self._shape

    def _get_qcmask(self):
        return np.zeros(self.shape, dtype=bool)

    def get_variable_names(self):
        return ["SolZen"]

    @property
    def SolZen(self):
        from reame.utils import hdf_read
        return hdf_read(self.filenames, "SolarZenith")
