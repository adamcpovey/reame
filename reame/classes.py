"""
Classes, largely forked from cis.data_io. These mostly work around masked data
and making multidimensional HyperPoints.

HistogramGrid: Container for the rectangular grid the histograms average over.
    When indexed, returns a shapely Polygon for that grid cell.
UngriddedBoundedData: Fork of ungridded_data.UngriddedData that adds the
    method bounds_flattened().
UngriddedArrayPointView: Fork of ungridded_data.UngriddedHyperPointView that
    can receive multiple data values for each point.
UngriddedCube: Mashup of UngriddedData and UngriddedDataList that uses the
    fact that the variables share a coordinate system.
UngriddedBoundedCube: Extension of UngriddedCube that includes coordinate bounds.
BProduct: Fork of cis.AProduct that creates UngriddedCube or
    UngriddedBoundedCube.
"""

import numpy as np

from abc import abstractmethod
from cis.data_io.hyperpoint_view import UngriddedHyperPointView
from cis.data_io.ungridded_data import UngriddedData, UngriddedDataList


class HistogramGrid(object):
    """Manipulations of the rectilinear grid I'm averaging over"""
    def __init__(self, path):
        """Load histogram grid from file"""
        super(HistogramGrid, self).__init__()
        try:
            with np.load(path) as file_ob:
                self.lat_bins = file_ob["x"]
                self.lon_bins = file_ob["y"]
                self.aod_bins = file_ob["z"]
        except KeyError:
            raise ValueError(f"Incorrect grid file format: {path}")

        self._path = path

    def __repr__(self):
        return f"Grid of shape {self.shape} loaded from {self._path}"

    def __getitem__(self, key):
        """Return Polygon for a given grid cell"""
        from shapely.geometry import Polygon

        try:
            i, j = key
            assert isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))
        except (TypeError, ValueError):
            if isinstance(key, (int, np.integer)):
                return list(self._iter_column(key))
            raise IndexError("Index must be one or two integers")
        except AssertionError:
            raise NotImplementedError("Haven't done slices yet")

        x0, x1 = self.lon_bins[j:j+2]
        y0, y1 = self.lat_bins[i:i+2]
        return Polygon(((x0,y0), (x0,y1), (x1,y1), (x1,y0)))

    def __iter__(self):
        """Iterate over rows"""
        for i in range(self.nlat):
            yield self._iter_column(i)

    def _iter_column(self, i):
        """Iterate over a column"""
        for j in range(self.nlon):
            yield self[i,j]

    @property
    def nlon(self):
        return len(self.lon_bins)-1

    @property
    def nlat(self):
        return len(self.lat_bins)-1

    @property
    def naod(self):
        return len(self.aod_bins)-1

    @property
    def shape(self):
        return self.nlat, self.nlon, self.naod

    @property
    def central_longitude(self):
        return 0.5*(self.lon_bins[0] + self.lon_bins[-1])

    @central_longitude.setter
    def central_longitude(self, value):
        while True:
            m = self.lon_bins > value+180.
            if not np.any(m):
                break
            self.lon_bins[m] -= 360.
        while True:
            m = self.lon_bins < value-180.
            if not np.any(m):
                break
            self.lon_bins[m] += 360.

    def pcolormesh(self, ax, z, *args, **kwargs):
        """Wrapper for pcolormesh, swapping ax for x,y"""
        from cartopy.crs import PlateCarree

        xx, yy = np.meshgrid(self.lat_bins, self.lon_bins)
        kwargs["transform"] = PlateCarree(central_longitude=self.central_longitude)
        # Apparently dims should be lat, lon, ... for good plotting
        return ax.pcolormesh(xx, yy, z.T, *args, **kwargs)


class UngriddedBoundedData(UngriddedData):
    """Workaround UngriddedData._post_processing() and add bounds."""
    def _post_process(self):
        pass

    @property
    def bounds_flattened(self):
        all_coords = self.coords().find_standard_coords()
        return [
            c.bounds.reshape(-1, c.bounds.shape[-1]) if c is not None else None
            for c in all_coords
        ]


class UngriddedArrayPointView(UngriddedHyperPointView):
    """Hack of UngriddedHyperPointView.

    Returns the bounds of each coordinate and an array of values.
    """
    def __init__(self, coords, data, non_masked_iteration=False):
        """We need to override the length attribute."""
        super(UngriddedArrayPointView, self).__init__(coords, data, non_masked_iteration)
        self.length = coords[0].shape[0]

    def __iter__(self):
        """Iterates over all or non-masked points according to the value of non_masked_iteration
        :return: next HyperPoint
        """
        from reame.utils import invalid
        for idx in range(self.length):
            if (self.non_masked_iteration and self.data is not None
                and all(map(invalid, self.data[idx]))):
                continue
            yield self.__getitem__(idx)

    def iter_non_masked_points(self):
        """Iterates over non-masked points regardless of the value of non_masked_iteration
        :return: next HyperPoint
        """
        from reame.utils import invalid
        for idx in range(self.length):
            if self.data is not None and all(map(invalid, self.data[idx])):
                continue
            yield self.__getitem__(idx)

    def enumerate_non_masked_points(self):
        """Iterates over non-masked points returning the index in the full
        data array and the corresponding HyperPoint.
        :return: tuple(index of point, HyperPoint)
        """
        from reame.utils import invalid
        for idx in range(self.length):
            if self.data is not None and all(map(invalid, self.data[idx])):
                continue
            yield (idx, self.__getitem__(idx))


class UngriddedCube(list):
    """Cube of variables that share a set of ungridded coordinates.

    Definitely not lazy in data management. Mostly copies UngriddedData except,

    Methods:
    __init__: Fork of cis.read_data_list to open necessary data and applies quality control.
    data_flattened: A 2D array where each variable is flattened.
    """
    def __init__(self, data, metadata, coords):
        from cis.data_io.Coord import CoordList
        from cis.utils import listify

        def getmask(arr):
            mask = np.ma.getmaskarray(arr)
            try:
                mask |= np.isnan(arr)
            except ValueError:
                pass
            return mask

        data = listify(data)
        metadata = listify(metadata)

        if isinstance(coords, list):
            self._coords = CoordList(coords)
        elif isinstance(coords, CoordList):
            self._coords = coords
        elif isinstance(coords, Coord):
            self._coords = CoordList([coords])
        else:
            raise ValueError("Invalid Coords type")

        # Throw out points where any coordinate is masked
        combined_mask = np.zeros(data[0].shape, dtype=bool)
        for coord in self._coords:
            combined_mask |= getmask(coord.data)
            coord.update_shape()
            coord.update_range()

        if combined_mask.any():
            keep = np.logical_not(combined_mask)
            data = [variable[keep] for variable in data]
            for coord in self._coords:
                coord.data = coord.data[keep]
                coord.update_shape()
                coord.update_range()

        super(UngriddedCube, self).__init__(zip(data, metadata))

    def __add__(self, rhs):
        if self._coords is rhs._coords:
            return super(UngriddedCube, self).__add__(rhs)
        else:
            raise NotImplementedError("Requires a single coordinate system")

    def __getitem__(self, item):
        data, meta = list.__getitem__(self, item)
        return UngriddedData(data, meta, self._coords)

    def __setitem__(self, key, value):
        _, meta = list.__getitem__(self, key)
        list.__setitem__(self, key, (value, meta))

    def __iter__(self):
        for data, meta in list.__iter__(self):
            yield UngriddedData(data, meta, self._coords)

    def append(self, data, meta):
        super(UngriddedCube, self).append((data, meta))

    def extend(self, iterable):
        for data, meta in iterable:
            self.append(data, meta)

    def coords(self, name_or_coord=None, standard_name=None, long_name=None, attributes=None, axis=None, var_name=None,
               dim_coords=True):
        """
        :return: A list of coordinates in this UngriddedData object fitting the given criteria
        """
        return self._coords.get_coords(name_or_coord, standard_name, long_name, attributes, axis, var_name)

    def coord(self, name_or_coord=None, standard_name=None, long_name=None, attributes=None, axis=None, var_name=None):
        """
        :raise: CoordinateNotFoundError
        :return: A single coord given the same arguments as :meth:`coords`.
        """
        return self._coords.get_coord(name_or_coord, standard_name, long_name, attributes, axis, var_name)

    @property
    def x(self):
        return self.coord(axis="X")

    @property
    def y(self):
        return self.coord(axis="Y")

    @property
    def t(self):
        return self.coord(axis="T")

    @property
    def lat(self):
        return self.coord(standard_name="latitude")

    @property
    def lon(self):
        return self.coord(standard_name="longitude")

    @property
    def time(self):
        return self.coord(axis="T")

    @property
    def data(self):
        data_zip = np.ma.stack([
            data for data, _ in list.__iter__(self)
        ])
        return np.moveaxis(data_zip, 0, -1)

    @property
    def data_flattened(self):
        data_zip = np.stack([
            data.flatten().filled(np.nan) for data, _ in list.__iter__(self)
        ])
        return data_zip.T

    @property
    def coords_flattened(self):
        all_coords = self.coords().find_standard_coords()
        return [
            c.data_flattened if c is not None else None
            for c in all_coords
        ]

    def get_coordinates_points(self):
        """Returns a HyperPointView of the coordinates of all points."""
        return UngriddedHyperPointView(self.coords_flattened, None)

    def get_all_points(self):
        """Returns a HyperPointView of all points."""
        return UngriddedArrayPointView(self.coords_flattened, self.data_flattened)

    def get_non_masked_points(self):
        """Returns a HyperPointView for which the default iterator omits masked points."""
        return UngriddedArrayPointView(self.coords_flattened, self.data_flattened, non_masked_iteration=True)

    def collocated_onto(self, sample, how='', kernel=None, missing_data_for_missing_sample=True, fill_value=None,
                        var_name='', var_long_name='', var_units='', **kwargs):
        return sample.sampled_from(self, how=how, kernel=kernel,
                                   missing_data_for_missing_sample=missing_data_for_missing_sample,
                                   fill_value=fill_value, var_name=var_name, var_long_name=var_long_name,
                                   var_units=var_units, **kwargs)

    def aggregate(self, how=None, **kwargs):
        from cis.data_io.ungridded_data import _aggregate_ungridded
        agg = _aggregate_ungridded(self, how, **kwargs)
        # Return the single item if there's only one (this depends on the kernel used)
        if len(agg) == 1:
            agg = agg[0]
        return agg

    def _get_coord(self, name):
        from cis.utils import standard_axes
        def _try_coord(data, coord_dict):
            import cis.exceptions as cis_ex
            import iris.exceptions as iris_ex
            try:
                coord = data.coord(**coord_dict)
            except (iris_ex.CoordinateNotFoundError, cis_ex.CoordinateNotFoundError):
                coord = None
            return coord

        coord = _try_coord(self, dict(name_or_coord=name)) or _try_coord(self, dict(standard_name=name)) \
            or _try_coord(self, dict(standard_name=standard_axes.get(name.upper(), None))) or \
                _try_coord(self, dict(var_name=name)) or _try_coord(self, dict(axis=name))

        return coord


class UngriddedBoundedCube(UngriddedCube):
    """Cube of variables that share a set of ungridded coordinates.

    Definitely not lazy in data management. Mostly copies UngriddedData except,

    Methods:
    __init__: Fork of cis.read_data_list to open necessary data and applies quality control.
    data_flattened: A 2D array where each variable is flattened.
    bounds_flattened: The bounds of all coordinates, as a list of lists. Used to calculate pixel borders.
    get_all_bounded_points: HyperPointView that gives bounds of each coordinate.
    get_non_masked_bounded_points: As above, but skips points with only masked values.
    """
    def __init__(self, data, metadata, coords):
        from cis.data_io.Coord import CoordList
        from cis.utils import listify

        def getmask(arr):
            mask = np.ma.getmaskarray(arr)
            try:
                mask |= np.isnan(arr)
            except ValueError:
                pass
            return mask

        data = listify(data)
        metadata = listify(metadata)

        if isinstance(coords, list):
            self._coords = CoordList(coords)
        elif isinstance(coords, CoordList):
            self._coords = coords
        elif isinstance(coords, Coord):
            self._coords = CoordList([coords])
        else:
            raise ValueError("Invalid Coords type")

        # Throw out points where any coordinate is masked
        combined_mask = np.zeros(data[0].shape, dtype=bool)
        for coord in self._coords:
            combined_mask |= getmask(coord.data)
            for bound in np.moveaxis(coord.bounds, -1, 0):
                combined_mask |= getmask(bound)
            coord.update_shape()
            coord.update_range()

        if combined_mask.any():
            keep = np.logical_not(combined_mask)
            data = [variable[keep] for variable in data]
            for coord in self._coords:
                coord.data = coord.data[keep]
                new_bounds = np.array([
                    bound[keep]
                    for bound in np.moveaxis(coord.bounds, -1, 0)
                ])
                coord.bounds = np.moveaxis(new_bounds, 0, -1)
                coord.update_shape()
                coord.update_range()

        super(UngriddedCube, self).__init__(zip(data, metadata))

    def __getitem__(self, item):
        data, meta = list.__getitem__(self, item)
        return UngriddedBoundedData(data, meta, self._coords)

    def __iter__(self):
        for data, meta in list.__iter__(self):
            yield UngriddedBoundedData(data, meta, self._coords)

    @property
    def bounds_flattened(self):
        all_coords = self.coords().find_standard_coords()
        return [
            c.bounds.reshape(-1, c.bounds.shape[-1]) if c is not None else None
            for c in all_coords
        ]

    def get_all_bounded_points(self):
        """Returns a HyperPointView with pixel boundaries of all points."""
        return UngriddedArrayPointView(self.bounds_flattened, self.data_flattened)

    def get_non_masked_bounded_points(self):
        """Returns a HyperPointView with pixel boundaries for which the default iterator omits masked points."""
        return UngriddedArrayPointView(self.bounds_flattened, self.data_flattened, non_masked_iteration=True)


class BProduct(object):
    """Fork of cis.AProduct to create BoundedCubes."""
    def __init__(self, filenames):
        from cis.utils import listify
        from glob import glob
        if isinstance(filenames, str):
            self._filenames = glob(filenames)
        else:
            self._filenames = listify(filenames)

    def create_data_object(self, variables):
        from cis.data_io.ungridded_data import UngriddedData

        coords = self._create_coord_list()
        all_data, metadata = list(zip(*[
            getattr(self, var) for var in variables
        ]))
        qcmask = self.qcmask

        # Mask out points with missing coordinate values
        for coord in coords:
            qcmask |= np.ma.getmaskarray(coord.data)

            if coord.data.dtype != 'object':
                qcmask |= np.isnan(coord.data)

        for data in all_data:
            data[qcmask] = np.ma.masked

        return UngriddedCube(all_data, metadata, coords)

    def create_bounded_data_object(self, variables):
        coords = self._create_bounded_coord_list()
        all_data, metadata = list(zip(*[
            getattr(self, var) for var in variables
        ]))
        qcmask = self.qcmask

        # Mask out points with missing coordinate values
        for coord in coords:
            qcmask |= np.ma.getmaskarray(coord.data)

            if coord.data.dtype != 'object':
                qcmask |= np.isnan(coord.data)

        for data in all_data:
            data[qcmask] = np.ma.masked

        return UngriddedBoundedCube(all_data, metadata, coords)

    @property
    def filenames(self):
        return self._filenames

    @abstractmethod
    def get_variable_names(self):
        """Returns names of available variables to load."""
        ...

    @abstractmethod
    def _create_bounded_coord_list(self):
        """Returns lat,lon,time with bounds."""
        ...

    @property
    def qcmask(self):
        try:
            return self._qcmask
        except AttributeError:
            self._qcmask = self._get_qcmask()
            return self._qcmask

    @abstractmethod
    def _get_qcmask(self):
        """Returns boolean array of points to mask out."""
        ...


