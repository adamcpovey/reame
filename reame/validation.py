"""Classes and methods to validate the contents of my histograms.

Classes:
FailedCheck: Child of UserWarning describing how checks have failed.
HistogramGrid: Container for the list of Polygons that represent the spatial
    grid over which the histograms are generated.
HistogramValidator: Object to iterate over expected files for a given product
    and sensor combination.
HistogramValidation: Methods to validate a single histogram file.
"""

import numpy as np
import os


# Folder for histogram repository
_BASE_DIR = "/group_workspaces/jasmin2/nceo_aerosolfire/acpovey/histograms"
# Path of grid definition
_GRID_PATH = "/home/users/acpovey/ukesm_grid.4.npz"
# Dictionary of supported algorithms
_MEMBER_INFO = {
    # For each, a dictionary of supported sensors
    "ORAC_v4.01": {
        # For each, a tuple of [0] first offset, [1] last offset,
        # [2] strftime path to L1 files, [3] strfpath to geoloc files
        "slstr-a": (8217, 9190, "/gws/nopw/j04/eo_shared_data_vol1/satellite/slstr-a/aerosol_cci/ORAC_v4.01_l2/%Y/%Y_%m_%d", ["/neodc/sentinel3a/data/SLSTR/L1_RBT/%Y/%m/%d/*N1"]),
        "slstr-b": (8947, 9130, "/gws/nopw/j04/eo_shared_data_vol1/satellite/slstr-b/aerosol_cci/ORAC_v4.01_l2/%Y/%Y_%m_%d", ["/neodc/sentinel3b/data/SLSTR/L1_RBT/%Y/%m/%d/*E2"]),
        "aatsr": (2696, 6307, "/gws/nopw/j04/eo_shared_data_vol1/satellite/aatsr/aerosol_cci/ORAC_v4.01_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p/%Y/%m/%d/*N1"]),
        "atsr2": (151, 3094, "/gws/nopw/j04/eo_shared_data_vol1/satellite/atsr2/aerosol_cci/ORAC_v4.01_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p/%Y/%m/%d/*E2"]),
    },
    "SU_v4.3": {
        "aatsr": (2761, 6307, "/gws/nopw/j04/eo_shared_data_vol1/satellite/aatsr/aerosol_cci/SU_v4.3_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p/%Y/%m/%d/*N1"]),
        "atsr2": (212, 3094, "/gws/nopw/j04/eo_shared_data_vol1/satellite/atsr2/aerosol_cci/SU_v4.3_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p/%Y/%m/%d/*E2"]),
    },
    "ADV_v2.31": {
        "aatsr": (2761, 6307, "/gws/nopw/j04/eo_shared_data_vol1/satellite/aatsr/aerosol_cci/ADV_v2.31_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/aatsr-v3/data/ats_toa_1p/%Y/%m/%d/*N1"]),
        "atsr2": (151, 2921, "/gws/nopw/j04/eo_shared_data_vol1/satellite/atsr2/aerosol_cci/ADV_v2.31_l2/%Y/%Y_%m_%d", ["/neodc/aatsr_multimission/atsr2-v3/data/at2_toa_1p/%Y/%m/%d/*E2"]),
    },
    "DarkTarget_c61": {
        "MOD": (1879, 8977, "/neodc/modis/data/MOD04_L2/collection61/%Y/%m/%d", ["/neodc/modis/data/MOD03/collection61/%Y/%m/%d/*hdf"]),
        "MYD": (2740, 8977, "/neodc/modis/data/MYD04_L2/collection61/%Y/%m/%d", ["/neodc/modis/data/MYD03/collection61/%Y/%m/%d/*hdf"]),
    },
    "DeepBlue_c61": {
        "MOD": (1879, 8977, "/neodc/modis/data/MOD04_L2/collection61/%Y/%m/%d", ["/neodc/modis/data/MOD03/collection61/%Y/%m/%d/*hdf"]),
        "MYD": (2740, 8977, "/neodc/modis/data/MYD04_L2/collection61/%Y/%m/%d", ["/neodc/modis/data/MYD03/collection61/%Y/%m/%d/*hdf"]),
    },
    "MAIAC_c6": {
        "MCD": (1882, 8977, "/neodc/modis/data/MCD19A2/collection6/%Y/%m/%d", ["/neodc/modis/data/MOD03/collection61/%Y/%m/%d/*hdf", "/neodc/modis/data/MYD03/collection61/%Y/%m/%d/*hdf"]),
    },
}
_MEMBER_INFO["APORAC_v4.01"] = _MEMBER_INFO["ORAC_v4.01"]


class FailedCheck(UserWarning):
    pass


class HistogramValidator(object):
    """Convience for iterating over a set of histograms"""
    def __init__(self, product, sensor, fdr=_BASE_DIR):
        """Checks inputs are valid"""
        super(HistogramValidator, self).__init__()
        try:
            info = _MEMBER_INFO[product]
        except KeyError:
            raise NotImplementedError(
                f"Invalid product: {product}\nAvailable choices: " +
                ", ".join(_MEMBER_INFO.keys())
            )
        if sensor not in info:
            raise NotImplementedError(
                f"Invalid sensor for {product}: {sensor}\nAvailable choices: " +
                ", ".join(info.keys())
            )
        if not os.path.isdir(fdr):
            raise ValueError(f"Invalid folder: {fdr}")
        product_fdr = os.path.join(fdr, product)
        if not os.path.isdir(product_fdr):
            raise ValueError(f"Product folder unavailable: {product_fdr}")

        self._product = product
        self._sensor = sensor
        self._fdr = product_fdr

    def __repr__(self):
        return f"Product {self._product} evaluating {self._sensor} at {self._fdr}"

    def __getitem__(self, offset):
        """Return a specific file from the range"""
        if offset not in self.offset_range:
            raise IndexError(f"Index {offset} outside {self.offset_range}")
        return HistogramValidation(
            self._product, self._sensor, offset, self._fdr
        )

    def __iter__(self):
        """Iterate over full range"""
        for offset in self.offset_range:
            yield HistogramValidation(
                self._product, self._sensor, offset, self._fdr
            )

    @property
    def offset_range(self):
        """A range instance over all valid day offsets"""
        i, j, _, _ = _MEMBER_INFO[self._product][self._sensor]
        return range(i, j+1)


class HistogramValidation(object):
    """Container for things that could validate my histograms"""
    standard_checks = [
        "output_valid", "file_list", "total", "weights", "maxima", "hist_max", "orbit"
    ]

    def __init__(self, product, sensor, offset, fdr, atol=0., rtol=1e-5):
        """Naive initialisation, without error checking"""

        super(HistogramValidation, self).__init__()
        self._product = product
        self._sensor = sensor
        self._offset = offset
        self._fdr = fdr
        self.atol = atol
        self.rtol = rtol

    def __get__(self):
        """Evaluate all checks"""
        return all(self.__iter__())

    def __getitem__(self, index):
        """Run a check based on index"""
        return self.check(self.standard_checks[index])

    def __iter__(self):
        """Run all checks"""
        for key in self.standard_checks:
            yield self.check(key)

    def __len__(self):
        return len(self.standard_checks)

    def __repr__(self):
        return f"{self.date} (offset {self._offset}) for product {self._product} evaluating {self._sensor} at {self._fdr}"

    @property
    def algorithm(self):
        return self._product.split("_")[0]

    @property
    def date(self):
        """Return datetime for a given array index"""
        from datetime import timedelta
        from reame.utils import REFERENCE_TIME

        return REFERENCE_TIME + timedelta(days=self._offset)

    @property
    def filename(self):
        """Name of this histogram npz file"""
        return self.date.strftime(
            f"{self._product}_{self._sensor}_%Y-%m-%d.npz"
        )

    @property
    def grid(self):
        from reame.classes import HistogramGrid

        try:
            return self._grid
        except AttributeError:
            grid = HistogramGrid(self.path)
            grid.nlam = 2
            self._grid = grid
            return self._grid

    @property
    def path(self):
        """Full path to this histogram npz file"""
        return os.path.join(self._fdr, self.filename)

    @property
    def area_tabulation(self):
        from reame.utils import tabulate_data_points
        from reame.sensors import VAR

        try:
            return self._area_tabulation
        except AttributeError:
            variables = VAR[self.algorithm]
            prod = self.cis_product()
            data = prod.create_bounded_data_object(variables)
            points = data.get_non_masked_bounded_points()
            self._area_tabulation = tabulate_data_points(points, self.grid)
            return self._area_tabulation

    @property
    def point_tabulation(self):
        from reame.utils import tabulate_data_points
        from reame.sensors import VAR

        try:
            return self._point_tabulation
        except AttributeError:
            variables = VAR[self.algorithm]
            prod = self.cis_product()
            data = prod.create_data_object(variables)
            points = data.get_non_masked_points()
            self._point_tabulation = tabulate_data_points(points, self.grid)
            return self._point_tabulation

    @property
    def saved_tabulation(self):
        try:
            return self._saved_tabulation
        except AttributeError:
            self._saved_tabulation = (
                self.get("hist"), self.get("max_aod"),
                self.get("aod"), self.get("logaod")
            )
            return self._saved_tabulation

    @property
    def version(self):
        return self._product.split("_")[1]

    def cis_product(self):
        """Generates a BProduct instance for this histogram"""
        import reame.sensors

        constructor = getattr(reame.sensors, self.algorithm)
        try:
            _, _, _, geo_glob = _MEMBER_INFO[self._product][self._sensor]
            return constructor(self.file_list(), os.path.dirname(geo_glob[0]))
        except TypeError:
            return constructor(self.file_list())

    def notclose(self, a, b):
        return not np.allclose(a, b, atol=self.atol, rtol=self.rtol)

    def orbits(self):
        """Iterator over Orbit instances for this histogram"""
        import orbit
        from glob import iglob

        if self._sensor in ("atsr2", "aatsr"):
            constructor = orbit.ATSR
        elif self._sensor in ("slstr-a", "slstr-b"):
            constructor = orbit.SLSTR
        elif self._sensor in ("MOD", "MYD", "MCD"):
            constructor = orac.MODIS
        else:
            raise NotImplementedError(f"Unknown Orbit type: {self.algorithm}")

        _, _, _, geo_glob = _MEMBER_INFO[self._product][self._sensor]
        for fdr in geo_glob:
            for fname in iglob(self.date.strftime(fdr)):
                this_orbit = constructor(fname)
                this_orbit.central_longitude = self.grid.central_longitude
                yield this_orbit

    def file_list(self, segmented=True):
        """All files expected to be processed for this histogram"""
        from reame.sensors import IS_FILE
        is_file = IS_FILE[self.algorithm]

        _, _, fdr, _ = _MEMBER_INFO[self._product][self._sensor]
        try:
            with os.scandir(self.date.strftime(fdr)) as entries:
                for entry in entries:
                    stats = entry.stat()
                    if (is_file(entry.name) and stats.st_size > 250 and
                        (segmented or "_seg" not in entry.name)):
                        yield entry.path
        except FileNotFoundError:
            pass

    def get(self, key):
        """Fetch field from histogram npz file"""
        from zipfile import BadZipFile
        from running_mean import WeightedMean

        try:
            with np.load(self.path) as file_ob:
                if key+"_n" in file_ob:
                    return WeightedMean(dictionary=file_ob, label=key)
                else:
                    return file_ob[key]
        except IOError:
            raise FileNotFoundError(f"Unable to open {self.path}")
        except (AttributeError, BadZipFile):
            raise IOError(f"Corrupted file {self.path}")

    def check(self, key):
        """Wrapper for FileNotFoundError checking"""

        try:
            return getattr(self, "_check_" + key)()
        except FileNotFoundError:
            self.warn(f"Output {self.path} not present")
        except IOError:
            self.warn(f"Output {self.path} corrupted")
        except AttributeError:
            raise NotImplementedError(f"Check {key} does not exist")

        return False

    def warn(self, message):
        from warnings import warn

        warn(FailedCheck(message + ";, " + repr(self)))

    def _check_file_list(self, segmented=True):
        """Ensure all the files I desire were actually processed"""

        files_processed = self.get("files").tolist()

        try:
            for fname in self.file_list(segmented):
                files_processed.remove(fname)
        except ValueError:
            self.warn(f"File {fname} not processed")
            return False

        if len(files_processed) > 0:
            self.warn("Excess files processed: " + repr(files_processed))
            return False

        return True

    def _check_hist_max(self):
        """Ensure maximum AODs turn up in hist"""

        hist, max_aod, _, _ = self.saved_tabulation
        hist = hist.reshape((-1,) + hist.shape[2:4])
        max_aod = max_aod.reshape((-1, + max_aod.shape[2]))

        def valid_dist(dist, aod):
            if aod > -999.:
                k = np.searchsorted(self.grid.aod_bins, aod) - 1
                return dist[k] > 0 and dist[k+1:-1].sum() == 0
            else:
                return dist[:-1].sum() == 0

        for i, (point_hist, point_max) in enumerate(zip(hist, max_aod)):
            if not (valid_dist(point_hist.sum(axis=1), point_max[0]) and
                    valid_dist(point_hist.sum(axis=0), point_max[1])):
                point = np.unravel_index(i, self.grid.shape[0:2])
                self.warn("Inconsistent histogram and maxima at " + str(point))
                return False

        return True

    def _check_maxima(self):
        """Ensure global maxima is equal"""

        max_aod_stored = self.saved_tabulation[1]
        max_aod = self.point_tabulation[1]
        if max_aod_stored.max() == max_aod.max():
            return True

        self.warn("Incorrect maximum AOD")
        return False

    def _check_orbit(self):
        """Ensure all points within orbit"""
        from itertools import product

        edges = [e for o in self.orbits() for e in o.get_edges()]

        empty = []
        for i, j in product(range(self.grid.nlat), range(self.grid.nlon)):
            cell = self.grid[i,j]
            for e in edges:
                if cell.intersects(e):
                    empty.append(False)
                    break
            else:
                empty.append(True)

        empty = np.array(empty).reshape((self.grid.nlat, self.grid.nlon))
        hist = self.saved_tabulation[0].sum(axis=(2,3))
        # ATSR products are reported on a sinusoidal grid, which fuzzes the edges
        limit = 1 if self._sensor in ("atsr2", "aatsr") else 0
        bad_cell = np.logical_and(hist > limit, empty)

        if np.any(bad_cell):
            indices = np.nonzero(bad_cell)
            self.warn("Hits registered outside orbit at: "+str(indices))
            return False

        return True

    def _check_output_valid(self, segmented=True):
        """Ensure the output was generated"""

        try:
            if self.get("max_aod").max() == 0.:
                self.warn(f"Output {self.path} is empty")
                return False
            return True
        except FileNotFoundError:
            # See if there were any files to process in the first place
            files_processed = list(self.file_list(segmented))
            if len(files_processed) > 0:
                return False
            raise StopIteration

    def _check_tabulation(self):
        """Ensure the tabulation gives the same answer"""

        hist0, max_aod0, mean_aod0, mean_log0 = self.area_tabulation
        hist1, max_aod1, mean_aod1, mean_log1 = self.saved_tabulation
        if self.notclose(hist0, hist1):
            self.warn("Saved histogram inconsistent with current code")
            return False
        if self.notclose(max_aod0, max_aod1):
            self.warn("Saved maxima inconsistent with current code")
            return False
        if not mean_aod0 == mean_aod1:
            self.warn("Saved mean inconsistent with current code")
            return False
        if not mean_log0 == mean_log1:
            self.warn("Saved log mean inconsistent with current code")
            return False

        return True

    def _check_total(self):
        """Ensure histogram matches point tabulation over aod space"""
        from reame.sensors import VAR
        variables = VAR[self.algorithm]

        hist0 = self.point_tabulation[0]
        h_available = hist0.sum(axis=(0,1))
        hist1 = self.saved_tabulation[0]
        h_stored = hist1.sum(axis=(0,1))

        if not self.notclose(h_stored, h_available):
            return True

        n_stored = h_stored.sum()
        n_available = h_available.sum()
        self.warn(f"Incorrect histogram count: {n_stored} instead of {n_available}")
        return False

    def _check_weights(self):
        """Ensure weights on the means are consistent with the histograms"""

        hist, _, mean0, mean1 = self.saved_tabulation

        if self.notclose(mean0._W[:,:,0], hist[:,:,:-1,:].sum(axis=(2,3))):
            self.warn("Linear WeightedMean W for ch0 differs from hist")
            return False
        if self.notclose(mean0._W[:,:,1], hist[:,:,:,:-1].sum(axis=(2,3))):
            self.warn("Linear WeightedMean W for ch1 differs from hist")
            return False
        if self.notclose(mean1._W[:,:,0], hist[:,:,1:-1,:].sum(axis=(2,3))):
            self.warn("Log WeightedMean W for ch0 differs from hist")
            return False
        if self.notclose(mean1._W[:,:,1], hist[:,:,:,1:-1].sum(axis=(2,3))):
            self.warn("Log WeightedMean W for ch1 differs from hist")
            return False

        mean2 = self.point_tabulation[2]
        if not np.allclose(mean0._W.sum(), mean2.n.sum(),
                           atol=self.atol, rtol=self.rtol):
            self.warn("Total weights differ")
            return False

        return True
