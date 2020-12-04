# REAME
## The Regularised Ensemble of Aerosol for Model Evaluation
A set of standardised aggregation methods to produce an ensemble of aerosol optical depth observations.

Over a rectilinear latitude-longitude grid (currently, taken from UKESM1), this calculates, for two wavelengths of aerosol optical depth the,
 - Joint distribution over 44 bins roughly log-spaced between 0 and 5,
 - Maximum observed aerosol optical depth,
 - Mean;
 - Mean of the base-10 logarithm.
The precise wavelengths vary by instrument but aim to be 550 and 670 nm, with 440 nm substituting the later when unavailable.

The code currently supports the following datasets:
 - Three datasets for AATSR and ATSR-2 developed by the European Space Agency's Aerosol CCI program:
   - ATSR Dual View (ADV), from the Finish Meteorological Institute;
   - Swansea University's (SU) retrieval, which is used operationally by the Copernicus Climate Change Service;
   - Optimal Retrieval of Aerosol and Cloud (ORAC), from the University of Oxford, RAL Space, and others.
 - Three datasets for MODIS Terra and Aqua funded by NASA:
   - Dark Target;
   - Deep Blue;
   - MAIAC.

To Do:
 - Remove the hard-coded paths.
 - Complete code to determine the number of pixels observed by each sensor.
 - Fit bi-log-normal distributions to the histograms.
 - Convert current `npz` output files into CF-compliant netCDF files.
 - Implement additional algorithms that are currently available on JASMIN:
   - Deep Blue for AVHRR and SeaWIFS;
   - The AVHRR product released by FIDUCEO;
   - MISR v23.
