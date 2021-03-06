#!/usr/bin/env python
"""Generate histograms of pixel coverage on the UKESM grid."""

if __name__ == '__main__':
    import numpy as np
    import os

    from argparse import ArgumentParser
    from datetime import timedelta
    from shapely.geometry import Polygon
    from sys import exit

    import reame.sensors
    from reame.classes import HistogramGrid
    from reame.utils import tabulate_geo_points, REFERENCE_TIME

    try:
        default_day_offset = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        default_day_offset = 0

    # Command line arguments
    pars = ArgumentParser(description="Generate histograms of coverage.")
    pars.add_argument("algorithm",
                      help="Name of Sensor class used to read the data.")
    inpt = pars.add_mutually_exclusive_group()
    inpt.add_argument("--in_dir", help="strftime-compatible path to input files.")
    inpt.add_argument("--files", nargs="+", default=[],
                      help="Manually override the list of files to process.")
    pars.add_argument("--out_name", default="geo_%Y-%m-%d.npz",
                      help="strftime-compatible path for output file.")
    pars.add_argument("--day_offset", type=int, default=default_day_offset,
                      help="Number of days from Jan 1 1995 to evaluate.")
    pars.add_argument("--grid", default="/home/users/acpovey/ukesm_grid.4.npz",
                      help="Path to NPZ file specifying the grid to use.")
    pars.add_argument("--clobber", action="store_true",
                      help="Overwrite existing output files.")
    pars.add_argument("--threshold", type=float, default=80.,
                      help="Largest solar zenith angle to consider.")
    args = pars.parse_args()


    date = REFERENCE_TIME + timedelta(days=args.day_offset)
    out_name = date.strftime(args.out_name)
    if os.path.isfile(out_name) and not args.clobber:
        print("Output already generated: " + out_name)
        exit()

    # Manually skim directory to skip over empty/dud files
    if len(args.files) == 0:
        try:
            with os.scandir(date.strftime(args.in_dir)) as it:
                for entry in it:
                    stats = entry.stat()
                    if (reame.sensors.IS_FILE[args.algorithm](entry.name)
                        and stats.st_size > 250):
                        args.files.append(entry.path)

            assert len(args.files) > 0
        except (FileNotFoundError, AssertionError):
            print("No files found in " + date.strftime(args.in_dir))
            exit()

    grid = HistogramGrid(args.grid)

    # Initialise arrays
    hist = np.zeros((grid.nlat, grid.nlon), dtype=float)

    for fname in args.files:
        sensor = getattr(reame.sensors, args.algorithm)(fname)
        data = sensor.create_bounded_data_object(["SolZen"])
        # Cubes aren't really designed for data manipulation...
        data[0] = np.ma.masked_greater(data[0].data, args.threshold)

        points = data.get_non_masked_bounded_points()
        output = tabulate_geo_points(points, grid)

        hist += output
        print(fname)

    # Save results
    save = dict(files=args.files, hist=hist, x=grid.lat_bins, y=grid.lon_bins)
    np.savez_compressed(out_name, **save)
