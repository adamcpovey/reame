#!/usr/bin/env python
"""Generate histograms of AOD on a rectangular grid, opening a folder at once."""

if __name__ == '__main__':
    import numpy as np
    import os

    from argparse import ArgumentParser
    from datetime import timedelta
    from shapely.geometry import Polygon
    from sys import exit

    import reame.sensors
    from reame.classes import HistogramGrid
    from reame.utils import tabulate_data_points, REFERENCE_TIME


    try:
        default_day_offset = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        default_day_offset = 0

    # Command line arguments
    pars = ArgumentParser(description="Generate histograms of AOD.")
    pars.add_argument("algorithm",
                      help="Name of Sensor class used to read the data.")
    inpt = pars.add_mutually_exclusive_group()
    inpt.add_argument("--in_dir", help="strftime-compatible path to input files.")
    inpt.add_argument("--files", nargs="+", default=[],
                      help="Manually override the list of files to process.")
    pars.add_argument("--mod03_path",
                      help="Path to MODIS geolocation files.")
    pars.add_argument("--grid_path",
                      help="Path to MAIAC geolocation files.")
    pars.add_argument("--out_name", default="hist_%Y-%m-%d.npz",
                      help="strftime-compatible path for output file.")
    pars.add_argument("--day_offset", type=int, default=default_day_offset,
                      help="Number of days from Jan 1 1995 to evaluate.")
    pars.add_argument("--grid", default="/home/users/acpovey/ukesm_grid.4.npz",
                      help="Path to NPZ file specifying the grid to use.")
    pars.add_argument("--clobber", action="store_true",
                      help="Overwrite existing output files.")
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
    grid.nlam = 2 # Hardcoded number of wavelengths

    kwargs = {}
    if args.mod03_path:
        kwargs["mod03_path"] = args.mod03_path
    if args.grid_path:
        kwargs["grid_path"] = args.grid_path

    # Fetch appropriate class
    sensor = getattr(reame.sensors, args.algorithm)(args.files, **kwargs)

    variables = reame.sensors.VAR[args.algorithm]
    data = sensor.create_bounded_data_object(variables)
    points = data.get_non_masked_bounded_points()
    hist, max_aod, mean_aod, mean_log = tabulate_data_points(points, grid)

    # Save results
    save = dict(files=args.files, hist=hist, max_aod=max_aod, l=variables,
                x=grid.lat_bins, y=grid.lon_bins, z=grid.aod_bins)
    mean_aod.save(save)
    mean_log.save(save)
    np.savez_compressed(out_name, **save)
