#!/usr/bin/env python
# Make monthly files from daily histograms

import numpy as np
import os

from argparse import ArgumentParser
from itertools import product

from running_mean import WeightedMean

def intersection(a, b):
    """Intersection of two strings, from the start."""
    i = len(a)
    while a[:i] != b[:i]: i -= 1
    return a[:i]


pars = ArgumentParser()
pars.add_argument('files', nargs="+")
pars.add_argument('--out_dir', default=None)
pars.add_argument('--tidy_name', action="store_true")
pars.add_argument('--plot', action="store_true")
args = pars.parse_args()

for f in args.files:
    with np.load(f) as sv:
        try:
            files += list(sv["files"])
            hist += sv["hist"]
            #max_aod = np.max(np.stack((max_aod, sv["max_aod"])), axis=0)
            max_aod = np.maximum(max_aod, sv["max_aod"])
            try:
                mean_aod += WeightedMean(dictionary=sv, label="aod")
                mean_log += WeightedMean(dictionary=sv, label="logaod")
            except KeyError:
                pass
            assert np.all(l == sv["l"])
            assert np.all(x == sv["x"])
            assert np.all(y == sv["y"])
            assert np.all(z == sv["z"])
            out_name = intersection(out_name, f)
        except NameError:
            files = list(sv["files"])
            hist = sv["hist"]
            max_aod = sv["max_aod"]
            try:
                mean_aod = WeightedMean(dictionary=sv, label="aod")
                mean_log = WeightedMean(dictionary=sv, label="logaod")
            except KeyError:
                pass
            l = sv["l"]
            x = sv["x"]
            y = sv["y"]
            z = sv["z"]
            out_name = f

if out_name[-1] == "/":
    out_name = files[0]

if args.out_dir is not None:
    out_name = os.path.join(args.out_dir, os.path.basename(out_name))

if args.tidy_name:
    if (out_name.endswith("-0") or out_name.endswith("-1")
        or out_name.endswith("-2")):
        out_name = out_name[:-2]
    elif out_name.endswith("_19") or out_name.endswith("_20"):
        out_name = out_name[:-3]
    elif out_name.endswith("_199") or out_name.endswith("_200") or out_name.endswith("_201"):
        out_name = out_name[:-4]
    out_name = out_name.strip("-_.")

try:
    save = dict(files=files, hist=hist, max_aod=max_aod, l=l, x=x, y=y, z=z)
    try:
        mean_aod.save(save)
        mean_log.save(save)
    except NameError:
        pass
    np.savez_compressed(out_name + ".npz", **save)
except NameError:
    pass

if args.plot:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    from matplotlib.colors import LogNorm

    sns.set_style(
        "whitegrid", {"axes.axisbelow" : False, "font.family":"serif",
                      "font.serif":["Times New Roman"]}
    )
    sns.set_palette("tab10", 10)
    mpl.rcParams["mathtext.fontset"] = "stix"

    fig = plt.figure(figsize=(10, 20))

    for i, (aod, label) in enumerate(zip(
            np.rollaxis(mean_aod.mean(), 2), ("AOD550", "AOD670")
    )):
        ax = fig.add_subplot(5, 2, i+1, projection=ccrs.Robinson())
        ax.coastlines()
        ax.gridlines()
        ax.set_title(os.path.basename(out_name))
        im = ax.imshow(
            aod, norm=LogNorm(vmin=0.01), cmap="viridis", origin="lower",
            transform=ccrs.PlateCarree(central_longitude=180.)
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", label="Mean "+label)

    for i, (aod, label) in enumerate(zip(
            10.**np.rollaxis(mean_log.mean(), 2), ("AOD550", "AOD670")
    )):
        ax = fig.add_subplot(5, 2, i+3, projection=ccrs.Robinson())
        ax.coastlines()
        ax.gridlines()
        im = ax.imshow(
            aod, norm=LogNorm(vmin=0.01), cmap="viridis", origin="lower",
            transform=ccrs.PlateCarree(central_longitude=180.)
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", label="Log mean "+label)

    for i, (aod, label) in enumerate(zip(
           np.rollaxis(max_aod, 2), ("AOD550", "AOD670")
    )):
        ax = fig.add_subplot(5, 2, i+5, projection=ccrs.Robinson())
        ax.coastlines()
        ax.gridlines()
        im = ax.imshow(
            aod, norm=LogNorm(vmin=0.01), cmap="viridis", origin="lower",
            transform=ccrs.PlateCarree(central_longitude=180.)
        )
        fig.colorbar(im, ax=ax, orientation="horizontal", label="Maximal "+label)

    ax = fig.add_subplot(5, 2, 7, projection=ccrs.Robinson())
    ax.coastlines()
    ax.gridlines()
    im = ax.imshow(
        hist.sum(axis=(2,3)), norm=LogNorm(vmin=1), cmap="viridis", origin="lower",
        transform=ccrs.PlateCarree(central_longitude=180.)
    )
    fig.colorbar(im, ax=ax, orientation="horizontal", label="Pixel count")

    ax = fig.add_subplot(5, 2, 8, xscale="log", yscale="log")
    ax.set_xlabel("AOD670")
    ax.set_ylabel("AOD550")
    t = z.copy()
    t[0] = 0.008
    t[-2] = 1.1
    t[-1] = 2.0
    xx, yy = np.meshgrid(t, t)
    im = ax.pcolormesh(
        xx, yy, hist.sum(axis=(0,1)), norm=LogNorm(vmin=1), cmap="viridis"
    )
    ax.vlines(t[[1,-2]], *ax.get_ylim())
    ax.hlines(t[[1,-2]], *ax.get_xlim())
    fig.colorbar(im, label="Pixel count")

    ax = plt.subplot(5, 2, 9, xscale="log")
    ax.set_xlabel("AOD550")
    ax.set_ylabel("Latitude")
    xx, yy = np.meshgrid(t, x)
    im = ax.pcolormesh(
        xx, yy, hist.sum(axis=(1,3)), norm=LogNorm(vmin=1), cmap="viridis"
    )
    ax.vlines(t[[1,-2]], *ax.get_ylim())
    fig.colorbar(im, label="Pixel count")

    ax = plt.subplot(5, 2, 10, yscale="log")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("AOD550")
    xx, yy = np.meshgrid(y, t)
    im = ax.pcolormesh(
        xx, yy, hist.sum(axis=(0,3)).T, norm=LogNorm(vmin=1),
        cmap="viridis"
    )
    ax.hlines(t[[1,-2]], *ax.get_xlim())
    fig.colorbar(im, label="Pixel count")

    fig.tight_layout()
    fig.savefig(out_name + ".pdf", bbox_inches="tight", dpi=150)
