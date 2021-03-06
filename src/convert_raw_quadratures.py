#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import glob
import h5py
import logging
from numpy import float32
import scipy
import sys
from tools import create_dataset


def read_tabular_data(filename, dtype=float32):
    table = []
    with open(filename, "U") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            data = stripped_line.replace(",", ".").replace(";", "").replace("x", "\t").split("\t")
            table.append(list(map(dtype, data)))
    return scipy.array(table, dtype=dtype)


def read_information(basename):
    try:
        base = read_tabular_data(basename)
    except IOError:
        logging.warning("Basefile missing.")
    try:
        no_scans = read_tabular_data(basename+"-nscans", int)[0, 0]
    except IOError:
        logging.warning("-nscans file missing. Determining no_scans from data files.")
        no_scans = 0
    try:
        no_pulses_per_angle, no_angles = read_tabular_data(basename+"-pulses", int)[0]
    except IOError:
        logging.warning("-pulses file missing. Determining no_angles and no_pulses from first data file.")
        no_angles, no_pulses_per_angle = read_tabular_data(basename+"-scan-0-step-0").shape
    try:
        timestep_size, no_timesteps = read_tabular_data(basename+"-step", int)[0]
    except IOError:
        timestep_size = scipy.nan
        no_timesteps = 1
    fns = glob.glob(basename+"-scan-*-step-*")
    scans = set()
    steps = {}
    for fn in fns:
        data = fn[len(basename):].split("-")
        scan = int(data[2])
        step = int(data[4])
        scans.add(scan)
        try:
            step_list = steps[scan]
        except KeyError:
            step_list = []
            steps[scan] = step_list
        step_list.append(step)
    for scan in scans:
        steps[scan] = sorted(steps[scan])
    all_steps = list(steps.values())
    reference_steps = all_steps[0]
    for s in all_steps[1:]:
        if s != reference_steps:
            raise RuntimeError("Not all scans use the same number of steps. "
                               "This does not fit into a table.")
    if reference_steps != list(range(no_timesteps)):
        logging.warning("The actually used steps are not those described in the -step file.")
        no_timesteps = len(reference_steps)
        if (reference_steps != scipy.arange(no_timesteps)).any():
            raise RuntimeError("Timesteps are not consecutive. Aborting.")
    if scans != set(range(no_scans)):
        logging.warning("The actually used scans are not those described in the -nscans file.")
        no_scans = len(scans)
        if scans != set(range(no_scans)):
            raise RuntimeError("Scans are not consecutive. Aborting.")
    shape = (no_scans, no_timesteps, no_angles, no_pulses_per_angle)
    return shape


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("basename", help="basename of quadrature data")
    parser.add_argument("-f", "--force",
                        help="overwrite previous reconstructions (default: %(default)s)",
                        action="store_true")
    parser.add_argument("-v", "--vacuum", help="filename of vacuum data; "
                        "typically ends in -scan-0-step-0", required=True)
    parser.add_argument("-e", "--eta", help="detection efficiency eta (default: %(default).2f)",
                        type=float, default=.8)
    return parser.parse_args()


def import_data(args, ds):
    (no_scans, no_timesteps, no_angles, no_pulses_per_angle) = ds.shape
    for scan in range(no_scans):
        print("Starting scan {} of {}:".format(scan, no_scans))
        for step in range(no_timesteps):
            fn = "{}-scan-{}-step-{}".format(args.basename, scan, step)
            data = read_tabular_data(fn)
            if data.shape == ds.shape[2:]:
                ds[scan, step, :, :] = data
            else:
                logging.error("Shape mismatch. Either number of angles or "
                              "pulses per angle are inconsistent. "
                              "Trying to pad with nans.")
                no_angles, no_pulses = data.shape
                ds[scan, step, :no_angles, :no_pulses] = data
                ds[scan, step, no_angles:, :no_pulses] = scipy.nan
                ds[scan, step, :, no_pulses:] = scipy.nan
            sys.stderr.write("\r{0:3.2%}".format(float(step)/no_timesteps))
        sys.stderr.write("\r100.00%\n")


def import_vacuum(args, ds):
    fn = args.vacuum
    print("Importing vacuum data.")
    data = read_tabular_data(fn)
    if data.shape == ds.shape:
        ds[:, :] = data
    else:
        logging.error("Shape mismatch. Either number of angles or "
                      "pulses per angle are inconsistent. "
                      "Trying to pad with nans.")
        no_angles, no_pulses = data.shape
        ds[:no_angles, :no_pulses] = data
        ds[no_angles:, :no_pulses] = scipy.nan
        ds[:, no_pulses:] = scipy.nan


def main():
    args = parse_args()
    shape = read_information(args.basename)
    if args.force:
        mode = "w"
    else:
        mode = "w-"
    with h5py.File(args.basename+".h5", mode) as h5:
        ds_v = create_dataset(args, h5, "vacuum_quadratures", shape[2:])
        import_vacuum(args, ds_v)
        ds_q = create_dataset(args, h5, "raw_quadratures", shape)
        ds_q.attrs.create("eta", args.eta, dtype=float32)
        import_data(args, ds_q)


if __name__ == "__main__":
    main()
