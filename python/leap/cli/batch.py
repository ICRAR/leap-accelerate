import io
import logging
from pathlib import Path
import sys
from typing import Annotated

import astropy.units as u
import numpy as np
from xarray import Dataset
from astropy.coordinates import ICRS, SkyCoord
from astropy.units import Quantity
from matplotlib import pyplot as plt
from casacore import tables
import typer
import functools
import json
import csv
import contextlib

from leap.functional import calibrate_leap_gain_tables
from leap.cli.player import FuncPlayer
from leap.cli.parsers import az_el, meters, log_level, video_filepath
from mpl_toolkits.mplot3d.axes3d import Axes3D

batch_app = typer.Typer()

outfile_option = typer.Option(
    "--outfile", "-o",
    help="the output video filepath",
    writable=True, parser=video_filepath
)


@batch_app.command(name="plot")
def plot_command(
    ms: Path,
    outfile: Annotated[Path | None, outfile_option] = None,
    impl: Annotated[str, typer.Option("--impl", "-i", parser=str)] = "cpu",
    start: Annotated[int, typer.Option("--start", "-s", parser=int)] = 0,
    end: Annotated[int | None, typer.Option("--end", "-e", parser=int)] = None,
    interval: Annotated[int, typer.Option("--interval", "-n", parser=int)] = None,
    min_baseline_threshold: Annotated[Quantity, typer.Option("--min_baseline_threshold", "-m", parser=meters)] = "0.0m",
    directions: Annotated[list, typer.Option("--directions", "-d", parser=az_el)] = "[[0.0, 0.0]]",
    verbosity: Annotated[int, typer.Option("--verbosity", "-v", parser=log_level)] = "WARNING"
):
    # calibration
    obs_directions = [
        SkyCoord([direction], frame=ICRS, unit=u.rad) for direction in directions
    ]
    solution_interval = slice(start, end, interval)
    gain_tables: list[Dataset] = calibrate_leap_gain_tables(
        str(ms),
        obs_directions,
        solution_interval=solution_interval,
        impl=impl,
        min_baseline_threshold=min_baseline_threshold,
        verbosity=verbosity,
    )

    # flags
    with tables.table(str(ms), readonly=True, ack=False) as ms_main:
        antenna_ids = list(set(ms_main.getcol("ANTENNA1")).union(set(ms_main.getcol("ANTENNA2"))))

    # uvws
    with tables.table(str(ms / "ANTENNA"), readonly=True, ack=False) as ms_antenna:
        xyzs = ms_antenna.getcol("POSITION")[antenna_ids]
    logging.info(xyzs)
    xs = xyzs[:, 0]
    ys = xyzs[:, 1]

    axes: list[Axes3D] = []  # direction
    plots: list[plt.Figure] = []  # direction
    zs: list[list] = []  # direction, frame

    for d_idx, gain_table in enumerate(gain_tables):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        assert isinstance(ax, Axes3D)
        axes.append(ax)

        # calculates zs
        zs.append([])
        for time_idx, time in enumerate(gain_table.coords['time']):
            zs[d_idx].append(
                np.angle(gain_table.data_vars["gain"][time_idx, antenna_ids, 0, 0, 0])
            )

        ax.xaxis.set_label("x")
        ax.xaxis.set_major_formatter('{x} m')
        ax.yaxis.set_label("y")
        ax.yaxis.set_major_formatter('{x} m')
        ax.zaxis.set_label("phase")
        ax.zaxis.set_major_formatter("{x} rad")
        ax.set_zlim(-np.pi, np.pi)
        ax.set_title(str(gain_table.attrs["phasecentre"]))

        # initial plots
        plots.append(ax.plot_trisurf(
            xs, ys, zs[d_idx][0],
            cmap="seismic",
            vmin=-np.pi,
            vmax=np.pi,
            linewidth=1
        ))
        fig.colorbar(plots[d_idx])

        # TODO: plot to seperate non-interactive figures and
        # save to image

        # update plot
        def set_frame(time_idx, d_idx):
            plots[d_idx].remove()
            plots[d_idx] = axes[d_idx].plot_trisurf(
                xs, ys, zs[d_idx][time_idx],
                cmap="seismic",
                vmin=-np.pi, vmax=np.pi,
                linewidth=1
            )

        player = FuncPlayer(
            fig,
            functools.partial(set_frame, d_idx=d_idx),
            mini=0,
            maxi=len(zs[d_idx]) - 1,
            cache_frame_data=True,
            save_count=len(zs[d_idx]) - 1,
            interval=100
        )

        if outfile:
            if outfile.suffix == ".png":
                # images
                for t_idx in range(len(zs[d_idx]) - 1):
                    fig.savefig(outfile.parent / f"d{d_idx}_t{t_idx}_{outfile.stem}{outfile.suffix}")
                    player.oneforward()
            else:
                # video
                player.save(outfile.parent / f"d{d_idx}_{outfile.stem}{outfile.suffix}")

        if not outfile:
            plt.show()


@batch_app.command(name="dump")
def dump_command(
    ms: Path,
    out_file: Annotated[Path | None, typer.Option("--out_file", "-o", parser=Path)] = None,
    stat_file: Annotated[Path | None, typer.Option("--stat_file", "-t", parser=Path)] = None,
    impl: Annotated[str, typer.Option("--impl", "-i", parser=str)] = "cpu",
    start: Annotated[int, typer.Option("--start", "-s", parser=int)] = 0,
    end: Annotated[int | None, typer.Option("--end", "-e", parser=int)] = None,
    interval: Annotated[int, typer.Option("--interval", "-n", parser=int)] = 1,
    min_baseline_threshold: Annotated[Quantity, typer.Option("--min_baseline_threshold", "-m", parser=meters)] = "0.0m",
    directions: Annotated[list, typer.Option("--directions", "-d", parser=az_el)] = "[[0.0, 0.0]]",
    verbosity: Annotated[int, typer.Option("--verbosity", "-v", parser=log_level)] = "WARNING"
):
    # calibration
    obs_directions = [
        SkyCoord([direction], frame=ICRS, unit=u.rad) for direction in directions
    ]
    solution_interval = slice(start, end, interval)
    gain_tables: list[Dataset] = calibrate_leap_gain_tables(
        str(ms),
        obs_directions,
        solution_interval=solution_interval,
        impl=impl,
        min_baseline_threshold=min_baseline_threshold,
        verbosity=verbosity,
    )

    dump_calibration_json(
        gain_tables,
        out_file,
        impl,
        start,
        end,
        interval,
        min_baseline_threshold,
        directions
    )

    dump_stats_csv(
        gain_tables,
        stat_file
    )


def dump_calibration_json(
    gain_tables: list[Dataset], out_file: Path | None,
    impl,
    start,
    end,
    interval,
    min_baseline_threshold,
    directions
):
    """dump calibration to json document"""
    result = {
        "impl": impl,
        "start": start,
        "end": end,
        "interval": interval,
        "min_baseline_threshold": str(min_baseline_threshold),
        "directions": directions
    }
    result["command"] = " ".join([arg if " " not in arg else f"'{arg}'" for arg in sys.argv])
    result["gainCalibrations"] = []
    for table in gain_tables:
        direction = {
            "direction": {
                "ra": list(table.attrs["phasecentre"].ra.rad),
                "dec": list(table.attrs["phasecentre"].dec.rad),
            },
            "solutions": []
        }
        for time_idx, time in enumerate(table.coords["time"]):
            solution = {
                "time": time.to_numpy()[()],
                "antennaPhases": np.angle(table.data_vars["gain"][time_idx, :, 0, 0, 0]).tolist()
            }
            direction["solutions"].append(solution)
        result["gainCalibrations"].append(direction)

    with text_ostream_context(out_file) as out_stream:
        json.dump(result, out_stream, indent=2)


def dump_stats_csv(gain_tables: list[Dataset], out_file: Path | None):
    with text_ostream_context(out_file) as out_stream:
        writer = csv.writer(out_stream, delimiter=' ')
        writer.writerow(['ra', 'dec', 'time', 'nantenna', 'mean', 'rms', 'stddev', 'rmse','plane_orig', 'gradx', 'grady'])
        
        for table in gain_tables:
            direction = (table.attrs['phasecentre'].ra.rad, table.attrs['phasecentre'].dec.rad)
            for time_idx, time in enumerate(table.coords['time'].to_numpy()):
                phases = np.angle(table.data_vars["gain"][time_idx, :, :, :, :])
                nant=len(phases)
                rms = np.sqrt(np.mean(phases**2))
                plane_orig=0;gradx=0;grady=0; # plane parameters -- to be updated in the future
                mean = np.mean(phases)
                stddev = np.std(phases)
                rmse = stddev  # rms - phase_plane
                writer.writerow([direction[0][0], direction[1][0], time, nant, mean, rms, stddev, rmse, plane_orig, gradx, grady])


def text_ostream_context(out_file: Path | None) -> io.TextIOBase:
    """Creates an opening and closing context for a text-based output stream.

    Args:
        out_file (Path | None): File path for text file writing. None indicates using stdout.

    Returns:
        _type_: Returns a context for an output stream.
    """
    if not out_file:
        out_stream = contextlib.nullcontext(sys.stdout)
    else:
        out_stream = open(out_file, 'w+', newline='', encoding='utf-8')
    return out_stream
