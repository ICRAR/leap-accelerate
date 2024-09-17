from pathlib import Path
from typing import Annotated

import asyncio
import logging
import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.units import Quantity
import typer

from leap.functional import leap_stream_calibrate
from leap.cli.parsers import meters, az_el, log_level


realtime_app = typer.Typer()


@realtime_app.command(name="print")
def print_command(
    ms: Path,
    impl: Annotated[str, typer.Option("--impl", "-i", parser=str)] = "cpu",
    start: Annotated[int, typer.Option("--start", "-s", parser=int)] = 0,
    end: Annotated[int | None, typer.Option("--end", "-e", parser=int)] = None,
    interval: Annotated[int, typer.Option("--interval", "-n", parser=int)] = 1,
    min_baseline_threshold: Annotated[Quantity, typer.Option("--min_baseline_threshold", "-m", parser=meters)] = "0.0m",
    directions: Annotated[list, typer.Option("--directions", "-d", parser=az_el)] = "[[0.0, 0.0]]",
    verbosity: Annotated[int, typer.Option("--verbosity", "-v", parser=log_level)] = "WARNING"
):
    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        level=verbosity
    )

    # calibration
    obs_directions = [
        SkyCoord([direction], frame=ICRS, unit=u.rad) for direction in directions
    ]
    solution_interval = slice(start, end, interval)

    async def realtime_print_calibrations():
        async for calibration in leap_stream_calibrate(
            str(ms),
            obs_directions,
            solution_interval=solution_interval,
            impl=impl,
            min_baseline_threshold=min_baseline_threshold,
            verbosity=verbosity,
        ):
            start_epoch, end_epoch = (calibration.start_epoch, calibration.start_epoch)
            logging.info("got solution for timerange %s %s", start_epoch, end_epoch)
            beam_calibrations = calibration.beam_calibrations
            for beam_calibration in beam_calibrations:
                print("direction:", beam_calibration.direction)
                print("antennaPhases:", beam_calibration.antenna_phases)

    asyncio.run(realtime_print_calibrations())
