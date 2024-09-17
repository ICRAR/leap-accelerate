

import numpy as np
import os
from pathlib import Path

import casacore.tables as tables
import xarray
import astropy
import asyncio
import time

from astropy.coordinates import SkyCoord, FK5
import astropy.units as u


import logging

from leap.functional import leap_stream_calibrate

MWA_MS = Path(
    os.path.dirname(__file__),
    "../../../testdata/mwa/1197638568-split4-flagged.ms"
).resolve()


async def amain():
    ms_field = tables.table(str(MWA_MS/"FIELD"), readonly=True, ack=False)
    ms_phase_center = ms_field.getcell("PHASE_DIR", 0)
    ms_units = ms_field.getcoldesc("PHASE_DIR")["keywords"]["QuantumUnits"]
    obs_phase_center = SkyCoord(ms_phase_center, frame=FK5, unit=ms_units)

    obs_directions = [
        SkyCoord([[7.09767229e-01, -1.98773609e-04]], frame=FK5, unit=u.rad)
    ]

    offsets = [
        obs_phase_center.spherical_offsets_to(obs_direction) for obs_direction in obs_directions
    ]
    print(offsets)

    directions = np.array([[d.ra.rad, d.dec.rad] for d in obs_directions]).squeeze(axis=2)
    async for calibration in leap_stream_calibrate(
        str(MWA_MS),
        directions=directions,
        impl="cpu",
        verbosity=logging.WARNING
    ):
        start_epoch, end_epoch = (calibration.start_epoch, calibration.end_epoch)
        logging.info(
            "got solution for timerange %s %s",
            start_epoch,
            end_epoch
        )
        beam_calibrations = calibration.beam_calibrations
        for beam_calibration in beam_calibrations:
            logging.info(beam_calibration.direction)
            logging.info(beam_calibration.antenna_phases.shape)

        # some slow operation
        time.sleep(1)


def main():
    logging.basicConfig(
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    asyncio.run(amain())


if __name__ == "__main__":
    main()
