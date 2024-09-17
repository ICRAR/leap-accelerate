"""LEAP utility functions"""
import asyncio
import concurrent.futures
import logging

import astropy.units as u
import casacore.tables as tables
import numpy as np
import xarray
from astropy.coordinates import FK5, SkyCoord
from astropy.units import Quantity
from astropy.time import Time, TimeDelta

import leap


def leap_calibrate(
    ms_path: str,
    directions: np.ndarray,
    solution_interval=slice(0, None, 1),
    min_baseline_threshold: Quantity = Quantity(0, u.m),
    impl: str = 'cpu',
    verbosity: int = logging.WARNING,
) -> list[leap.Calibration]:
    """Calibrates selected visibilities to an output collection.

    Args:
        ms_path (str): Filesystem path to a measurement set with visibilities to calibrate.
        directions (np.ndarray): Set of ra/dec directions to calibrate.
        solution_interval (slice, optional): Constigous timestep intervals to calculate solutions for. Defaults to slice(0, None, 1).
        min_baseline_threshold (float): minimum baseline threshold for selecting baselines.
        impl (leap.ComputeImplementation | str, optional): Compute backend to use (cpu or cuda). Defaults to "cpu".

    Returns:
        list[leap.Calibration]: collection of output leap calibrations in time order.
    """
    outputs = []
    leap.LeapCalibrator(impl, verbosity).calibrate(
        ms_path=ms_path,
        directions=directions,
        solution_interval=solution_interval,
        min_baseline_threshold=min_baseline_threshold.to(u.m).value,
        callback=outputs.append,
    )
    return outputs


async def leap_stream_calibrate(
    ms_path: str,
    obs_directions: list[SkyCoord],
    solution_interval=slice(0, None, 1),
    min_baseline_threshold: Quantity = Quantity(0, u.m),
    impl: str = "cpu",
    verbosity: int = logging.WARNING
):
    """Calibrate using an async stream and a background worker thread.

    Args:
        ms_path (str): Filesystem path to a measurement set with visibilities to calibrate.
        obs_directions (list[SkyCoord]): Set of ra/dec directions to calibrate.
        solution_interval (slice, optional): Constigous timestep intervals to calculate solutions for. Defaults to slice(0, None, 1).
        min_baseline_threshold (float): minimum baseline threshold for selecting baselines.
        impl (leap.ComputeImplementation | str, optional): Compute backend to use (cpu or cuda). Defaults to "cpu".

    Yields:
        leap.Calibration: output leap calibrations in time order.
    """
    class StopSentinal:
        """Unique sentinal class for signalling the end of iteration"""

    stop_sentinel = StopSentinal()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

    outputs: asyncio.Queue[leap.Calibration | StopSentinal] = asyncio.Queue()

    directions = np.array([[d.ra.rad, d.dec.rad] for d in obs_directions]).squeeze(axis=2)

    def callback(value: leap.Calibration):
        asyncio.run_coroutine_threadsafe(outputs.put(value), loop)

    def run_leap():
        leap.LeapCalibrator(impl, verbosity).calibrate(
            ms_path=ms_path,
            directions=directions,
            solution_interval=solution_interval,
            min_baseline_threshold=min_baseline_threshold.to(u.m).value,
            callback=callback
        )

    async def arun_executor():
        logging.info("running calibrator in background thread")
        await loop.run_in_executor(executor, run_leap)
        logging.info("calibrator background thread done")
        await outputs.put(stop_sentinel)

    leap_task = asyncio.create_task(arun_executor())

    while True:
        v = await outputs.get()
        if not isinstance(v, StopSentinal):
            yield v
        else:
            break
    await leap_task


def calibrate_leap_gain_tables(
    ms_path: str,
    obs_directions: list[SkyCoord],
    solution_interval: slice = slice(0, None, 1),
    min_baseline_threshold: Quantity = Quantity(0, u.m),
    impl: str = "cpu",
    verbosity: int = logging.WARNING
) -> list[xarray.Dataset]:
    """
    Performs synchronous LEAP calibration and generates a gain table per direction.

    see https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/blob/main/src/ska_sdp_datamodels/calibration/calibration_model.py

    Args:
        ms_path (str): Filesystem path to a measurement set with visibilities to calibrate.
        obs_directions (list[SkyCoord]): Set of ra/dec directions to calibrate.
        impl (leap.ComputeImplementation | str, optional): Compute backend to use (cpu or cuda). Defaults to "cpu".

    Returns:
        list[xarray.Dataset]: xarray Dataset of conforming to SKA GainTable layout.
    """

    ms = tables.table(ms_path, readonly=True, ack=False)
    ms_antenna = tables.table(f"{ms_path}/ANTENNA", readonly=True, ack=False)
    ms_field = tables.table(f"{ms_path}/FIELD", readonly=True, ack=False)
    ms_phase_center = ms_field.getcell("PHASE_DIR", 0)
    ms_units = ms_field.getcoldesc("PHASE_DIR")["keywords"]["QuantumUnits"]
    obs_phase_center = SkyCoord(ms_phase_center, frame=FK5, unit=ms_units)

    directions = np.array([[d.ra.rad, d.dec.rad] for d in obs_directions]).squeeze(axis=2)  # type: ignore
    calibrations = leap_calibrate(
        ms_path,
        directions=directions,
        solution_interval=solution_interval,
        min_baseline_threshold=min_baseline_threshold,
        impl=impl,
        verbosity=verbosity
    )

    # coords
    antennas = list(range(len(ms_antenna)))
    times = []

    # direction independant variables
    intervals = []
    datetimes = []

    # direction dependant variables
    gains = [[] for direction in directions]

    # calibration results
    for calibration in calibrations:
        start_epoch, end_epoch = (calibration.start_epoch, calibration.end_epoch)
        interval = end_epoch - start_epoch

        times.append(end_epoch)
        intervals.append(interval)
        scale = ms.getcoldesc("TIME")['keywords']['MEASINFO']['Ref']
        epoch = ms.getcoldesc("TIME")['comment']
        if epoch == "Modified Julian Day" and scale == "UTC":
            datetimes.append((Time(0, format='mjd', scale='tai') + TimeDelta(end_epoch, format='sec')).datetime64)
        else:
            raise ValueError("Uknown epoch+timescale combination", epoch)

        for d_idx, beam_calibration in enumerate(calibration.beam_calibrations):
            gains[d_idx].append(np.expand_dims(np.exp(1j * beam_calibration.antenna_phases), (1, 2, 3)))

    models = []
    for d_idx, direction in enumerate(directions):
        models.append(xarray.Dataset(
            coords=dict(
                time=times,
                antenna=antennas,
                frequency=[0],
                receptor1=["I"],
                receptor2=["I"],
            ),
            data_vars=dict(
                gain=xarray.DataArray(gains[d_idx], dims=["time", "antenna", "frequency", "receptor1", "receptor2"]),
                # weight=xarray.DataArray(0, dims=["time", "antenna", "frequency", "receptor1", "receptor2"]),
                # residual=xarray.DataArray(residual, dims=["time", "frequency", "receptor1", "receptor2"]),
                interval=xarray.DataArray(intervals, dims=["time"]),
                datetime=xarray.DataArray(datetimes, dims=["time"]),
            ),
            attrs=dict(
                data_model="GainTable",
                # phasecentre=obs_phase_center,
                phasecentre=obs_directions[d_idx]
            )
        ))
    return models
