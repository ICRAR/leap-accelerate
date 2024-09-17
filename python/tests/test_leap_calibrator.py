#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import tempfile
import asyncio
import threading

import numpy as np
import leap
import json
import os
from pathlib import Path

MWA_MS = str(Path(
    os.path.dirname(__file__),
    "../../testdata/mwa/1197638568-split.ms"
))

def test_calibrate_minimal():
    cal = leap.LeapCalibrator("cpu")
    output = list()
    cal.calibrate(
        ms_path=MWA_MS,
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        callback=output.append
    )
    assert len(output) == 14


def test_calibrate_to_file():
    cal = leap.LeapCalibrator(leap.ComputeImplementation.cpu)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix='output_', suffix=".json"
    ) as out_file:
        cal.calibrate_to_file(
            ms_path=MWA_MS,
            directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
            solution_interval=slice(0,None,1),
            output_path=out_file.name
        )
    with open(out_file.name, "r") as f:
        output = json.load(f)
    assert len(output) == 14

def test_calibrate_callback():
    cal = leap.LeapCalibrator("cpu")
    
    output = list()
    cal.calibrate(
        ms_path=MWA_MS,
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        callback=output.append)

    assert len(output) == 14


def test_calibrate_callback2():
    cal = leap.LeapCalibrator("cpu")
    output = list()

    # python threading never executes in parallel
    async def calibrate_async(calibrator, callback):
        cal.calibrate(
            ms_path=MWA_MS,
            directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
            solution_interval=slice(0,None,1),
            callback=callback)

    t1 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, output.append), ))
    t2 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, output.append), ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(output) == 28


@pytest.mark.skip
@pytest.mark.asyncio
async def test_calibrate_async():
    cal = leap.LeapCalibrator("cpu")
    
    output = list()
    await cal.calibrate_async(
        ms_path=MWA_MS,
        directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
        solution_interval=slice(0,None,1),
        callback=output.append
    )
    assert len(output) == 14

@pytest.mark.skip
def test_calibrate_async_thread():
    cal = leap.LeapCalibrator("cpu")
    
    output = list()
    async def calibrate_async(calibrator, callback):
        await calibrator.calibrate_async(
            ms_path=MWA_MS,
            directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
            solution_interval=slice(0,None,1),
            callback=callback
        )

    t1 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, output.append), ))
    t1.start()
    assert len(output) == 0
    t1.join()
    assert len(output) == 14

#@pytest.mark.skip
def test_calibrate_async2():
    # NOTE: cuda calibrator currently not thread safe
    cal = leap.LeapCalibrator("cpu")

    # NOTE: MeasurementSet reading is not thread safe from same file
    ms1 = leap.MeasurementSet(MWA_MS)
    ms2 = leap.MeasurementSet(MWA_MS)
    ms3 = leap.MeasurementSet(MWA_MS)

    output1 = list()
    output2 = list()
    output3 = list()
    async def calibrate_async(calibrator, ms, callback):
        await calibrator.calibrate_async(
            ms=ms,
            directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5, 0.6]]),
            solution_interval=slice(0,None,1),
            callback=callback
        )

    t1 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, ms1, output1.append), ))
    t2 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, ms2, output2.append), ))
    t3 = threading.Thread(target=asyncio.run, args=(calibrate_async(cal, ms3, output3.append), ))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    assert len(output1) == 14
    assert len(output2) == 14
    assert len(output3) == 14
    
    # list.append is not threadsafe for callback on multiple processes
    # could use a lambda with mutex on a single list and entries should be interleaved
    combined = output1 + output2 + output3
    assert len(combined) == 42


@pytest.mark.skip
def test_plasma_calibration():
    cal = leap.LeapCalibrator("cpu")
    # cal.plasma_calibrate()

# def test_calibrate_config():
#     cal = leap.LeapCalibrator("cpu")
#     cal.calibrate(config="../../testdata/mwa.json")