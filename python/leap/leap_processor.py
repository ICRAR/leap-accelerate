
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia
# Copyright by UWA(in the framework of the ICRAR)
# All rights reserved

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111 - 1307  USA

import collections
import argparse
import asyncio
import collections
import concurrent.futures
import logging
import queue
import signal
import subprocess as sp
import threading
from typing import Deque, Optional

import numpy as np
import ska_ser_logging
import ska_sdp_dal as rpc
from sdp_dal_schemas import PROCEDURES
from cbf_sdp import icd
from cbf_sdp.plasma_tm import PlasmaTM
from cbf_sdp.plasma_processor import PlasmaProcessor, PlasmaPayload

import leap

class PlasmaToLeapRunner(object):
    """
    A Service that runs leap calibration on incoming plasma observations.
    """

    def __init__(self, output_file: str, plasma_socket: str):
        self.processor = PlasmaProcessor(plasma_socket)
        self.calibrator = leap.LeapCalibrator("cpu")
        self.active = True

    async def run(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        self.active = True
        while self.active:
            await loop.run_in_executor(executor, self.processor.process, self.process_timeout)
            payload: PlasmaTM = self.processor.pop_payload()
            if payload:
                self.calibrator.calibrate()


