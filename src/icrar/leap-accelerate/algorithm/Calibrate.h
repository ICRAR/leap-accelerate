/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    class MeasurementSet;

        /**
         * @brief Performs Leap calibration using a specialized implementation.
         * 
         * @param impl selects the calibration implementation
         * @param ms the mesurement set containing all input measurements
         * @param directions the directions to calibrate for
         * @param minimumBaselineThreshold the minimum baseline length to use in calibrations
         * @param isFileSystemCacheEnabled enable to use the filesystem to cache data between calibration calls
         * @return CalibrateResult the calibrationn result
         */
    cpu::CalibrateResult Calibrate(
        ComputeImplementation impl,
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions,
        double minimumBaselineThreshold,
        bool isFileSystemCacheEnabled);
}