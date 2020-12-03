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

#include <casacore/ms/MeasurementSets.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/CalibrateResult.h>

#include <Eigen/Core>

#include <boost/optional.hpp>

#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace icrar
{
    class MeasurementSet;

    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class CalibrationResult;
    } // namespace cpu
} // namespace icrar

namespace icrar
{
namespace cpu
{
    class MetaData;

    /**
     * @copydoc ILeapEngine::ILeapCalibrator
     * Calibrates by performing phase rotation for each direction in @p directions
     * by splitting uvws into integration batches.
     */
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<MVDirection>& directions,
        double minimumBaselineThreshold,
        bool isFileSystemCacheEnabled);

    /**
     * @brief Performs rotation, summing and calibration for @p direction
     * 
     * @param metadata metadata object containing data required for calibration
     * @param direction the direction to calibrate for 
     * @param input batches of uvws and visibilities to process
     * @param output_integrations output from summing a function of uvws and visibilities
     * @param output_calibrations output calibration from summing a function of uvws and visibilities
     */
    void PhaseRotate(
        MetaData& metadata,
        const MVDirection& direction,
        std::vector<Integration>& input,
        std::vector<IntegrationResult>& output_integrations,
        std::vector<CalibrationResult>& output_calibrations);

    /**
     * @brief Performs averaging over each baseline, channel and polarization.
     * 
     * @param integration The input integration batch of uvws and visibilities
     * @param metadata The metadata object where AverageData is written to
     */
    void RotateVisibilities(
        Integration& integration,
        MetaData& metadata);
} // namespace cpu
} // namespace icrar