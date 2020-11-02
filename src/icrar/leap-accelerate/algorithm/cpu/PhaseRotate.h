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

namespace casacore
{
    class MDirection;
    class MVDirection;
    class MVuvw;
}

namespace icrar
{
    class MeasurementSet;

    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class CalibrationResult;
    }
}

namespace icrar
{
namespace cpu
{
    class MetaData;

    /**
     * @brief Performs LEAP calibration for stations in @c ms for
     * each direction in @c directions
     * 
     * @param ms the measurement set
     * @param directions the directions to calibrate for
     * @return CalibrateResult 
     */
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<MVDirection>& directions);

    /**
     * @brief Calculates calibrations required to change the phase centre of an observation
     * by rotating visibilities and performing phase detection.
     * 
     * @param metadata 
     * @param direction 
     * @param input 
     * @param output_integrations 
     * @param output_calibrations 
     */
    void PhaseRotate(
        MetaData& metadata,
        const MVDirection& direction,
        std::vector<Integration>& input,
        std::vector<IntegrationResult>& output_integrations,
        std::vector<CalibrationResult>& output_calibrations);

    /**
     * @brief Performs averaging of integration and metadata to populate @c metadata.avg_data
     * 
     * @param integration 
     * @param metadata 
     */
    void RotateVisibilities(
        Integration& integration,
        MetaData& metadata);
}
}