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

#if PYTHON_ENABLED

#include <icrar/leap-accelerate/model/cpu/calibration.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace icrar
{
namespace python
{
    class PyBeamCalibration
    {
        std::shared_ptr<BeamCalibration> m_beamcalibration;
    public:
        PyBeamCalibration();

        Eigen::Vector2d GetDirection();
        Eigen::VectorXi GetAntennaPhases();
    };

    /**
     * @brief An boost::python adapter/wrapper for Calibrations. Method overloading
     * and types without python bindings are not allowed in signatures.
     */
    class PyCalibration
    {
        std::shared_ptr<Calibration> m_calibration;

    public:
        PyCalibration();

        double GetStartEpoch();
        double GetEndEpoch();
        std::vector<PyBeamCalibration> GetBeamCalibrations();
    };
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED
