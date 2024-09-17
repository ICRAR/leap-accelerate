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

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace icrar
{
namespace python
{
    /**
     * @brief An boost::python adapter/wrapper for MeasurementSets. Method overloading
     * and types without python bindings are not allowed in signatures.
     */
    class PyMeasurementSet
    {
        std::shared_ptr<MeasurementSet> m_measurementSet;

    public:
        PyMeasurementSet(std::string msPath);
        const MeasurementSet& Get() const { return *m_measurementSet; }

        Eigen::Tensor<double, 3> ReadCoords(std::uint32_t startTimestep, std::uint32_t intervalTimesteps);
        Eigen::Tensor<std::complex<double>, 4> ReadVis(std::uint32_t startTimestep, std::uint32_t intervalTimesteps);
    };
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED
