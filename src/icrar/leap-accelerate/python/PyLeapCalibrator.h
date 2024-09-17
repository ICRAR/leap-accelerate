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

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <Eigen/Core>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <future>

namespace icrar
{
namespace log
{
    enum class Verbosity;
}
namespace python
{
    class PyMeasurementSet;

    /**
     * @brief An adapter to a boost::python compatible class
     * @note the Calibrate signature may change as needed
     */
    class PyLeapCalibrator
    {
        std::shared_ptr<ILeapCalibrator> m_calibrator;

    public:
        PyLeapCalibrator(ComputeImplementation impl, icrar::log::Verbosity verbosity = icrar::log::Verbosity::warn);
        PyLeapCalibrator(std::string impl, int verbosity = 40);
        PyLeapCalibrator(const PyLeapCalibrator& other);

        void Calibrate(
            const MeasurementSet& msPath,
            const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
            const Slice& solutionInterval,
            const double minimumBaselineThreshold,
            const std::function<void(const cpu::Calibration&)>& callback);

        void CalibrateToFile(
            const MeasurementSet& msPath,
            const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
            const Slice& solutionInterval,
            const double minimumBaselineThreshold,
            boost::optional<std::string> outputPath);


        /**
         * @brief A python interop compatible signature for leap calibration
         */
        void PythonCalibrate(
            const std::string& msPath,
            const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
            const pybind11::slice& solutionInterval,
            const double minimumBaselineThreshold,
            const std::function<void(const cpu::Calibration&)> callback);


        /**
         * @brief A python interop compatible signature for leap calibration
         */
        void PythonCalibrateToFile(
            const std::string& msPath,
            const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
            const pybind11::slice& solutionInterval,
            const double minimumBaselineThreshold,
            const pybind11::object& outputPath);

        /**
         * @brief A python interop campatible signature for
         * calibrate
         */
        void PythonPlasmaCalibrate(
            const pybind11::object& plasmaTM,
            const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
            const pybind11::object& outputPath);
    };
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED
