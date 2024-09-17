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

#if PYTHON_ENABLED

#include "PyLeapCalibrator.h"
#include "PyMeasurementSet.h"
#include <icrar/leap-accelerate/core/log/logging.h>

#include <future>

namespace py = pybind11;

namespace
{
    template<typename T>
    inline T ternary(bool condition, T trueValue, T falseValue)
    {
        return condition ? trueValue : falseValue;
    }

    const std::map<int, icrar::log::Verbosity> python_to_log_level = {
        {50, icrar::log::Verbosity::fatal},
        {40, icrar::log::Verbosity::error},
        {30, icrar::log::Verbosity::warn},
        {20, icrar::log::Verbosity::info},
        {10, icrar::log::Verbosity::debug},
    };

    template<typename T>
    inline boost::optional<T> ToOptional(const py::object& obj)
    {
        boost::optional<T> output;
        if(!obj.is_none())
        {
            output = obj.cast<T>();
        }
        return output;
    }

    icrar::Slice ToSlice(const py::slice& obj)
    {
        return icrar::Slice(
            ToOptional<int64_t>(obj.attr("start")),
            ToOptional<int64_t>(obj.attr("stop")),
            ToOptional<int64_t>(obj.attr("step"))
        );
    }
}

namespace icrar
{
namespace python
{
    std::vector<SphericalDirection> ToSphericalDirectionVector(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions)
    {
        auto output = std::vector<SphericalDirection>();
        for(int64_t row = 0; row < directions.rows(); ++row)
        {
            output.push_back(directions(row, Eigen::placeholders::all));
        }
        return output;
    }

    PyLeapCalibrator::PyLeapCalibrator(ComputeImplementation impl, icrar::log::Verbosity verbosity)
    {
        icrar::log::Initialize(verbosity);
        m_calibrator = LeapCalibratorFactory::Create(impl);
    }

    PyLeapCalibrator::PyLeapCalibrator(std::string impl, int verbosity)
    {
        icrar::log::Initialize(python_to_log_level.at(verbosity));
        m_calibrator = LeapCalibratorFactory::Create(ParseComputeImplementation(impl));
    }

    PyLeapCalibrator::PyLeapCalibrator(const PyLeapCalibrator& other)
    {
        m_calibrator = other.m_calibrator;
    }

    void PyLeapCalibrator::Calibrate(
        const MeasurementSet& measurementSet,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const Slice& solutionInterval,
        const double minimumBaselineThreshold,
        const std::function<void(const cpu::Calibration&)>& callback)
    {
        // No python types used past this block
        py::gil_scoped_release nogil;

        auto validatedDirections = ToSphericalDirectionVector(directions);
        int referenceAntenna = 0;
        ComputeOptionsDTO computeOptions = {boost::none, boost::none, boost::none};

        m_calibrator->Calibrate(
            callback,
            measurementSet,
            validatedDirections,
            solutionInterval,
            minimumBaselineThreshold,
            true,
            referenceAntenna,
            computeOptions
        );
    }

    void PyLeapCalibrator::PythonCalibrate(
        const std::string& msPath,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const pybind11::slice& solutionInterval,
        const double minimumBaselineThreshold,
        const std::function<void(const cpu::Calibration&)> callback)
    {
        auto measurementSet = std::make_unique<MeasurementSet>(msPath);
        Calibrate(
            *measurementSet,
            directions,
            ToSlice(solutionInterval),
            minimumBaselineThreshold,
            callback
        );
    }

    void PyLeapCalibrator::CalibrateToFile(
        const MeasurementSet& measurementSet,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const Slice& solutionInterval,
        const double minimumBaselineThreshold,
        boost::optional<std::string> outputPath)
    {
        py::gil_scoped_release nogil{};

        auto validatedDirections = ToSphericalDirectionVector(directions);
        int referenceAntenna = 0;
        ComputeOptionsDTO computeOptions = {boost::none, boost::none, boost::none};

        std::vector<cpu::Calibration> calibrations;
        m_calibrator->Calibrate(
            [&](const cpu::Calibration& cal) {
                calibrations.push_back(cal);
            },
            measurementSet,
            validatedDirections,
            solutionInterval,
            minimumBaselineThreshold,
            true,
            referenceAntenna,
            computeOptions
        );

        auto calibrationCollection = cpu::CalibrationCollection(std::move(calibrations));
        if(outputPath.has_value())
        {
            std::ofstream file(outputPath.value());
            calibrationCollection.Serialize(file);
        }
        else
        {
            calibrationCollection.Serialize(std::cout);
        }
    }

    void PyLeapCalibrator::PythonCalibrateToFile(
        const std::string& msPath,
        const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& directions,
        const py::slice& solutionInterval,
        const double minimumBaselineThreshold,
        const py::object& outputPath)
    {
        auto measurementSet = std::make_unique<MeasurementSet>(msPath);
        CalibrateToFile(
            *measurementSet,
            directions,
            ToSlice(solutionInterval),
            minimumBaselineThreshold,
            ToOptional<std::string>(outputPath)
        );
    }
} // namespace python
} // namespace icrar

#endif // PYTHON_ENABLED