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
// #include "PyLeapCalibration.h"
#include "async.h"
#include "pybind_eigen.h"

#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>
#include <icrar/leap-accelerate/core/log/logging.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>

#include <future>
#include <string>

namespace py = pybind11;

// Expose all vector methods is python
// PYBIND11_MAKE_OPAQUE(std::vector<int>);

PYBIND11_MODULE(LeapAccelerate, m)
{
    m.doc() = "Linear Execision of the Atmosphere in Parallel";

    m.def("enable_log", []() {
        icrar::log::Initialize(icrar::log::Verbosity::warn);
    });

    // See https://stackoverflow.com/questions/39995149/expand-a-type-n-times-in-template-parameter
    // for automatically generating parameter packs (requires a wrapper type)
    PybindEigenTensor<double, 3, int, int, int>(m, "Tensor3d");
    PybindEigenTensor<double, 4, int, int, int, int>(m, "Tensor4d");
    PybindEigenTensor<std::complex<double>, 3, int, int, int>(m, "Tensor3cd");
    PybindEigenTensor<std::complex<double>, 4, int, int, int, int>(m, "Tensor4cd");
    py::bind_vector<std::vector<icrar::cpu::BeamCalibration>>(m, "BeamCalibrationVector");

    py::class_<icrar::cpu::BeamCalibration>(m, "BeamCalibration")
        .def(py::init<Eigen::Vector2d, Eigen::MatrixXd>())
        .def_property_readonly("direction", &icrar::cpu::BeamCalibration::GetDirection,
            "Beam ra and dec in radians"
        )
        .def_property_readonly("antenna_phases", &icrar::cpu::BeamCalibration::GetAntennaPhases,
            "Calibrated phases of input antennas"
        );

    std::vector<icrar::cpu::BeamCalibration>& (icrar::cpu::Calibration::*gbcp)()
        = &icrar::cpu::Calibration::GetBeamCalibrations;
    py::class_<icrar::cpu::Calibration>(m, "Calibration")
        .def(py::init<int, int>())
        .def_property_readonly("start_epoch", &icrar::cpu::Calibration::GetStartEpoch)
        .def_property_readonly("end_epoch", &icrar::cpu::Calibration::GetEndEpoch)
        .def_property_readonly("beam_calibrations", py::cpp_function(
           gbcp,
           py::return_value_policy::reference_internal
        ));

    py::enum_<icrar::ComputeImplementation>(m, "ComputeImplementation")
        .value("cpu", icrar::ComputeImplementation::cpu)
        .value("cuda", icrar::ComputeImplementation::cuda)
        .export_values();

    py::enum_<icrar::log::Verbosity>(m, "Verbosity")
        .value("fatal", icrar::log::Verbosity::fatal)
        .value("error", icrar::log::Verbosity::error)
        .value("warn", icrar::log::Verbosity::warn)
        .value("info", icrar::log::Verbosity::info)
        .value("debug", icrar::log::Verbosity::debug)
        .value("trace", icrar::log::Verbosity::trace)
        .export_values();

    // def_async extension only available on class_async, need to cast def() return type
    // to move elsewhere or integrate into pybind11::class_
    py::async::enable_async(m);
    py::async::class_async<icrar::python::PyLeapCalibrator>(m, "LeapCalibrator")
        .def(py::init<icrar::ComputeImplementation>())
        .def(py::init<std::string>())
        .def(py::init<icrar::ComputeImplementation, icrar::log::Verbosity>())
        .def(py::init<std::string, int>())
        .def("calibrate", &icrar::python::PyLeapCalibrator::PythonCalibrate,
            py::arg("ms_path"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("min_baseline_threshold")=0.0,
            py::arg("callback")
        )
        .def("calibrate_to_file", &icrar::python::PyLeapCalibrator::PythonCalibrateToFile,
            py::arg("ms_path"),
            py::arg("directions").noconvert(),
            py::arg("solution_interval")=py::slice(0,1,1),
            py::arg("min_baseline_threshold")=0.0,
            py::arg("output_path")
        );

    py::class_<icrar::python::PyMeasurementSet>(m, "MeasurementSet")
        .def(py::init<std::string>())
        .def("read_coords", &icrar::python::PyMeasurementSet::ReadCoords,
            py::arg("start_timestep"),
            py::arg("num_timesteps")
        )
        .def("read_vis", &icrar::python::PyMeasurementSet::ReadVis,
            py::arg("start_timestep"),
            py::arg("num_timesteps")
            //py::arg("polarizationSlice")=py::slice(0,-1,1)
        );
}

#endif // PYTHON_ENABLED