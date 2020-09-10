/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define EIGEN_HAS_CXX11 1
#define EIGEN_VECTORIZE_GPU 1
#define EIGEN_CUDACC 1
#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Arrays/Matrix.h>

#include <queue>

namespace casacore
{
    class MDirection;
    class MVDirection;
    class MVuvw;
}

namespace icrar
{
    class Integration;
    class IntegrationResult;
    class CalibrationResult;
}

namespace icrar
{
namespace cuda
{
    class MetaData;
    class DeviceMetaData;

    std::queue<IntegrationResult> PhaseRotate(
        DeviceMetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations);

    void RotateVisibilities(
        Integration& integration,
        DeviceMetaData& metadata);

    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
         const Eigen::VectorXi& a1,
         const Eigen::VectorXi& a2,
         int refAnt,
         bool map);
}
}