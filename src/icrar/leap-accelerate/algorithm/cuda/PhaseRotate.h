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

#ifdef __CUDACC_VER__
#undef __CUDACC_VER__
#define __CUDACC_VER__ ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif
#define EIGEN_HAS_CXX11 1
#define EIGEN_VECTORIZE_GPU 1
#define EIGEN_CUDACC 1
#include <eigen3/Eigen/Core>

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
    class MetaDataCudaHost;
    class MetaDataCudaDevice;

    std::queue<IntegrationResult> PhaseRotate(
        MetaDataCudaDevice& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations);

    void RotateVisibilities(
        Integration& integration,
        MetaDataCudaDevice& metadata,
        const casacore::MVDirection& direction);

    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
         const Eigen::VectorXi& a1,
         const Eigen::VectorXi& a2,
         int refAnt,
         bool map);
}
}