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

#include <eigen3/Eigen/Core>



#include <queue>

namespace casacore
{
    class MeasurementSet;
    class MDirection;
    class MVDirection;
    class MVuvw;
    template<typename T>
    class Array;
    template<typename T>
    class Matrix;
    template<typename T>
    class Vector;
}

namespace icrar
{
    class Integration;
    class IntegrationResult;
    class MetaData;
}

namespace icrar
{
namespace cuda
{ 
    std::queue<IntegrationResult> PhaseRotate(MetaData& metadata, const std::vector<casacore::MVDirection>& directions, std::queue<Integration>& input);

    void RotateVisibilities(Integration& integration, MetaData& metadata, const casacore::MVDirection& direction);

    std::pair<casacore::Matrix<double>, casacore::Vector<std::int32_t>> PhaseMatrixFunction(
        const casacore::Vector<std::int32_t>& a1,
        const casacore::Vector<std::int32_t>& a2,
        int refAnt,
        bool map);
}
}