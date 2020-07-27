
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

#include "PhaseRotate.h"

#include <icrar/leap-accelerate/math/math.h>

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <icrar/leap-accelerate/math/cuda/matrix.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>

using Radians = double;

using namespace casacore;

namespace icrar
{
namespace cuda
{ 
    std::queue<IntegrationResult> PhaseRotate(MetaData& metadata, const std::vector<MVDirection>& directions, std::queue<Integration>& input)
    {
        throw std::runtime_error("not implemented"); //TODO
    }

    void RotateVisibilities(Integration& integration, MetaData& metadata, const MVDirection& direction)
    {
        throw std::runtime_error("not implemented"); //TODO
    }

    std::pair<Matrix<double>, Vector<std::int32_t>> PhaseMatrixFunction(
        const Vector<std::int32_t>& a1,
        const Vector<std::int32_t>& a2,
        int refAnt, bool map)
    {
        throw std::runtime_error("not implemented"); //TODO
    }
}
}
