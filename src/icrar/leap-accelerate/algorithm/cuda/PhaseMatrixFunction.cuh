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

#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <eigen3/Eigen/Core>

#include <queue>
#include <vector>
#include <set>

namespace icrar
{
namespace cuda
{
    template<typename T>
    __global__ void g_phase_matrix_function(const int m, const int n, const int k, const T *a, const T* b, T* c)
    {
        throw std::runtime_error("not implemented");
    }

    std::pair<casacore::Matrix<double>, casacore::Array<std::int32_t>> PhaseMatrixFunction(
        const casacore::Array<std::int32_t>& a1,
        const casacore::Array<std::int32_t>& a2,
        int refAnt, bool map)
    {
        using namespace casacore;

        auto unique = std::set<std::int32_t>(a1.cbegin(), a1.cend());
        unique.insert(a2.cbegin(), a2.cend());
        int nAnt = unique.size();
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        casacore::Matrix<double> A = casacore::Matrix<double>(a1.size() + 1, icrar::ArrayMax(a1));
        for(auto v : A)
        {
            v = 0;
        }

        casacore::Matrix<int> I = casacore::Matrix<int>(a1.size() + 1, a1.size() + 1);
        for(auto v : I)
        {
            v = 1;
        }

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(IPosition(n)) != a2(IPosition(n)))
            {
                if((refAnt < 0) | ((refAnt >= 0) & ((a1(IPosition(n))==refAnt) | (a2(IPosition(n)) == refAnt))))
                {
                    A(IPosition(k, a1(IPosition(n)))) = 1;
                    A(IPosition(k, a2(IPosition(n)))) = -1;
                    I(IPosition(k)) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
            A(IPosition(k,refAnt)) = 1;
            k++;
            
            A = A(Slice(0), Slice(k));
            I = I(Slice(0), Slice(k));
        }

        return std::make_pair(A, I);
    }
}
}