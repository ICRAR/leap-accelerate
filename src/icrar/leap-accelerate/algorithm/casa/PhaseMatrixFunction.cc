
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

#include "PhaseMatrixFunction.h"

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/casa/matrix.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <set>

using namespace casacore;

namespace icrar
{
namespace casalib
{
    std::pair<casacore::Matrix<double>, casacore::Vector<std::int32_t>> PhaseMatrixFunction(
        const casacore::Vector<std::int32_t>& a1,
        const casacore::Vector<std::int32_t>& a2,
        const casacore::Vector<bool>& fg,
        int refAnt)
    {
        if(a1.size() != a2.size() || a1.size() != fg.size())
        {
            throw std::invalid_argument("a1, a2 and fg must be equal size");
        }

        auto unique = std::set<std::int32_t>(a1.cbegin(), a1.cend());
        unique.insert(a2.cbegin(), a2.cend());
        int nAnt = unique.size();

        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        // Cross pairs and reference antenna entries
        // Thus A will be maximum antenna _number_ rather than maximum number of antennas. If, for example Ante `1' is missing the column 1 will be all zeros
        Matrix<double> A = Matrix<double>(a1.size() + 1, std::max(icrar::ArrayMax(a1), icrar::ArrayMax(a2)) + 1);
        A = 0.0;

        Vector<int> I = Vector<int>(a1.size()); // I will be 1 less row than A.
        I = -1;


        int STATIONS = A.shape()[1];
        int k = 0;

        for(size_t n = 0; n < a1.size(); n++)
        {
            if(a1(n) != a2(n))
            {
                // skip entry if data not flagged
                if(!fg(n) && ((refAnt < 0) || ((refAnt >= 0) && ((a1(n) == refAnt) || (a2(n) == refAnt)))))
                {
                    A(k, a1(n)) = 1.0; // set scalear
                    A(k, a2(n)) = -1.0; // set scalear
                    I(k) = n; //set scalear
                    k++;
                }  // Otherwise the baseline entry (and therefore weight) is zero
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
        }

        A(k, refAnt) = 1;
        k++;
        
        auto Atemp = casacore::Matrix<double>(k-1, STATIONS);
        Atemp = A(Slice(0, k-1), Slice(0, STATIONS));
        A.resize(0,0);
        A = Atemp;


        auto Itemp = casacore::Vector<int>(k-1);
        Itemp = I(Slice(0, k-1));
        I.resize(0);
        I = Itemp;

        return std::make_pair(A, I);
    }
}
}