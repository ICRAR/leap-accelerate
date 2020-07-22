
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

#include "../../pch.h"
#include "PhaseRotate.h"

#include <icrar/leap-accelerate/MetaData.h>

#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math.h>
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
#include <set>

using Radians = double;

using namespace casacore;

namespace icrar
{
namespace cuda
{ 
    std::queue<IntegrationResult> PhaseRotate(
        MetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        throw std::runtime_error("not implemented"); //TODO

        // auto cal = std::vector<casacore::Array<double>>();

        // while(true)
        // {
        //     boost::optional<Integration> integration = !input.empty() ? input.front() : (boost::optional<Integration>)boost::none;
        //     input.pop();

        //     if(integration.is_initialized())
        //     {
        //         icrar::cpu::RotateVisibilities(integration.get(), metadata, direction);
        //         output_integrations.push(IntegrationResult(direction, integration.get().integration_number, boost::none));
        //     }
        //     else
        //     {
        //         std::function<Radians(std::complex<double>)> getAngle = [](std::complex<double> c) -> Radians
        //         {
        //             return std::arg(c);
        //         };
        //         casacore::Matrix<Radians> avg_data = MapCollection(metadata.avg_data, getAngle);
        //         casacore::Array<double> cal1 = icrar::cuda::multiply(metadata.Ad1, avg_data.column(0));
        //         casacore::Matrix<double> dInt = avg_data(Slice(0, 0), Slice(metadata.I.shape()[0], metadata.I.shape()[1]));
                
        //         for(int n = 0; n < metadata.I.size(); ++n)
        //         {
        //             dInt[n] = avg_data(IPosition(metadata.I)) - metadata.A(IPosition(n)) * cal1;
        //         }
        //         cal.push_back(icrar::cuda::multiply(metadata.Ad, dInt) + cal1);
        //         break;
        //     }
        // }

        // output_calibrations.push(CalibrationResult(direction, cal));
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
        auto unique = std::set<std::int32_t>(a1.cbegin(), a1.cend());
        unique.insert(a2.cbegin(), a2.cend());
        int nAnt = unique.size();
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Matrix<double> A = Matrix<double>(a1.size() + 1, icrar::ArrayMax(a1));
        for(auto v : A)
        {
            v = 0;
        }

        Matrix<int> I = Matrix<int>(a1.size() + 1, a1.size() + 1);
        for(auto v : I)
        {
            v = 1;
        }

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(IPosition(n)) != a2(IPosition(n)))
            {
                if((refAnt < 0) || ((refAnt >= 0) & ((a1(IPosition(n)) == refAnt) || (a2(IPosition(n)) == refAnt))))
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
