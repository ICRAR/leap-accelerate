
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

#include <icrar/leap-accelerate/cuda/MetaDataCuda.h>

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

#include <complex>
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
        MetaDataCudaHost& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        throw std::runtime_error("not implemented"); //TODO
    }

    void RotateVisibilities(
        Integration& integration,
        MetaDataCudaHost& metadata,
        const MVDirection& direction)
    {
        //using namespace std::literals::complex_literals;
        auto& data = integration.data;
        auto& uvw = integration.uvw;
        auto parameters = integration.parameters;

        if(metadata.init)
        {
            //metadata['nbaseline']=metadata['stations']*(metadata['stations']-1)/2
            
            metadata.SetDD(direction);
            metadata.SetWv();
            // Zero a vector for averaging in time and freq
            metadata.avg_data = Eigen::MatrixXcd(integration.baselines, metadata.GetConstants().num_pols);
            metadata.init = false;
        }
        metadata.CalcUVW(uvw);

        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
            shiftFactor = shiftFactor + 2 * pi * (metadata.GetConstants().phase_centre_ra_rad * metadata.oldUVW[baseline].get()[0]);
            shiftFactor = shiftFactor -2 * pi * (direction.get()[0] * uvw[baseline].get()[0] - direction.get()[1] * uvw[baseline].get()[1]);

            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.GetConstants().channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.GetConstants().channel_wavelength[channel];
                std::complex<double> v = data(channel, baseline);

                data(channel, baseline) = v * std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));
                if(data(channel, baseline).real() == NAN
                || data(channel, baseline).imag() == NAN)
                {
                    metadata.avg_data(1, baseline) += data(channel,baseline);
                }
            }
        }
    }

    std::pair<casacore::Matrix<double>, casacore::Vector<std::int32_t>> PhaseMatrixFunction(
        const casacore::Vector<std::int32_t>& a1,
        const casacore::Vector<std::int32_t>& a2,
        int refAnt,
        bool map)
         {
             throw std::runtime_error("cuda::PhaseMatrixFunction not implemented");
         }
}
}
