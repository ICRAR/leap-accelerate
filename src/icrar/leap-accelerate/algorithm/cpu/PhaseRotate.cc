
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
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/utils.h>
#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <icrar/leap-accelerate/math/cpu/matrix.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>

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
namespace cpu
{
    void RemoteCalibration(MetaData& metadata, const std::vector<casacore::MVDirection>& directions)
    {
        auto input_queues = std::vector<std::queue<Integration>>(directions.size());
        auto output_integrations = std::vector<std::queue<IntegrationResult>>(directions.size());
        auto output_calibrations = std::vector<std::queue<CalibrationResult>>(directions.size());

        for(int i = 0; i < directions.size(); ++i)
        {
            icrar::cpu::PhaseRotate(metadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }
    }

    void PhaseRotate(
        MetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        auto cal = std::vector<casacore::Array<double>>();

        while(true)
        {
            boost::optional<Integration> integration = !input.empty() ? input.front() : (boost::optional<Integration>)boost::none;
            input.pop();

            if(integration.is_initialized())
            {
                icrar::cpu::RotateVisibilities(integration.get(), metadata, direction);
                output_integrations.push(IntegrationResult(direction, integration.get().integration_number, boost::none));
            }
            else
            {
                std::function<Radians(std::complex<double>)> getAngle = [](std::complex<double> c) -> Radians
                {
                    return std::arg(c);
                };
                casacore::Matrix<Radians> avg_data = MapCollection(metadata.avg_data, getAngle);
                casacore::Array<double> cal1 = icrar::cpu::multiply(metadata.Ad1, avg_data.column(0));// TODO: (IPosition(0, metadata.I1)); //diagonal???
                casacore::Matrix<double> dInt = avg_data(Slice(0, 0), Slice(metadata.I.shape()[0], metadata.I.shape()[1]));
                
                for(int n = 0; n < metadata.I.size(); ++n)
                {
                    dInt[n] = avg_data(IPosition(metadata.I)) - metadata.A(IPosition(1, n)) * cal1;
                }
                cal.push_back(icrar::cpu::multiply(metadata.Ad, dInt) + cal1);
                break;
            }
        }

        output_calibrations.push(CalibrationResult(direction, cal));
    }

    void RotateVisibilities(Integration& integration, MetaData& metadata, const casacore::MVDirection& direction)
    {
        using namespace std::literals::complex_literals;
        auto& data = integration.data;
        auto& uvw = integration.uvw;
        auto parameters = integration.parameters;

        if(metadata.init)
        {
            //metadata['nbaseline']=metadata['stations']*(metadata['stations']-1)/2
            
            SetDD(metadata, direction);
            SetWv(metadata);
            // Zero a vector for averaging in time and freq
            metadata.avg_data = casacore::Matrix<DComplex>(integration.baselines, metadata.num_pols);
            metadata.init = false;
        }
        CalcUVW(uvw, metadata);

        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
            shiftFactor = shiftFactor + 2 * pi * (metadata.phase_centre_ra_rad * metadata.oldUVW[baseline].get()[0]);
            shiftFactor = shiftFactor -2 * pi * (direction.get()[0] * uvw[baseline].get()[0] - direction.get()[1] * uvw[baseline].get()[1]);

            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.channel_wavelength[channel];
                double rs = sin(shiftRad);
                double rc = cos(shiftRad);
                std::complex<double> v = data[channel][baseline];

                data[channel][baseline] = v * std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));
                if(data[channel][baseline].real() == NAN
                || data[channel][baseline].imag() == NAN)
                {
                    metadata.avg_data(casacore::IPosition(1, baseline)) += data[channel][baseline];
                }
            }
        }
    }

    std::pair<casacore::Matrix<double>, casacore::Vector<std::int32_t>> PhaseMatrixFunction(
        const Vector<std::int32_t>& a1,
        const Vector<std::int32_t>& a2,
        int refAnt,
        bool map)
    {
        if(a1.size() != a2.size())
        {
            throw std::invalid_argument("a1 and a2 must be equal size");
        }

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

        Vector<int> I = Vector<int>(a1.size() + 1);
        for(auto v : I)
        {
            v = 1;
        }

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(IPosition(1, n)) != a2(IPosition(1, n)))
            {
                if((refAnt < 0) || ((refAnt >= 0) && ((a1(IPosition(1, n))==refAnt) || (a2(IPosition(1, n)) == refAnt))))
                {
                    A(IPosition(2, k, a1(IPosition(1, n)))) = 1;
                    A(IPosition(2, k, a2(IPosition(1, n)))) = -1;
                    I(IPosition(1, k)) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
            A(IPosition(2, k, refAnt)) = 1;
            k++;
            
            //A = A(Slice(0, k), Slice(0, 127)); TODO
            //I = I(Slice(0, k)); TODO
        }

        return std::make_pair(A, I);
    }
}
}
