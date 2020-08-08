
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

#include <icrar/leap-accelerate/utils.h>
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <icrar/leap-accelerate/model/cuda/MetaDataCuda.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

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
    void RemoteCalibration(cuda::MetaData& metadata, const Eigen::Matrix<casacore::MVDirection, Eigen::Dynamic, 1>& directions)
    {

    }

    void PhaseRotate(
        cuda::MetaData& metadata,
        const casacore::MVDirection& directions,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        
    }

    void RotateVisibilities(Integration& integration, cuda::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        auto& data = integration.data;
        auto& uvw = metadata.UVW;
        auto parameters = integration.parameters;

        assert(uvw.size() == integration.baselines);
        assert(data.rows() == metadata.m_constants.channels);
        assert(data.cols() == integration.baselines);
        assert(metadata.oldUVW.size() == integration.baselines);
        assert(metadata.m_constants.channel_wavelength.size() == metadata.m_constants.channels);
        assert(metadata.avg_data.rows() == integration.baselines);
        assert(metadata.avg_data.cols() == metadata.m_constants.num_pols);
        
        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
            shiftFactor = shiftFactor + 2 * pi * (metadata.m_constants.phase_centre_ra_rad * metadata.oldUVW[baseline].get()[0]);
            shiftFactor = shiftFactor - 2 * pi * (metadata.direction.get()[0] * uvw[baseline].get()[0] - metadata.direction.get()[1] * uvw[baseline].get()[1]);

            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.m_constants.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.m_constants.channel_wavelength[channel];

                data(channel, baseline) *= std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));

                if(!data(channel, baseline).hasNaN())
                {
                    for(int i = 0; i < data(channel, baseline).cols(); i++)
                    {
                        metadata.avg_data(baseline, i) += data(channel, baseline)(i);
                    }
                }
            }
        }
    }

std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        int refAnt,
        bool map)
    {
        if(a1.size() != a2.size())
        {
            throw std::invalid_argument("a1 and a2 must be equal size");
        }

        auto unique = std::set<std::int32_t>(a1.begin(), a1.end());
        unique.insert(a2.begin(), a2.end());
        int nAnt = unique.size();
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a1.size() + 1, a1.maxCoeff() + 1);

        int STATIONS = A.cols() - 1; //TODO verify correctness

        Eigen::VectorXi I = Eigen::VectorXi(a1.size() + 1);
        I.setConstant(1);

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(n) != a2(n))
            {
                if((refAnt < 0) || ((refAnt >= 0) && ((a1(n) == refAnt) || (a2(n) == refAnt))))
                {
                    A(k, a1(n)) = 1;
                    A(k, a2(n)) = -1;
                    I(k) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
            A(k, refAnt) = 1;
            k++;
            
            auto Atemp = Eigen::MatrixXd(k, STATIONS);
            Atemp = A(Eigen::seq(0, k), Eigen::seq(0, STATIONS));
            A.resize(0,0);
            A = Atemp;

            auto Itemp = Eigen::VectorXi(k);
            Itemp = I(Eigen::seq(0, k));
            I.resize(0);
            I = Itemp;
        }

        return std::make_pair(A, I);
    }
}
}
