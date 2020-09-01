
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

#include <icrar/leap-accelerate/model/Integration.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
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
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<casacore::MVDirection>& directions,
        int solutionInterval)
    {
        auto metadata = icrar::casalib::MetaData(ms);

        auto output_integrations = std::make_unique<std::vector<std::queue<IntegrationResult>>>();
        auto output_calibrations = std::make_unique<std::vector<std::queue<CalibrationResult>>>();
        auto input_queues = std::vector<std::queue<Integration>>();
        
        for(int i = 0; i < directions.size(); ++i)
        {
            auto queue = std::queue<Integration>(); 
            queue.push(Integration(
                ms,
                i,
                metadata.channels,
                metadata.GetBaselines(),
                metadata.num_pols,
                metadata.GetBaselines()));

            input_queues.push_back(queue);
            output_integrations->push_back(std::queue<IntegrationResult>());
            output_calibrations->push_back(std::queue<CalibrationResult>());
        }

        for(int i = 0; i < directions.size(); ++i)
        {
            metadata.SetDD(directions[i]);
            metadata.SetWv();
            metadata.avg_data = casacore::Matrix<DComplex>(metadata.GetBaselines(), metadata.num_pols);

            auto metadatahost = icrar::cuda::MetaData(metadata); // TODO: use other constructor
            icrar::cpu::PhaseRotate(metadatahost, directions[i], input_queues[i], (*output_integrations)[i], (*output_calibrations)[i]);
        }

        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    void PhaseRotate(
        cuda::MetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        auto cal = std::vector<casacore::Matrix<double>>();
        while(true)
        {
            boost::optional<Integration> integration = !input.empty() ? input.front() : (boost::optional<Integration>)boost::none;
            if(integration.is_initialized())
            {
                input.pop();
            }

             if(integration.is_initialized())
             {
                icrar::cpu::RotateVisibilities(integration.get(), metadata);
                output_integrations.push(IntegrationResult(direction, integration.get().integration_number, boost::none));
            }
            else
            {
                std::function<Radians(std::complex<double>)> getAngle = [](std::complex<double> c) -> Radians
                {
                    return std::arg(c);
                };

                auto avg_data = metadata.avg_data.unaryExpr(getAngle);
                auto& indexes = metadata.I1;

                auto avg_data_t = avg_data(indexes, 0); // 1st pol only
                auto cal1 = metadata.Ad1 * avg_data_t;
                //assert(cal1.shape()[1] == 1);

//                 casacore::Matrix<double> dInt = casacore::Matrix<double>(metadata.I.size(), avg_data.shape()[1]);
//                 dInt = 0;

//                 for(int n = 0; n < metadata.I.size(); ++n)
//                 {
//                     Eigen::VectorXi e_i = ToVector(metadata.I);
//                     Eigen::MatrixXd e_avg_data_slice = ToMatrix(avg_data)(e_i, Eigen::all);
//                     casacore::Matrix<double> avg_data_slice = ConvertMatrix(e_avg_data_slice);

//                     casacore::Matrix<double> cumsum = metadata.A.data()[n] * cal1;
//                     dInt.row(n) = avg_data_slice.row(n) - casacore::sum(cumsum);
//                 }
                
//                 casacore::Matrix<double> dIntColumn = dInt.column(0); // 1st pol only
//                 dIntColumn = dIntColumn.reform(IPosition(2, dIntColumn.shape()[0], dIntColumn.shape()[1]));
//                 assert(dIntColumn.shape()[1] == 1);

//                 cal.push_back(icrar::casalib::multiply(metadata.Ad, dIntColumn) + cal1);
                 break;
             }
         }

         output_calibrations.push(CalibrationResult(direction, cal));
    }

    void RotateVisibilities(Integration& integration, cuda::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 3>& integration_data = integration.data;
        auto& uvw = integration.uvw;
        auto parameters = integration.parameters;

        //metadata.CalcUVW(uvw); TODO

        assert(metadata.GetConstants().nbaselines == integration.baselines);
        assert(uvw.size() == integration.baselines);
        assert(integration_data.dimension(0) == metadata.m_constants.channels);
        assert(integration_data.dimension(1) == integration.baselines);
        assert(metadata.oldUVW.size() == integration.baselines);
        assert(metadata.avg_data.rows() == integration.baselines);
        assert(metadata.avg_data.cols() == metadata.m_constants.num_pols);
        
        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline](2) - metadata.oldUVW[baseline](2);
            shiftFactor = shiftFactor - 2 * pi *
            (
                metadata.direction(0) * uvw[baseline](0)
                - metadata.direction(1) * uvw[baseline](1)
            );
            shiftFactor = shiftFactor + 2 * pi *
            (
                metadata.GetConstants().phase_centre_ra_rad * metadata.oldUVW[baseline](0)
                - metadata.GetConstants().phase_centre_dec_rad * metadata.oldUVW[baseline](1)
            );


            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.m_constants.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.m_constants.GetChannelWavelength(channel);

                for(int polarization = 0; polarization < integration_data.dimension(2); ++polarization)
                {
                    integration_data(channel, baseline, polarization) *= std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));
                }

                bool hasNaN = false;
                const Eigen::Tensor<std::complex<double>, 1> polarizations = integration_data.chip(channel, 0).chip(baseline, 0);
                for(int i = 0; i < polarizations.dimension(0); ++i)
                {
                    hasNaN |= polarizations(i).real() == NAN || polarizations(i).imag() == NAN;
                }

                if(!hasNaN)
                {
                    for(int polarization = 0; polarization < integration_data.dimension(2); ++polarization)
                    {
                        metadata.avg_data(baseline, polarization) += integration_data(channel, baseline, polarization);
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

        int STATIONS = A.cols(); //TODO verify correctness

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
        }

        A(k, refAnt) = 1;
        k++;
        
        auto Atemp = Eigen::MatrixXd(k, STATIONS);
        Atemp = A(Eigen::seqN(0, k), Eigen::seqN(0, STATIONS));
        A.resize(0,0);
        A = Atemp;

        auto Itemp = Eigen::VectorXi(k);
        Itemp = I(Eigen::seqN(0, k));
        I.resize(0);
        I = Itemp;
    

        return std::make_pair(A, I);
    }
}
}
