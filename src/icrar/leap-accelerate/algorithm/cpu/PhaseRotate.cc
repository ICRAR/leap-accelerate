
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

#include <icrar/leap-accelerate/model/cpu/Integration.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>

#include <icrar/leap-accelerate/core/logging.h>
#include <icrar/leap-accelerate/core/profiling_timer.h>

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

namespace icrar
{
namespace cpu
{
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions,
        int solutionInterval)
    {
        BOOST_LOG_TRIVIAL(info) << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size();

        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queues = std::vector<std::vector<cpu::Integration>>();
        
        auto timer = profiling_timer();
        timer.start();

        unsigned int integrationNumber = 0;

        // Flooring to remove incomplete measurements
        int integrations = ms.GetNumRows() / ms.GetNumBaselines();
        auto integration = Integration(
                integrationNumber,
                ms,
                0,
                ms.GetNumChannels(),
                integrations * ms.GetNumBaselines(),
                ms.GetNumPols());

        for(int i = 0; i < directions.size(); ++i)
        {
            auto queue = std::vector<cpu::Integration>();
            queue.push_back(integration);

            input_queues.push_back(queue);
            output_integrations.emplace_back();
            output_calibrations.emplace_back();
        }
        
        timer.stop();
        timer.log("integration read time");
        timer.restart();
        auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW());

        timer.stop();
        timer.log("metadata read time");
        timer.restart();
        for(int i = 0; i < directions.size(); ++i)
        {
            metadata.SetDD(directions[i]);
            metadata.CalcUVW();
            metadata.avg_data.setConstant(std::complex<double>(0.0,0.0));
            icrar::cpu::PhaseRotate(metadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }

        timer.stop();
        timer.log("PhaseRotate time");

        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    void PhaseRotate(
        cpu::MetaData& metadata,
        const icrar::MVDirection& direction,
        std::vector<cpu::Integration>& input,
        std::vector<cpu::IntegrationResult>& output_integrations,
        std::vector<cpu::CalibrationResult>& output_calibrations)
    {
        auto cal = std::vector<casacore::Matrix<double>>();
        for(auto& integration : input)
        {
            icrar::cpu::RotateVisibilities(integration, metadata);
            output_integrations.emplace_back(direction, integration.integration_number, boost::none);
        }


        auto avg_data_angles = metadata.avg_data.unaryExpr([](std::complex<double> c) -> Radians { return std::arg(c); });
        
        Eigen::VectorXi indexes1 = metadata.GetI1();
        indexes1(indexes1.size() + 1) = 0; // TODO: check -1 behaviour, should result in 0
        Eigen::VectorXd cal_avg_data = avg_data_angles(indexes1, 0); // 1st pol only
        cal_avg_data(cal_avg_data.size() - 1) = 0.0; // Value at last index of avg_data_t must be 0 (which is the reference antenna phase value)


        auto cal1 = metadata.GetAd1() * cal_avg_data;
        std::cout << "cal1(0):" << cal1(1,0) << std::endl;
        std::cout << "cal1("<<cal1.rows()-1<<"):" << cal1(cal1.rows()-1,0) << std::endl;

        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(metadata.GetI().size(), metadata.avg_data.cols());
        Eigen::VectorXi indexes = metadata.GetI();
        indexes(indexes.size() + 1) = 0; // HACK
        
        Eigen::MatrixXd avg_data_slice = avg_data_angles(indexes, Eigen::all);
        avg_data_slice(avg_data_slice.rows() - 1, Eigen::all).setConstant(0.0);
        std::cout << "avg_data_slice(0):" << avg_data_slice(0,0) << std::endl;
        std::cout << "avg_data_slice("<<avg_data_slice.rows()-1<<"):" << avg_data_slice(avg_data_slice.rows()-1,0) << std::endl;
        
        for(int n = 0; n < metadata.GetI().size(); ++n)
        {
            Eigen::MatrixXd cumsum = metadata.GetA()(n, Eigen::all) * cal1;
            double sum = cumsum.sum();
            dInt(n, Eigen::all) = avg_data_slice(n, Eigen::all).unaryExpr([&](double v) { return v - sum; });
        }
        std::cout << "dInt(0,0):" << dInt(0,0) << std::endl;
        std::cout << "dInt("<<dInt.rows()-1<<",0):" << dInt(dInt.rows()-1,0) << std::endl;

        Eigen::MatrixXd dIntColumn = dInt(Eigen::all, 0); // 1st pol only
        assert(dIntColumn.cols() == 1);

        cal.push_back(ConvertMatrix(Eigen::MatrixXd((metadata.GetAd() * dIntColumn) + cal1)));

        output_calibrations.emplace_back(direction, cal);
    }

    void RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 3>& integration_data = integration.GetData();

        metadata.CalcUVW();
        const auto polar_direction = icrar::to_polar(metadata.direction);
        
        // loop over smeared baselines
        int baselines = metadata.GetConstants().nbaselines;
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            int md_baseline = baseline % metadata.GetConstants().nbaselines; //metadata baseline

            constexpr double two_pi = 2 * boost::math::constants::pi<double>();

            double shiftFactor = -(metadata.GetUVW()[baseline](2) - metadata.GetOldUVW()[baseline](2));

            shiftFactor +=
            (
                metadata.GetConstants().phase_centre_ra_rad * metadata.GetOldUVW()[baseline](0)
                - metadata.GetConstants().phase_centre_dec_rad * metadata.GetOldUVW()[baseline](1)
            );
            shiftFactor -=
            (
                polar_direction(0) * metadata.GetUVW()[baseline](0)
                - polar_direction(1) * metadata.GetUVW()[baseline](1)
            );
            shiftFactor *= two_pi;

            // Loop over channels
            for(int channel = 0; channel < metadata.GetConstants().channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.GetConstants().GetChannelWavelength(channel);
                for(int polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                {
                    integration_data(polarization, baseline, channel) *= std::exp(std::complex<double>(0.0, shiftRad));
                }

                bool hasNaN = false;
                const Eigen::Tensor<std::complex<double>, 1> polarizations = integration_data.chip(channel, 2).chip(baseline, 1);
                for(int polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                {
                    hasNaN |= std::isnan(polarizations(polarization).real()) || std::isnan(polarizations(polarization).imag());
                }

                if(!hasNaN)
                {
                    for(int polarization = 0; polarization < metadata.GetConstants().num_pols; ++polarization)
                    {
                        metadata.avg_data(md_baseline, polarization) += integration_data(polarization, baseline, channel);
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

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(a1.size() + 1, std::max(a1.maxCoeff(), a2.maxCoeff()) + 1);
        int STATIONS = A.cols(); //TODO verify correctness

        Eigen::VectorXi I = Eigen::VectorXi(a1.size() + 1);
        I.setConstant(-1);

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
