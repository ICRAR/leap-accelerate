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

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/model/cpu/Integration.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>

#include <icrar/leap-accelerate/common/eigen_extensions.h>

#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

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

#include <sstream>


using Radians = double;

namespace icrar
{
namespace cpu
{
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions)
    {
        LOG(info) << "Starting Calibration using cpu";
        LOG(info)
		<< "stations: " << ms.GetNumStations() << ", "
		<< "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumRows() / ms.GetNumBaselines();

        profiling::timer calibration_timer;

        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queues = std::vector<std::vector<cpu::Integration>>();

        profiling::timer integration_read_timer;

        unsigned int integrationNumber = 0;

        auto integration = Integration(
                integrationNumber,
                ms,
                0,
                ms.GetNumChannels(),
                ms.GetNumRows(),
                ms.GetNumPols());

        for(size_t i = 0; i < directions.size(); ++i)
        {
            auto queue = std::vector<cpu::Integration>();
            queue.push_back(integration);

            input_queues.push_back(queue);
            output_integrations.emplace_back();
            output_calibrations.emplace_back();
        }
        LOG(info) << "Read integration data in " << integration_read_timer;

        profiling::timer metadata_read_timer;
        LOG(info) << "Loading MetaData";
        auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW());
        LOG(info) << "Read metadata in " << metadata_read_timer;

        profiling::timer phase_rotate_timer;
        for(size_t i = 0; i < directions.size(); ++i)
        {
            LOG(info) << "Processing direction " << i;
            metadata.SetDD(directions[i]);
            metadata.CalcUVW();
            metadata.GetAvgData().setConstant(std::complex<double>(0.0,0.0));
            icrar::cpu::PhaseRotate(metadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }
        LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;

        LOG(info) << "Finished calibration in " << calibration_timer;
        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    double get_arg(std::complex<double> v)
    {
        return std::arg(v);
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
            LOG(info) << "Rotating Integration " << integration.GetIntegrationNumber();
            icrar::cpu::RotateVisibilities(integration, metadata);
            output_integrations.emplace_back(direction, integration.GetIntegrationNumber(), boost::none);
        }
        trace_matrix(metadata.GetAvgData(), "avg_data");

        LOG(info) << "Calculating Calibration";

        auto phaseAverages = metadata.GetAvgData().unaryExpr([](std::complex<double> c){ return std::arg(c); });
        
        // PhaseAverages I1
        Eigen::VectorXd phaseAveragesI1 = icrar::cpu::VectorRangeSelect(phaseAverages, metadata.GetI1(), 0); // 1st pol only
        // Value at last index of phaseAveragesI1 must be 0 (which is the reference antenna phase value)
        phaseAveragesI1.conservativeResize(phaseAveragesI1.rows() + 1);
        phaseAveragesI1(phaseAveragesI1.rows() - 1) = 0;

        // PhaseAverages I
        Eigen::MatrixXd phaseAveragesI = icrar::cpu::MatrixRangeSelect(phaseAverages, metadata.GetI(), Eigen::all);
        // Value at last index of phaseAveragesI must be 0 (which is the reference antenna phase value)
        //phaseAveragesI.conservativeResize(phaseAveragesI.cols() + 1);
        //phaseAveragesI(phaseAveragesI.size() - 1) = 0;

        if(metadata.GetAd1().cols() != phaseAveragesI1.rows())
        {
            std::cout << metadata.GetAd1().cols() << "!=" << phaseAveragesI1.rows() << std::endl;
            throw exception("Ad1 x cal_avg_data", __FILE__, __LINE__);
        }
        Eigen::VectorXd cal1 = metadata.GetAd1() * phaseAveragesI1;
        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(metadata.GetI().size(), metadata.GetAvgData().cols());

        for(int n = 0; n < metadata.GetI().size(); ++n)
        {
            double sum = metadata.GetA()(n, Eigen::all) * cal1;
            dInt(n, Eigen::all) = phaseAveragesI(n, Eigen::all).unaryExpr([&](double v) { return v - sum; });
        }

        Eigen::VectorXd dIntColumn = dInt(Eigen::all, 0); // 1st pol only
        dIntColumn.conservativeResize(dIntColumn.size() + 1);
        dIntColumn(dIntColumn.size() - 1) = 0;

        cal.push_back(ConvertMatrix(Eigen::MatrixXd((metadata.GetAd() * dIntColumn) + cal1)));

        output_calibrations.emplace_back(direction, cal);
    }

    void RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 3>& integration_data = integration.GetVis();

        metadata.CalcUVW();

        const auto polar_direction = icrar::ToPolar(metadata.GetDirection());
        
        // loop over smeared baselines
        for(size_t baseline = 0; baseline < integration.GetBaselines(); ++baseline)
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
                        metadata.GetAvgData()(md_baseline, polarization) += integration_data(polarization, baseline, channel);
                    }
                }
            }
        }
    }
}
}
