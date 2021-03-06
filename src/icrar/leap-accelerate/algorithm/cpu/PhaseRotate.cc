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
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

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
using namespace boost::math::constants;

namespace icrar
{
namespace cpu
{
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions,
        double minimumBaselineThreshold,
		bool isFileSystemCacheEnabled)
    {
        LOG(info) << "Starting Calibration using cpu";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << "m, "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumRows() / ms.GetNumBaselines();

        profiling::timer calibration_timer;

        auto output_integrations = std::vector<std::vector<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::vector<cpu::CalibrationResult>>();
        auto input_queues = std::vector<std::vector<cpu::Integration>>();

        profiling::timer integration_read_timer;

        constexpr unsigned int integrationNumber = 0;

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
        auto metadata = icrar::cpu::MetaData(ms, integration.GetUVW(), minimumBaselineThreshold, isFileSystemCacheEnabled);
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

    void PhaseRotate(
        cpu::MetaData& metadata,
        const icrar::MVDirection& direction,
        std::vector<cpu::Integration>& input,
        std::vector<cpu::IntegrationResult>& output_integrations,
        std::vector<cpu::CalibrationResult>& output_calibrations)
    {
        for(auto& integration : input)
        {
            LOG(info) << "Rotating Integration " << integration.GetIntegrationNumber();
            icrar::cpu::RotateVisibilities(integration, metadata);
            output_integrations.emplace_back(integration.GetIntegrationNumber(), direction, boost::none);
        }
        trace_matrix(metadata.GetAvgData(), "avg_data");

        LOG(info) << "Calculating Calibration";

        auto phaseAngles = icrar::arg(metadata.GetAvgData());

        // PhaseAngles I1
        // Value at last index of phaseAnglesI1 must be 0 (which is the reference antenna phase value)
        Eigen::VectorXd phaseAnglesI1 = icrar::cpu::VectorRangeSelect(phaseAngles, metadata.GetI1(), 0); // 1st pol only
        phaseAnglesI1.conservativeResize(phaseAnglesI1.rows() + 1);
        phaseAnglesI1(phaseAnglesI1.rows() - 1) = 0;

        Eigen::VectorXd cal1 = metadata.GetAd1() * phaseAnglesI1;
        Eigen::MatrixXd dInt = Eigen::MatrixXd::Zero(metadata.GetI().size(), metadata.GetAvgData().cols());
        for(int n = 0; n < metadata.GetI().size(); ++n)
        {
            double sum = metadata.GetA()(n, Eigen::all) * cal1;
            dInt(n, Eigen::all) = icrar::arg(std::exp(std::complex<double>(0, -sum * two_pi<double>())) * metadata.GetAvgData()(n, Eigen::all));
        }

        Eigen::VectorXd deltaPhaseColumn = dInt(Eigen::all, 0); // 1st pol only
        deltaPhaseColumn.conservativeResize(deltaPhaseColumn.size() + 1);
        deltaPhaseColumn(deltaPhaseColumn.size() - 1) = 0;

        output_calibrations.emplace_back(direction, (metadata.GetAd() * deltaPhaseColumn) + cal1);
    }

    void RotateVisibilities(cpu::Integration& integration, cpu::MetaData& metadata)
    {
        using namespace std::literals::complex_literals;
        Eigen::Tensor<std::complex<double>, 3>& integration_data = integration.GetVis();

        // loop over smeared baselines
        for(size_t baseline = 0; baseline < integration.GetBaselines(); ++baseline)
        {
            int md_baseline = static_cast<int>(baseline % static_cast<size_t>(metadata.GetConstants().nbaselines)); // metadata baseline

            double shiftFactor = two_pi<double>() * (metadata.GetUVW()[baseline](2) - metadata.GetOldUVW()[baseline](2));

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
} // namespace cpu
} // namespace icrar
