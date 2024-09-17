/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */


#include "CpuLeapCalibrator.h"

#include <icrar/leap-accelerate/common/eigen_stringutils.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuComputeOptions.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cpu/LeapData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceLeapData.h>
#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/thread.hpp>

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
    void CpuLeapCalibrator::Calibrate(
        std::function<void(const cpu::Calibration&)> outputCallback,
        const icrar::MeasurementSet& ms,
        const std::vector<SphericalDirection>& directions,
        const Slice& solutionInterval,
        double minimumBaselineThreshold,
        bool computeCal1,
        boost::optional<unsigned int> referenceAntenna,
        const ComputeOptionsDTO& computeOptions)
    {
        auto cpuComputeOptions = CpuComputeOptions(computeOptions, ms);

        LOG(info) << "Starting calibration using cpu";
        LOG(info)
        << "stations: " << ms.GetNumStations() << ", "
        << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "solutionInterval: [" << solutionInterval.GetStart() << "," << solutionInterval.GetInterval() << "," << solutionInterval.GetEnd() << "], "
        << "reference antenna: " << referenceAntenna << ", "
        << "flagged baselines: " << ms.GetNumFlaggedBaselines() << ", "
        << "baseline threshold: " << minimumBaselineThreshold << "m, "
        << "short baselines: " << ms.GetNumShortBaselines(minimumBaselineThreshold) << ", "
        << "filtered baselines: " << ms.GetNumFilteredBaselines(minimumBaselineThreshold) << ", "
        << "compute cal1: " << computeCal1 << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumTimesteps();

        profiling::timer calibration_timer;
        Rangei validatedSolutionInterval = solutionInterval.Evaluate(boost::numeric_cast<int32_t>(ms.GetNumTimesteps()));
        std::vector<double> epochs = ms.GetEpochs();

        profiling::timer leapdata_read_timer;
        LOG(info) << "Loading LeapData";
        auto leapData = icrar::cpu::LeapData(
            ms,
            referenceAntenna,
            minimumBaselineThreshold,
            true,
            cpuComputeOptions.IsFileSystemCacheEnabled());
        LOG(info) << "leap data loaded in " << leapdata_read_timer;

        int32_t solutions = validatedSolutionInterval.GetSize();
        auto output_calibrations = std::vector<cpu::Calibration>();
        output_calibrations.reserve(solutions);

        constexpr unsigned int integrationNumber = 0;
        for(int32_t solution = 0; solution < solutions; ++solution)
        {
            profiling::timer solution_timer;
            auto input_queues = std::vector<std::vector<cpu::Integration>>();

            int32_t solution_start = validatedSolutionInterval.GetStart() + solution * validatedSolutionInterval.GetInterval();
            int32_t solution_stop = solution_start + validatedSolutionInterval.GetInterval();

            output_calibrations.emplace_back(
                epochs[solution_start] - ms.GetTimeInterval() * 0.5,
                epochs[solution_stop-1] + ms.GetTimeInterval() * 0.5
            );

            //Iterate solutions
            profiling::timer integration_read_timer;

            //Convert solution intervals to seperate constigous slices
            Slice solutionTimestepSlice(solution_start, solution_stop);
            const auto integration = Integration::CreateFromMS(
                    ms,
                    integrationNumber,
                    solutionTimestepSlice,
                    Slice(0, ms.GetNumPols(), ms.GetNumPols()-1)); // XX + YY pols
                    
            LOG(info) << "Read integration data in " << integration_read_timer;

            for(size_t direction = 0; direction < directions.size(); ++direction)
            {
                auto queue = std::vector<cpu::Integration>();
                queue.push_back(integration);
                input_queues.push_back(std::move(queue));
            }

            profiling::timer beam_calibrate_timer;
            for(size_t direction = 0; direction < directions.size(); ++direction)
            {
                LOG(info) << "Processing direction " << direction;
                leapData.SetDirection(directions[direction]);
                trace_matrix(leapData.GetDD(), "DD" + std::to_string(direction));

                leapData.GetAvgData().setConstant(std::complex<double>(0.0,0.0));
                BeamCalibrate(
                    leapData,
                    directions[direction],
                    computeCal1,
                    input_queues[direction],
                    output_calibrations[solution].GetBeamCalibrations());
            }

            LOG(info) << "Performed BeamCalibrate in " << beam_calibrate_timer;
            LOG(info) << "Calculated solution in " << solution_timer;

            profiling::timer write_timer;
            outputCallback(output_calibrations[solution]);
            LOG(info) << "Write out in " << write_timer;
        }
        LOG(info) << "Finished calibration in " << calibration_timer;
    }

    void CpuLeapCalibrator::BeamCalibrate(
        cpu::LeapData& leapData,
        const SphericalDirection& direction,
        bool computeCal1,
        std::vector<cpu::Integration>& integrations,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {
        for(auto& integration : integrations)
        {
            LOG(info) << "Rotating and Averaging " << integration.GetIntegrationNumber();
            PhaseRotateAverageVisibilities(integration, leapData);
        }

        trace_matrix(leapData.GetAvgData(), "AvgData");
        LOG(info) << "Applying Inversion";
        ApplyInversion(leapData, direction, computeCal1, output_calibrations);
    }

    void CpuLeapCalibrator::PhaseRotateAverageVisibilities(
        // const Eigen::Tensor<double, 3> uvw, TODO: move out uvw
        cpu::Integration& integration,
        cpu::LeapData& leapData)
    {
        using namespace std::literals::complex_literals;

        Eigen::Tensor<std::complex<double>, 4>& visibilities = integration.GetVis();

        for(size_t timestep = 0; timestep < integration.GetNumTimesteps(); ++timestep)
        {
            for(size_t baseline = 0; baseline < integration.GetNumBaselines(); ++baseline)
            {
                // TODO(calgray) Eigen::Tensor::chip creates array-bounds warnings on gcc-12+
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Warray-bounds"
                const Eigen::VectorXd uvw = ToVector(Eigen::Tensor<double, 1>(
                    integration.GetUVW()
                    .chip(timestep, 2)
                    .chip(baseline, 1)
                ));
                #pragma GCC diagnostic pop

                auto rotatedUVW = leapData.GetDD() * uvw;
                double shiftFactor = -two_pi<double>() * (rotatedUVW.z() - uvw.z());

                for(uint32_t channel = 0; channel < integration.GetNumChannels(); ++channel)
                {
                    // Phase Rotate
                    double shiftRad = shiftFactor / leapData.GetConstants().GetChannelWavelength(channel);

                    for(uint32_t polarization = 0; polarization < integration.GetNumPolarizations(); ++polarization)
                    {
                        //OPTIMIZATION: additonally pass const visibilities and write to unitialized memory buffer in integration
                        visibilities(polarization, channel, baseline, timestep) *= std::exp(std::complex<double>(0.0, shiftRad));
                    }

                    bool hasNaN = false;
                    for(uint32_t polarization = 0; polarization < integration.GetNumPolarizations(); ++polarization)
                    {
                        hasNaN |= std::isnan(visibilities(polarization, channel, baseline, timestep).real())
                               || std::isnan(visibilities(polarization, channel, baseline, timestep).imag());
                    }

                    if(!hasNaN)
                    {
                        // Averaging with XX and YY polarizations
                        leapData.GetAvgData()(baseline) += visibilities(0, channel, baseline, timestep);
                        leapData.GetAvgData()(baseline) += visibilities(visibilities.dimension(0) - 1, channel, baseline, timestep);
                    }
                }
            }
        }
    }

    void CpuLeapCalibrator::ApplyInversion(
        cpu::LeapData& leapData,
        const SphericalDirection& direction,
        bool computeCal1,
        std::vector<cpu::BeamCalibration>& output_calibrations)
    {
        auto avgDataI1 = leapData.GetAvgData().wrapped_row_select(leapData.GetI1());
        Eigen::VectorXd phaseAnglesI1 = avgDataI1.arg();

        // Value at last index of phaseAnglesI1 must be 0 (which is the reference antenna phase value)
        phaseAnglesI1.conservativeResize(phaseAnglesI1.rows() + 1);
        phaseAnglesI1(phaseAnglesI1.rows() - 1) = 0;

        Eigen::VectorXd cal1 = leapData.GetAd1() * phaseAnglesI1;
        if(!computeCal1)
        {
            cal1.setZero();
        }
        Eigen::VectorXd ACal1 = leapData.GetA() * cal1;

        Eigen::VectorXd deltaPhase = Eigen::VectorXd::Zero(leapData.GetI().size());
        for(int n = 0; n < leapData.GetI().size(); ++n)
        {
            deltaPhase(n) = std::arg(std::exp(std::complex<double>(0, -ACal1(n))) * leapData.GetAvgData()(n));
        }

        deltaPhase.conservativeResize(deltaPhase.size() + 1);
        deltaPhase(deltaPhase.size() - 1) = 0;

        output_calibrations.emplace_back(direction, (leapData.GetAd() * deltaPhase) + cal1);
    }

} // namespace cpu
} // namespace icrar
