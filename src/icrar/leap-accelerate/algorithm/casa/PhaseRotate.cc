
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

#include <icrar/leap-accelerate/algorithm/casa/PhaseMatrixFunction.h>

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <icrar/leap-accelerate/math/casa/vector.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/casa/matrix.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/stream_extensions.h>
#include <icrar/leap-accelerate/core/profiling/timer.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>

#include <utility>
#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <exception>
#include <memory>
#include <vector>
#include <chrono>

using Radians = double;
using namespace boost::math::constants;

namespace icrar
{
namespace casalib
{
    // leap_remote_calibration
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<casacore::MVDirection>& directions,
        double minimumBaselineThreshold)
    {
        LOG(info) << "Starting Calibration using casa library";
        LOG(info) << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "min baseline length: " << minimumBaselineThreshold << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size() << ", "
        << "timesteps: " << ms.GetNumRows() / ms.GetNumBaselines();

        profiling::timer calibration_timer;

        profiling::timer metadata_read_timer;
        auto metadata = casalib::MetaData(ms, minimumBaselineThreshold);
        LOG(info) << "Read metadata in " << metadata_read_timer;

        profiling::timer integration_read_timer;
        auto output_integrations = std::vector<std::queue<IntegrationResult>>();
        auto output_calibrations = std::vector<std::queue<CalibrationResult>>();
        auto input_queues = std::vector<std::queue<Integration>>();
        for(size_t i = 0; i < directions.size(); ++i)
        {
            auto queue = std::queue<Integration>(); 
            int startRow = 0;
            int integrationNumber = 0;
            while((startRow + ms.GetNumBaselines()) < ms.GetNumRows())
            {
                queue.emplace(
                    integrationNumber++,
                    ms,
                    startRow,
                    metadata.channels,
                    metadata.GetBaselines(),
                    metadata.num_pols);
                startRow += metadata.GetBaselines();
            }
            assert(metadata.channels == queue.front().data.dimension(2)); //metadata.channels
            assert(metadata.GetBaselines() == queue.front().data.dimension(1)); //metadata.baselines
            assert(metadata.num_pols == queue.front().data.dimension(0)); //metadata.polarizations
            input_queues.push_back(queue);
            output_integrations.emplace_back();
            output_calibrations.emplace_back();
        }
        LOG(info) << "Read integration data in " << integration_read_timer;

        profiling::timer phase_rotate_timer;
        for(size_t i = 0; i < directions.size(); ++i)
        {
            metadata = casalib::MetaData(ms, minimumBaselineThreshold);
            casalib::PhaseRotate(metadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }
        LOG(info) << "Performed PhaseRotate in " << phase_rotate_timer;

        LOG(info) << "Finished calibration in " << calibration_timer;
        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    //leap_calibrate_from_queue
    void PhaseRotate(
        MetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        using namespace std::complex_literals;
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
                casalib::RotateVisibilities(integration.get(), metadata, direction);
                output_integrations.emplace(integration.get().integration_number, direction, boost::none);
            }
            else
            {
                if(!metadata.avg_data.is_initialized())
                {
                    throw exception("avg_data must be initialized", __FILE__, __LINE__);
                }

                casacore::Matrix<Radians> phaseAngles = casa_matrix_map(metadata.avg_data.get(), [](std::complex<double> c)
                {
                    return std::arg(c);
                });

                auto e_phaseAngles = ToMatrix(phaseAngles);
                Eigen::VectorXd e_phaseAnglesI1 = cpu::VectorRangeSelect(e_phaseAngles, ToVector(metadata.I1), 0); // 1st pol only
                // Value at last index of phaseAnglesI1 must be 0 (which is the reference antenna phase value)
                e_phaseAnglesI1.conservativeResize(e_phaseAnglesI1.rows() + 1);
                e_phaseAnglesI1(e_phaseAnglesI1.size() - 1) = 0.0;
                auto phaseAnglesI1 = ConvertVector(e_phaseAnglesI1);
                
                casacore::Matrix<double> cal1 = casalib::multiply(metadata.Ad1, phaseAnglesI1);

                Eigen::MatrixXd e_phaseAnglesI = cpu::MatrixRangeSelect(e_phaseAngles, ToVector(metadata.I), Eigen::all);
                casacore::Matrix<double> phaseAnglesI = ConvertMatrix(e_phaseAnglesI);

                // Calculate DInt
                casacore::Matrix<double> dInt = casacore::Matrix<double>(metadata.I.size() + 1, phaseAngles.shape()[1]);
                dInt = 0;

                for(size_t n = 0; n < metadata.I.size(); ++n)
                {
                    double sum = casacore::sum((casacore::Array<double>)metadata.A.row(n) * (casacore::Array<double>)cal1.column(0));
                    auto scalear = std::exp(std::complex<double>(0.0, -sum * two_pi<double>()));
                    dInt.row(n) = casalib::arg(casalib::multiply(scalear, metadata.avg_data.get().row(n)));
                }
                dInt(dInt.shape()[0] - 1, 0) = 0;

                casacore::Matrix<double> dIntColumn = dInt.column(0); // 1st pol only

                cal.push_back(casalib::multiply(metadata.Ad, dIntColumn) + cal1);
                break;
            }
        }

        output_calibrations.emplace(direction, cal);
    }

    void RotateVisibilities(Integration& integration, MetaData& metadata, const casacore::MVDirection& direction)
    {
        using namespace std::literals::complex_literals;
        
        auto& integration_data = integration.data;
        auto& uvw = integration.uvw;

        if(!metadata.dd.is_initialized())
        {
            metadata.SetDD(direction);
            metadata.SetWv();
            
            // Allocate a zero vector for averaging in time and freq
            metadata.avg_data = casacore::Matrix<casacore::DComplex>(integration.baselines, metadata.num_pols);
            metadata.avg_data.get() = 0;
            metadata.m_initialized = true;
        }
        metadata.CalcUVW(uvw);

        assert(uvw.size() == integration.baselines);
        assert(integration_data.dimension(0) == static_cast<std::int64_t>(metadata.num_pols));
        assert(integration_data.dimension(1) == static_cast<std::int64_t>(integration.baselines));
        assert(integration_data.dimension(2) == static_cast<std::int64_t>(metadata.channels));

        assert(metadata.oldUVW.size() == integration.baselines);
        assert(metadata.channel_wavelength.size() == static_cast<size_t>(metadata.channels));

        // loop over baselines
        for(size_t baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            double shiftFactor = two_pi<double>() * (uvw[baseline](2) - metadata.oldUVW[baseline](2));

            // Loop over channels
            for(int channel = 0; channel < metadata.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.channel_wavelength[channel];

                for(int polarization = 0; polarization < metadata.num_pols; polarization++)
                {
                    integration_data(polarization, baseline, channel) *= std::exp((std::complex<double>(0.0, 1.0)) * std::complex<double>(shiftRad, 0.0));
                }

                bool hasNaN = false;

                const Eigen::Tensor<std::complex<double>, 1> polarizations = integration_data.chip(channel, 2).chip(baseline, 1);
                for(int i = 0; i < metadata.num_pols; ++i)
                {
                    hasNaN |= std::isnan(polarizations(i).real()) || std::isnan(polarizations(i).imag());
                }

                if(!hasNaN)
                {
                    for(int polarization = 0; polarization < metadata.num_pols; polarization++)
                    {
                        metadata.avg_data.get()(baseline, polarization) += integration_data(polarization, baseline, channel);
                    }
                }
            }
        }
    }
}
}
