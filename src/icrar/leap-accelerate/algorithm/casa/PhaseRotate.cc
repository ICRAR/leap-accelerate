
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
#include <icrar/leap-accelerate/math/casa/matrix.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/common/stream_extensions.h>
#include <icrar/leap-accelerate/core/profiling_timer.h>

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

using namespace casacore;

namespace icrar
{
namespace casalib
{
    // leap_remote_calibration
    CalibrateResult Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<casacore::MVDirection>& directions)
    {
        BOOST_LOG_TRIVIAL(info) << "Starting Calibration using casa library";
        BOOST_LOG_TRIVIAL(info) << "rows: " << ms.GetNumRows() << ", "
        << "baselines: " << ms.GetNumBaselines() << ", "
        << "channels: " << ms.GetNumChannels() << ", "
        << "polarizations: " << ms.GetNumPols() << ", "
        << "directions: " << directions.size();

        auto timer = profiling_timer();
        timer.start();
        auto metadata = casalib::MetaData(ms);
        timer.stop();
        timer.log("metadata read time");
        timer.restart();


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

        timer.stop();
        timer.log("integration read time");
        timer.restart();

        for(size_t i = 0; i < directions.size(); ++i)
        {
            metadata = MetaData(ms);
            icrar::casalib::PhaseRotate(metadata, directions[i], input_queues[i], output_integrations[i], output_calibrations[i]);
        }

        timer.stop();
        timer.log("PhaseRotate time");

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
                icrar::casalib::RotateVisibilities(integration.get(), metadata, direction);
                output_integrations.emplace(direction, integration.get().integration_number, boost::none);
            }
            else
            {
                if(!metadata.avg_data.is_initialized())
                {
                    throw icrar::exception("avg_data must be initialized", __FILE__, __LINE__);
                }


                casacore::Matrix<Radians> avg_data_angles = casa_matrix_map(metadata.avg_data.get(), [](std::complex<double> c) -> Radians
                {
                    return std::arg(c);
                });

                auto indices1 = ToVector(metadata.I1);
                for(int& v : indices1)
                {
                    if(v < 0)
                    {
                        v += avg_data_angles.shape()[0]; // -behaviour in python is to wrap around
                        //TODO: reference antenna should be included, set to 0?
                    }
                }

                casacore::Matrix<double> cal_avg_data = ConvertMatrix(static_cast<Eigen::MatrixXd>(ToMatrix(avg_data_angles)(indices1, 0))); // 1st pol only
                // TODO: Value at last index of cal_avg_data must be 0 (which is the reference antenna phase value)
                //cal_avg_data(cal_avg_data.size() - 1, 0) = 0.0; // Value at last index of avg_data_t must be 0 (which is the reference antenna phase value)

                casacore::Matrix<double> cal1 = icrar::casalib::multiply(metadata.Ad1, cal_avg_data);

                Eigen::VectorXi indices = ToVector(metadata.I);
                for(int& v : indices)
                {
                    if(v < 0)
                    {
                        v += avg_data_angles.shape()[0]; // -behaviour in python is to wrap around
                        //TODO: reference antenna should be included, set to 0?
                    }
                }

                Eigen::MatrixXd e_avg_data_slice = ToMatrix(avg_data_angles)(indices, Eigen::all);
                casacore::Matrix<double> avg_data_slice = ConvertMatrix(e_avg_data_slice);

                // Calculate DInt
                casacore::Matrix<double> dInt = casacore::Matrix<double>(metadata.I.size(), avg_data_angles.shape()[1]);
                dInt = 0;
                for(size_t n = 0; n < metadata.I.size(); ++n)
                {
                    dInt.row(n) = avg_data_slice.row(n) - (casacore::sum((casacore::Array<double>)metadata.A.row(n) * (casacore::Array<double>)cal1.column(0)));
                }

                casacore::Matrix<double> dIntColumn = dInt.column(0); // 1st pol only
                cal.push_back(icrar::casalib::multiply(metadata.Ad, dIntColumn) + cal1);
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
        auto parameters = integration.parameters;

        if(!metadata.dd.is_initialized())
        {
            metadata.SetDD(direction);
            metadata.SetWv();
            // Zero a vector for averaging in time and freq
            metadata.avg_data = casacore::Matrix<DComplex>(integration.baselines, metadata.num_pols);
            metadata.avg_data.get() = 0;
            metadata.m_initialized = true;
        }
        metadata.CalcUVW(uvw);

        assert(uvw.size() == integration.baselines);
        assert(integration_data.dimension(0) == metadata.num_pols);
        assert(integration_data.dimension(1) == integration.baselines);
        assert(integration_data.dimension(2) == metadata.channels);

        assert(metadata.oldUVW.size() == integration.baselines);
        assert(metadata.channel_wavelength.size() == metadata.channels);

        // loop over baselines
        for(size_t baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            const double two_pi = 2 * boost::math::constants::pi<double>();

            double shiftFactor = -(uvw[baseline](2) - metadata.oldUVW[baseline](2)); // check these are correct

            shiftFactor +=
            (
                metadata.phase_centre_ra_rad * metadata.oldUVW[baseline](0)
                - metadata.phase_centre_dec_rad * metadata.oldUVW[baseline](1)
            );
            shiftFactor -=
            (
                //NOTE: polar direction
                direction.get()[0] * uvw[baseline](0)
                - direction.get()[1] * uvw[baseline](1)
            );
            shiftFactor *= two_pi;


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

    std::pair<casacore::Matrix<double>, casacore::Vector<std::int32_t>> PhaseMatrixFunction(
        const casacore::Vector<std::int32_t>& a1,
        const casacore::Vector<std::int32_t>& a2,
        int refAnt)
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

        Matrix<double> A = Matrix<double>(a1.size() + 1, std::max(icrar::ArrayMax(a1), icrar::ArrayMax(a2)) + 1);
        A = 0.0;

        Vector<int> I = Vector<int>(a1.size() + 1);
        I = -1;

        int STATIONS = A.shape()[1]; //TODO verify correctness
        int k = 0;

        for(size_t n = 0; n < a1.size(); n++)
        {
            if(a1(n) != a2(n))
            {
                if((refAnt < 0) || ((refAnt >= 0) && ((a1(n) == refAnt) || (a2(n) == refAnt))))
                {
                    A(k, a1(n)) = 1.0; // set scalear
                    A(k, a2(n)) = -1.0; // set scalear
                    I(k) = n; //set scalear
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
        
        auto Atemp = casacore::Matrix<double>(k, STATIONS);
        Atemp = A(Slice(0, k), Slice(0, STATIONS));
        A.resize(0,0);
        A = Atemp;

        auto Itemp = casacore::Vector<int>(k);
        Itemp = I(Slice(0, k));
        I.resize(0);
        I = Itemp;

        return std::make_pair(A, I);
    }
}
}
