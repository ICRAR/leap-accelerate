
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

#include <icrar/leap-accelerate/model/cuda/MetaDataCuda.h>

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

#include <cuComplex.h>
#include <math_constants.h>

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
        DeviceMetaData& metadata,
        const casacore::MVDirection& direction,
        std::queue<Integration>& input,
        std::queue<IntegrationResult>& output_integrations,
        std::queue<CalibrationResult>& output_calibrations)
    {
        throw std::runtime_error("not implemented"); //TODO
    }

    __device__ __forceinline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
    {
        // see https://forums.developer.nvidia.com/t/complex-number-exponential-function/24696/2
        cuDoubleComplex res;
        sincos(z.y, &res.y, &res.x);
        double t = exp(z.x);
        res.x *= t;
        res.y *= t;
        return res;
    }

    __global__ void g_RotateVisibilities(
        //Integration integration,
        Constants constants,
        Eigen::Matrix3d dd,
        double2 direction,
        double3* uvw, int uvwLength,
        double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pavg_data, int avg_dataRows, int avg_dataCols)
    {
        using VectorXcucd = Eigen::Matrix<cuDoubleComplex, Eigen::Dynamic, 1>;
        using MatrixXcucd = Eigen::Matrix<cuDoubleComplex, Eigen::Dynamic, Eigen::Dynamic>;
        using Tensor3Xcucd = Eigen::Matrix<VectorXcucd, Eigen::Dynamic, Eigen::Dynamic>;

        int integration_baselines = 0;
        Tensor3Xcucd integration_data; //TODO: incomplete
        Eigen::Map<MatrixXcucd> avg_data = Eigen::Map<MatrixXcucd>(pavg_data, avg_dataRows, avg_dataCols);

        /// loop over baselines
        for(int baseline = 0; baseline < integration_baselines; ++baseline)
        {
            const double pi = CUDART_PI;
            double shiftFactor = -2 * pi * uvw[baseline].z - oldUVW[baseline].z;
            shiftFactor = shiftFactor + 2 * pi * (constants.phase_centre_ra_rad * oldUVW[baseline].x);
            shiftFactor = shiftFactor - 2 * pi * (direction.x * uvw[baseline].x - direction.y * uvw[baseline].y);

            // loop over channels
            for(int channel = 0; channel < constants.channels; channel++)
            {
                double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);
                double rs = sin(shiftRad);
                double rc = cos(shiftRad);
                VectorXcucd v = integration_data(channel, baseline);

                //integration_data(channel, baseline) = v * std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));
                integration_data(channel, baseline) = v;
                cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));
                for(int i = 0; i < integration_data(channel, baseline).cols(); i++)
                {
                    integration_data(channel, baseline)(i) = cuCmul(integration_data(channel, baseline)(i), exp);
                }

                for(int i = 0; i < integration_data(channel, baseline).cols(); i++)
                {
                    //if(!integration_data(channel, baseline).hasNaN())
                    {
                        // make_cuDoubleComplex(0.0, 0.0);
                        avg_data(baseline, i) = cuCadd(avg_data(baseline, i), integration_data(channel, baseline)(i));
                    }
                }
            }
        }
    }

    __host__ void RotateVisibilities(
        Integration& integration,
        DeviceMetaData& metadata)
    {
        //unpack metadata
        g_RotateVisibilities<<<1,1>>>(
            //integration,
            metadata.constants,
            metadata.dd,
            make_double2(metadata.direction.get()[0], metadata.direction.get()[1]),
            (double3*)metadata.UVW.Get(), metadata.UVW.GetCount(), //TODO: change uvw to double3
            (double3*)metadata.oldUVW.Get(), metadata.oldUVW.GetCount(), //TODO: change olduvw to double3
            (cuDoubleComplex*)metadata.avg_data.Get(), metadata.avg_data.GetRows(), metadata.avg_data.GetCols());
    }

    __host__ void RotateVisibilitiesExample(
        Integration& integration,
        DeviceMetaData& metadata)
    {
        // using namespace std::literals::complex_literals;
        // auto& data = integration.data;
        // auto& uvw = metadata.UVW;
        // auto parameters = integration.parameters;

        // assert(uvw.GetCount() == integration.baselines);
        // assert(data.rows() == metadata.constants.channels);
        // assert(data.cols() == integration.baselines);
        // assert(metadata.oldUVW.GetCount() == integration.baselines);
        // assert(metadata.constants.channel_wavelength.size() == metadata.constants.channels);
        // assert(metadata.avg_data.GetRows() == integration.baselines);
        // assert(metadata.avg_data.GetCols() == metadata.constants.num_pols);

        // // loop over baselines
        // for(int baseline = 0; baseline < integration.baselines; ++baseline)
        // {
        //     const double pi = boost::math::constants::pi<double>();
        //     double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
        //     shiftFactor = shiftFactor + 2 * pi * (metadata.constants.phase_centre_ra_rad * metadata.oldUVW[baseline].get()[0]);
        //     shiftFactor = shiftFactor - 2 * pi * (metadata.direction.get()[0] * uvw[baseline].get()[0] - metadata.direction.get()[1] * uvw[baseline].get()[1]);

        //     if(baseline % 1000 == 1)
        //     {
        //         std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
        //     }

        //     // Loop over channels
        //     for(int channel = 0; channel < metadata.constants.channels; channel++)
        //     {
        //         double shiftRad = shiftFactor / metadata.constants.channel_wavelength[channel];
        //         double rs = sin(shiftRad);
        //         double rc = cos(shiftRad);
        //         Eigen::VectorXcd v = data(channel, baseline);

        //         data(channel, baseline) = v * std::exp(std::complex<double>(0.0, 1.0) * std::complex<double>(shiftRad, 0.0));

        //         if(!data(channel, baseline).hasNaN())
        //         {
        //             for(int i = 0; i < data(channel, baseline).cols(); i++)
        //             {
        //                 metadata.avg_data(baseline, i) += data(channel, baseline)(i);
        //             }
        //         }
        //     }
        // }
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

        auto unique = std::set<std::int32_t>(a1.cbegin(), a1.cend());
        unique.insert(a2.cbegin(), a2.cend());
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
            Atemp = A(Eigen::seqN(0, k), Eigen::seqN(0, STATIONS));
            A.resize(0,0);
            A = Atemp;

            auto Itemp = Eigen::VectorXi(k);
            Itemp = I(Eigen::seqN(0, k));
            I.resize(0);
            I = Itemp;
        }

        return std::make_pair(A, I);
    }
}
}
