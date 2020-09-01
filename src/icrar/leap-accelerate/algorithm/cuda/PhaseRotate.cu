
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

#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math.h>

#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

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

    // template<int Dims>
    // __device __forceinline__ cuDoubleComplex cuCexp(const Eigen::Tensor<cuDoubleComplex, Dims>& tensor)
    // {
        
    // }

    __global__ void g_RotateVisibilities(
        cuDoubleComplex* pintegration_data, int integration_data_rows, int integration_data_cols, int integration_data_depth,
        int integration_channels,
        int integration_baselines,
        int polarizations,
        icrar::cpu::Constants constants,
        Eigen::Matrix3d dd,
        double2 direction,
        double3* uvw, int uvwLength,
        double3* oldUVW, int oldUVWLegth,
        cuDoubleComplex* pavg_data, int avg_dataRows, int avg_dataCols)
    {
        using Tensor2Xcucd = Eigen::Tensor<cuDoubleComplex, 2>;
        using Tensor3Xcucd = Eigen::Tensor<cuDoubleComplex, 3>;

        auto integration_data = Eigen::TensorMap<Tensor3Xcucd>(pintegration_data, integration_data_rows, integration_data_cols, integration_data_depth);
        auto avg_data = Eigen::TensorMap<Tensor2Xcucd>(pavg_data, avg_dataRows, avg_dataCols);

        // /// loop over baselines
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
                cuDoubleComplex exp = cuCexp(make_cuDoubleComplex(0.0, shiftRad));

                for(int polarization = 0; polarization < polarizations; polarization++)
                {
                    integration_data(channel, baseline, polarization) = cuCmul(integration_data(channel, baseline, polarization), exp);
                }

                bool hasNaN = false;
                for(int polarization = 0; polarization < polarizations; polarization++)
                {
                    auto n = integration_data(channel, baseline, polarization);
                    hasNaN |= n.x == NAN || n.y == NAN;
                }

                if(!hasNaN)
                {
                    for(int polarization = 0; polarization < polarizations; ++polarization)
                    {
                        avg_data(baseline, polarization) = cuCadd(avg_data(baseline, polarization), integration_data(channel, baseline, polarization));
                    }
                }
            }
        }
    }

    __host__ void RotateVisibilities(
        DeviceIntegration& integration,
        DeviceMetaData& metadata)
    {
        assert(metadata.GetConstants().channels == integration.channels && integration.channels == integration.data.GetDimensionSize(0));
        assert(integration.baselines == integration.data.GetDimensionSize(1));
        assert(metadata.GetConstants().num_pols == integration.data.GetDimensionSize(2));

        // TODO: calculate grid size using constants.channels, integration_baselines, integration_data(channel, baseline).cols()
        // unpack metadata
        g_RotateVisibilities<<<1,1,1>>>(
            (cuDoubleComplex*)integration.data.Get(), integration.data.GetDimensionSize(0), integration.data.GetDimensionSize(1), integration.data.GetDimensionSize(2),
            integration.channels, integration.baselines, metadata.GetConstants().num_pols,
            metadata.GetConstants(),
            metadata.dd,
            make_double2(metadata.direction(0), metadata.direction(1)),
            (double3*)metadata.UVW.Get(), metadata.UVW.GetCount(),
            (double3*)metadata.oldUVW.Get(), metadata.oldUVW.GetCount(),
            (cuDoubleComplex*)metadata.avg_data.Get(), metadata.avg_data.GetRows(), metadata.avg_data.GetCols());
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
