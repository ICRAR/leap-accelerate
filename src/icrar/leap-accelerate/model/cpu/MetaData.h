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

#pragma once

#if CUDA_ENABLED
#include <cuda_runtime.h>
#else
#ifndef __host__
#define __host__
#endif // __host__
#ifndef __device__
#define __device__
#endif // __device__
#endif // CUDA_ENABLED

#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/common/constants.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <Eigen/Core>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
    class MeasurementSet;
    namespace cuda
    {
        class DeviceMetaData;
        class ConstantBuffer;
    } // namespace cuda
} // namespace icrar

namespace icrar
{
namespace cpu
{
    struct Constants
    {
        int nbaselines; //the total number station pairs (excluding self cycles) 

        int channels; // The number of channels of the current observation
        int num_pols; // The number of polarizations used by the current observation
        int stations; // The number of stations used by the current observation
        int rows;

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

        double phase_centre_ra_rad;
        double phase_centre_dec_rad;
        double dlm_ra;
        double dlm_dec;

        __host__ __device__ double GetChannelWavelength(int i) const
        {
            return constants::speed_of_light / (freq_start_hz + i * freq_inc_hz);
        }

        bool operator==(const Constants& rhs) const;
    };

    /**
     * @brief container of phaserotation constants and variables
     * 
     */
    class MetaData
    {
        MetaData() = default;

        Constants m_constants;
        double m_minimumBaselineThreshold;

        Eigen::MatrixXd m_A;
        Eigen::VectorXi m_I; // The flagged indexes of A
        Eigen::MatrixXd m_Ad; // The pseudo-inverse of m_A

        Eigen::MatrixXd m_A1;
        Eigen::VectorXi m_I1;
        Eigen::MatrixXd m_Ad1;

        std::vector<icrar::MVuvw> m_oldUVW;
        std::vector<icrar::MVuvw> m_UVW; // late initialized
    
        icrar::MVDirection m_direction; // calibration direction, late initialized

        Eigen::Matrix3d m_dd; // direction matrix, late initialized
        
        Eigen::MatrixXcd m_avgData; // matrix of size (baselines, polarizations), late initialized
    
    public:
        /**
         * @brief Construct a new MetaData object. SetDirection() must be called after construction
         * 
         * @param ms 
         * @param uvws 
         * @param minimumBaselineThreshold
         * @param useCache
         */
        MetaData(const icrar::MeasurementSet& ms, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold = 0.0, bool useCache = true);

        /**
         * @brief Construct a new MetaData object
         * 
         * @param ms 
         * @param uvws 
         * @param minimumBaselineThreshold
         * @param useCache
         */
        MetaData(const icrar::MeasurementSet& ms, const icrar::MVDirection& direction, const std::vector<icrar::MVuvw>& uvws, double minimumBaselineThreshold = 0.0, bool useCache = true);

        const Constants& GetConstants() const;

        const Eigen::MatrixXd& GetA() const;
        const Eigen::VectorXi& GetI() const;
        const Eigen::MatrixXd& GetAd() const;

        const Eigen::MatrixXd& GetA1() const;
        const Eigen::VectorXi& GetI1() const;
        const Eigen::MatrixXd& GetAd1() const;

        const std::vector<icrar::MVuvw>& GetOldUVW() const { return m_oldUVW; }
        const std::vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }

        const icrar::MVDirection& GetDirection() const { return m_direction; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }
        void SetDirection(const icrar::MVDirection& direction);

        void SetOldUVW(const std::vector<icrar::MVuvw>& uvws);

        /**
         * @brief Updates the rotated UVW vector using the DD matrix
         * @pre DD is set, oldUVW is set
         */
        void CalcUVW();

        const Eigen::MatrixXcd& GetAvgData() const { return m_avgData; }
        Eigen::MatrixXcd& GetAvgData() { return m_avgData; }

        bool operator==(const MetaData& rhs) const;
        bool operator!=(const MetaData& rhs) const { return !(*this == rhs); }

        friend class icrar::cuda::DeviceMetaData;
        friend class icrar::cuda::ConstantBuffer;
    };
} // namespace cpu
} // namespace icrar
