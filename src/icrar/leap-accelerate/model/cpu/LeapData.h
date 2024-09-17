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

#pragma once

#if CUDA_ENABLED
#include <cuda_runtime.h>
#else
#ifndef __host__
// Ignore __host__ keyword on non-cuda compilers
#define __host__
#endif // __host__
#ifndef __device__
// Ignore __device__ keyword on non-cuda compilers
#define __device__
#endif // __device__
#endif // CUDA_ENABLED

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>
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
        class DeviceLeapData;
        class ConstantBuffer;
    } // namespace cuda
} // namespace icrar

namespace icrar
{
namespace cpu
{
    /**
     * @brief Container of variables that do not change throughout calibration
     */
    struct Constants
    {
        uint32_t nbaselines; //the total number station pairs (excluding self cycles) 

        uint32_t referenceAntenna;

        uint32_t channels; // The number of channels of the current observation
        uint32_t num_pols; // The number of polarizations used by the current observation
        uint32_t stations; // The number of stations used by the current observation
        uint32_t timesteps; // The number of visibility timesteps 
        uint32_t rows;

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
     * @brief container of phaserotation constants and variables for calibrating a single beam.
     * Can be mutated to calibrate for multiple directions.
     */
    class LeapData
    {
        LeapData() = default;

    protected:
        Constants m_constants;
        double m_minimumBaselineThreshold;
        bool m_useCache;

        Eigen::MatrixXd m_A;
        Eigen::VectorXi m_I; // The flagged indexes of A
        Eigen::MatrixXd m_A1;
        Eigen::VectorXi m_I1;
        Eigen::MatrixXd m_Ad; ///< The pseudo-inverse of m_A, late intitialized
        Eigen::MatrixXd m_Ad1; //< The pseudo-inverse of m_Ad1, late intitialized

        std::vector<icrar::MVuvw> m_UVW;

        SphericalDirection m_direction; // calibration direction, late initialized
        Eigen::Matrix3d m_dd; // direction dependant matrix, late initialized
        Eigen::VectorXcd m_avgData; // matrix of size (baselines), late initialized
    
    public:
        /**
         * @brief Construct a new LeapData object. SetDirection() must be called after construction
         * 
         * @param ms measurement set to read observations from
         * @param refAnt the reference antenna index, default is the last index
         * @param minimumBaselineThreshold baseline lengths less that the minimum in meters are flagged
         * @param computeInverse whether to compute inverse using cpu inversion
         * @param useCache whether to load Ad matrix from cache
         */
        LeapData(
            const icrar::MeasurementSet& ms,
            boost::optional<unsigned int> refAnt = boost::none,
            double minimumBaselineThreshold = 0.0,
            bool computeInverse = true,
            bool useCache = true);

        /**
         * @brief Construct a new LeapData object.
         * 
         * @param ms measurement set to read observations from
         * @param direction the direction of the beam to calibrate for
         * @param refAnt the reference antenna index, default is the last index
         * @param minimumBaselineThreshold baseline lengths less that the minimum in meters are flagged
         * @param useCache whether to load Ad matrix from cache
         */
        LeapData(
            const icrar::MeasurementSet& ms,
            const SphericalDirection& direction,
            boost::optional<unsigned int> refAnt = boost::none,
            double minimumBaselineThreshold = 0.0,
            bool computeInverse = true,
            bool useCache = true);

        const Constants& GetConstants() const;

        /**
         * @brief Matrix of baseline pairs of shape [baselines, stations] 
         */
        const Eigen::MatrixXd& GetA() const;

        /**
         * @brief Vector of indexes of the stations that are not flagged in A of shape [baselines]
         */
        const Eigen::VectorXi& GetI() const;

        /**
         * @brief The pseudoinverse of A with shape [stations, baselines]
         */
        const Eigen::MatrixXd& GetAd() const;
        virtual void SetAd(Eigen::MatrixXd&& Ad) { m_Ad = std::move(Ad); }

        /**
         * @brief Matrix of baselines using the reference antenna of shape [stations+1, stations]
         * where the last row represents the reference antenna
         */
        const Eigen::MatrixXd& GetA1() const;

        /**
         * @brief Vector of indexes of the stations that are not flagged in A1 of shape [stations]
         * 
         * @return const Eigen::VectorXi& 
         */
        const Eigen::VectorXi& GetI1() const;

        const Eigen::MatrixXd& GetAd1() const;
        virtual void SetAd1(Eigen::MatrixXd&& ad1) { m_Ad1 = std::move(ad1); }

        const SphericalDirection& GetDirection() const { return m_direction; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }
        void SetDirection(const SphericalDirection& direction);


        /**
         * @brief Computes the A and A1 inverse matrices 
         * 
         */
        void ComputeInverse();

        /**
         * @brief Output logs on the validity of inverse matrices
         * 
         */
        void ValidateInverse() const;

        /**
         * @brief Utility method to generate a direction matrix using the
         * configured zenith direction
         * 
         * @param direction 
         * @return Eigen::Matrix3d 
         */
        Eigen::Matrix3d GenerateDDMatrix(const SphericalDirection& direction) const;

        const Eigen::VectorXcd& GetAvgData() const { return m_avgData; }
        Eigen::VectorXcd& GetAvgData() { return m_avgData; }

        bool operator==(const LeapData& rhs) const;
        bool operator!=(const LeapData& rhs) const { return !(*this == rhs); }

        friend class icrar::cuda::DeviceLeapData;
        friend class icrar::cuda::ConstantBuffer;
    };
} // namespace cpu
} // namespace icrar
