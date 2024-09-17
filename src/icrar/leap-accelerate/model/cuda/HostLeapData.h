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

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/model/cpu/LeapData.h>
#include <cuda_runtime.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A cuda decorator for cpu::Integration. This class stores data on the host withs pinned memory
     * calls to allow for asyncronous read and write with cuda.
     */
    class HostLeapData : public cpu::LeapData
    {
    public:
        HostLeapData(
            const icrar::MeasurementSet& ms,
            boost::optional<unsigned int> refAnt,
            double minimumBaselineThreshold,
            bool computeInverse,
            bool useCache)
        : LeapData(ms, refAnt, minimumBaselineThreshold, computeInverse, useCache)
        {
            cudaHostRegister(m_A.data(), m_A.size() * sizeof(decltype(*m_A.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_I.data(), m_I.size() * sizeof(decltype(*m_I.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_A1.data(), m_A1.size() * sizeof(decltype(*m_A1.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_I1.data(), m_I1.size() * sizeof(decltype(*m_I1.data())), cudaHostRegisterPortable);
            if(m_Ad.size() != 0)
            {
                cudaHostRegister(m_Ad.data(), m_Ad.size() * sizeof(decltype(*m_Ad.data())), cudaHostRegisterPortable);
            }
            if(m_Ad1.size() != 0)
            {
                cudaHostRegister(m_Ad1.data(), m_Ad1.size() * sizeof(decltype(*m_Ad1.data())), cudaHostRegisterPortable);
            }
        }

        ~HostLeapData()
        {
            cudaHostUnregister(m_A.data());
            cudaHostUnregister(m_I.data());
            cudaHostUnregister(m_A1.data());
            cudaHostUnregister(m_I1.data());
            cudaHostUnregister(m_Ad.data());
            cudaHostUnregister(m_Ad1.data());
        }

        void SetAd(Eigen::MatrixXd&& Ad) override
        {
            if(m_Ad.size() != 0)
            {
                cudaHostUnregister(m_Ad.data());
            }
            m_Ad = std::move(Ad);
            cudaHostRegister(m_Ad.data(), m_Ad.size() * sizeof(decltype(*m_Ad.data())), cudaHostRegisterPortable);
        }

        void SetAd1(Eigen::MatrixXd&& Ad1) override
        {
            if(m_Ad1.size() != 0)
            {
                cudaHostUnregister(m_Ad1.data());
            }
            m_Ad1 = std::move(Ad1);
            cudaHostRegister(m_Ad1.data(), m_Ad1.size() * sizeof(decltype(*m_Ad1.data())), cudaHostRegisterPortable);
        }
    };
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
