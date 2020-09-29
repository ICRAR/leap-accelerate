/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include <icrar/leap-accelerate/model/casa/Integration.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/optional.hpp>

#include <vector>
#include <array>
#include <complex>


namespace icrar
{
namespace cpu
{
    class MeasurementSet;

    class Integration
    {
        std::vector<MVuvw> m_uvw; //uvw is an array uvw[3][nbl] //Eigen::MatrixX3d
        Eigen::Tensor<std::complex<double>, 3> m_data; //data is an array data[nch][nbl][npol]

    public:
        /**
         * @brief Construct a new Integration object from the equivalent casalib object
         */
        Integration(const icrar::casalib::Integration& integration);
        
        Integration(
            unsigned int integrationNumber,
            const icrar::MeasurementSet& ms,
            unsigned int index,
            unsigned int channels,
            unsigned int baselines,
            unsigned int polarizations);


        int integration_number;

        union
        {
            std::array<size_t, 4> parameters; // index, 0, channels, baselines
            struct
            {
                size_t index;
                size_t x;
                size_t channels;
                size_t baselines;
            };
        };

        bool operator==(const Integration& rhs) const;

        const std::vector<icrar::MVuvw>& GetUVW() const;

        [[deprecated("Use GetVis()")]]
        const Eigen::Tensor<std::complex<double>, 3>& GetData() const { return m_data; }

        const Eigen::Tensor<std::complex<double>, 3>& GetVis() const { return m_data; }


        /**
         * @brief Get the Data object of size (polarizations, baselines, channels)
         * 
         * @return Eigen::Tensor<std::complex<double>, 3>& 
         */
        Eigen::Tensor<std::complex<double>, 3>& GetData() { return m_data; }
    };
}
}
