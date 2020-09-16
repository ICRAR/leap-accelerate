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

#include <icrar/leap-accelerate/exception/exception.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <iterator>
#include <string>
#include <exception>
#include <memory>
#include <tuple>
#include <vector>

namespace icrar
{
    class MeasurementSet
    {
        std::unique_ptr<casacore::MeasurementSet> m_measurementSet;
        std::unique_ptr<casacore::MSColumns> m_msc;
        std::unique_ptr<casacore::MSMainColumns> m_msmc;

    public:
        MeasurementSet(std::string filepath);
        MeasurementSet(const casacore::MeasurementSet& ms);
        MeasurementSet(std::istream& stream);

        /**
         * @brief Gets a non-null pointer to a casacore::MeasurementSet
         * 
         * @return const casacore::MeasurementSet* 
         */
        const casacore::MeasurementSet* GetMS() const { return m_measurementSet.get(); }

        /**
         * @brief Gets a non-null pointer to a casacore::MSMainColumns
         * 
         * @return const casacore::MSMainColumns* 
         */
        const casacore::MSMainColumns* GetMSMainColumns() const { return m_msmc.get(); }
        
        /**
         * @brief Gets a non-null pointer to a casacore::MSColumns
         * 
         * @return const casacore::MSColumns* 
         */
        const casacore::MSColumns* GetMSColumns() const { return m_msc.get(); }

        unsigned int GetNumStations() const;

        unsigned int GetNumBaselines() const;

        unsigned int GetNumPols() const;

        unsigned int GetNumChannels() const;

        //std::vector<casacore::MVuvw> MeasurementSet::GetCoordsCasa(unsigned int start_row) const;
        Eigen::MatrixX3d GetCoords() const;
        Eigen::MatrixX3d GetCoords(unsigned int start_row, unsigned int nBaselines) const;

        Eigen::Tensor<std::complex<double>, 3> GetVis(std::uint32_t nChannels, std::uint32_t nBaselines, std::uint32_t nPolarizations) const;
        Eigen::Tensor<std::complex<double>, 3> GetVis() const;
    };
}
