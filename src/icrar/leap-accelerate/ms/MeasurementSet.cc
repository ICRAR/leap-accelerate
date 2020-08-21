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

#include "MeasurementSet.h"
#include <icrar/leap-accelerate/ms/utils.h>

namespace icrar
{
    MeasurementSet::MeasurementSet(std::string filepath)
    {
        m_measurementSet = std::make_unique<casacore::MeasurementSet>(filepath);
        m_msmc = std::make_unique<casacore::MSMainColumns>(*m_measurementSet);
        m_msc = std::make_unique<casacore::MSColumns>(*m_measurementSet);
    }

    MeasurementSet::MeasurementSet(const casacore::MeasurementSet& ms)
    {
        m_measurementSet = std::make_unique<casacore::MeasurementSet>(ms);
        m_msmc = std::make_unique<casacore::MSMainColumns>(*m_measurementSet);
        m_msc = std::make_unique<casacore::MSColumns>(*m_measurementSet);
    }

    unsigned int MeasurementSet::GetNumStations() const
    {
        return m_measurementSet->antenna().nrow();
    }

    unsigned int MeasurementSet::GetNumPols() const
    {
        if(m_measurementSet->polarization().nrow() > 0)
        {
            return m_msc->polarization().numCorr().get(0);
        }
        else
        {
            throw icrar::not_implemented_exception(__FILE__, __LINE__);
        }
        
    }

    unsigned int MeasurementSet::GetNumBaselines() const
    {
        const size_t num_stations = (size_t)GetNumStations();
        return num_stations * (num_stations + 1) / 2; //TODO: +/- 1???
    }

    unsigned int MeasurementSet::GetNumChannels() const
    {
        return m_msc->spectralWindow().numChan().get(0);
    }

    Eigen::MatrixX3d MeasurementSet::GetCoords(unsigned int start_row) const
    {
        auto num_baselines = GetNumBaselines();
        Eigen::MatrixX3d matrix = Eigen::MatrixX3d::Zero(num_baselines, 3);
        icrar::ms_read_coords(
            *m_measurementSet,
            start_row,
            num_baselines,
            matrix(Eigen::all, 0).data(),
            matrix(Eigen::all, 1).data(),
            matrix(Eigen::all, 2).data());
        return matrix;
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis() const
    {
        auto num_channels = GetNumChannels();
        auto num_baselines = GetNumBaselines();
        auto num_pols = GetNumPols();
        auto visibilities = Eigen::Tensor<std::complex<double>, 3>(num_channels, num_baselines, num_pols);
        icrar::ms_read_vis(*m_measurementSet, 0, 0, num_channels, num_baselines, num_pols, "DATA", (double*)visibilities.data());
        //TODO: implement
        return visibilities;
    }
}