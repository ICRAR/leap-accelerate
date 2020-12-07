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
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/common/eigen_extensions.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <cstddef>

namespace icrar
{
    MeasurementSet::MeasurementSet(const std::string& filepath, boost::optional<int> overrideNStations, bool readAutocorrelations)
    : m_measurementSet(std::make_unique<casacore::MeasurementSet>(filepath))
    , m_msmc(std::make_unique<casacore::MSMainColumns>(*m_measurementSet))
    , m_msc(std::make_unique<casacore::MSColumns>(*m_measurementSet))
    , m_filepath(filepath)
    , m_readAutocorrelations(readAutocorrelations)
    {
        // Check and use unique antennas 
        m_antennas = CalculateUniqueAntennas();

        if(overrideNStations.is_initialized())
        {
            m_stations = overrideNStations.get();
            LOG(warning) << "overriding number of stations will be removed in future releases";
        }
        else if(m_antennas.size() != m_measurementSet->antenna().nrow())
        {
            LOG(warning) << "ms antennas = " << m_measurementSet->antenna().nrow();
            LOG(warning) << "unique antennas = " << m_antennas.size();
            LOG(warning) << "using unique antennas";
            m_stations = m_antennas.size();
        }
        else
        {
            m_stations = m_measurementSet->antenna().nrow();
        }

        Validate();
    }

    void MeasurementSet::Validate() const
    {
        // Stations
        if(m_antennas.size() != GetNumStations())
        {
            LOG(error) << "unique antennas does not match number of stations";
            LOG(error) << "unique antennas: " << m_antennas.size();
            LOG(error) << "stations: " << GetNumStations();
        }

        // Baselines
        // Validate number of baselines in first epoch
        casacore::Vector<double> time = m_msmc->time().getColumn();
        auto epoch = time[0];
        auto epochRows = std::count(time.begin(), time.end(), epoch);

        if(epochRows != GetNumBaselines())
        {
            LOG(error) << "epoch rows does not match baselines";
            LOG(error) << "epoch rows: " << epochRows;
            LOG(error) << "baselines: " << GetNumBaselines();
            throw exception("epoch size doesnt match number of baselines", __FILE__, __LINE__);
        }

        if(GetNumRows() < GetNumBaselines())
        {
            std::stringstream ss;
            ss << "invalid number of rows, expected >=" << GetNumBaselines() << ", got " << GetNumRows();
            throw icrar::file_exception(GetFilepath().get_value_or("unknown"), ss.str(), __FILE__, __LINE__);
        }

        if(GetNumBaselines() == 0 || (GetNumRows() % GetNumBaselines() != 0))
        {
            LOG(error) << "number of rows not an integer multiple of number of baselines";
            LOG(error) << "baselines: " << GetNumBaselines()
                         << " rows: " << GetNumRows()
                         << "total epochs ~= " << static_cast<double>(GetNumRows()) / GetNumBaselines();
            throw exception("number of rows not an integer multiple of baselines", __FILE__, __LINE__);
        }
    }

    unsigned int MeasurementSet::GetNumRows() const
    {
        return m_msmc->uvw().nrow();
    }

    unsigned int MeasurementSet::GetNumStations() const
    {
        return m_stations;
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
        return GetNumBaselines(m_readAutocorrelations);
    }

    unsigned int MeasurementSet::GetNumBaselines(bool useAutocorrelations) const
    {
        //TODO(calgray): cache value
        if(useAutocorrelations)
        {
            const auto num_stations = GetNumStations();
            return num_stations * (num_stations + 1) / 2;
        }
        else
        {
            const auto num_stations = GetNumStations();
            return num_stations * (num_stations - 1) / 2;
        }
    }

    unsigned int MeasurementSet::GetNumChannels() const
    {
        if(m_msc->spectralWindow().nrow() > 0)
        {
            return m_msc->spectralWindow().numChan().get(0);
        }
        else
        {
            return 0;
        }
    }

    Eigen::Matrix<bool, -1, 1> MeasurementSet::GetFlaggedBaselines() const
    {
        // TODO(calgray): may want to consider using logical OR over for each channel and polarization.
        auto nBaselines = GetNumBaselines();

        // Selects the flags of the first epoch
        auto epochIndices = casacore::Slicer(casacore::Slice(0, nBaselines));
        
        // Selects only the flags of the first channel and polarization
        auto flagSlice = casacore::Slicer(
            casacore::IPosition(2, 0, 0),
            casacore::IPosition(2, 1, 1),
            casacore::IPosition(2, 1, 1));
        casacore::Vector<bool> flags = m_msmc->flag().getColumnRange(epochIndices, flagSlice);
        return ToVector(flags);
    }

    unsigned int MeasurementSet::GetNumFlaggedBaselines() const
    {
        auto flaggedBaselines = GetFlaggedBaselines();
        return std::count(flaggedBaselines.cbegin(), flaggedBaselines.cend(), true);
    }
	
	Eigen::Matrix<bool, -1, 1> MeasurementSet::GetShortBaselines(double minimumBaselineThreshold) const
    {
        auto nBaselines = GetNumBaselines();
        Eigen::Matrix<bool, -1, 1> baselineFlags = Eigen::Matrix<bool, -1, 1>::Zero(nBaselines); 

        // Filter short baselines
        if(minimumBaselineThreshold > 0.0)
        {
            auto firstChannelSlicer = casacore::Slicer(casacore::Slice(0, 1));
            casacore::Matrix<double> uv = m_msmc->uvw().getColumn(firstChannelSlicer);

            // TODO(calgray): uv is of size baselines * timesteps, consider throwing a warning if flags change
            //  in later timesteps
            for(unsigned int i = 0; i < nBaselines; i++)
            {
                if(std::sqrt(uv(i, 0) * uv(i, 0) + uv(i, 1) * uv(i, 1)) < minimumBaselineThreshold)
                {
                    baselineFlags(i) = true;
                }
            }
        }

        return baselineFlags;
    }

    unsigned int MeasurementSet::GetNumShortBaselines(double minimumBaselineThreshold) const
    {
        auto shortBaselines = GetShortBaselines(minimumBaselineThreshold);
        return std::count(shortBaselines.cbegin(), shortBaselines.cend(), true);
    }

    Eigen::Matrix<bool, -1, 1> MeasurementSet::GetFilteredBaselines(double minimumBaselineThreshold) const
    {
        Eigen::Matrix<bool, -1, 1> result = GetFlaggedBaselines() || GetShortBaselines(minimumBaselineThreshold);
        return result;
    }

    unsigned int MeasurementSet::GetNumFilteredBaselines(double minimumBaselineThreshold) const
    {
        auto filteredBaselines = GetFilteredBaselines(minimumBaselineThreshold);
        return std::count(filteredBaselines.cbegin(), filteredBaselines.cend(), true);
	}

    Eigen::MatrixX3d MeasurementSet::GetCoords() const
    {
        return GetCoords(0, GetNumBaselines());
    }

    Eigen::MatrixX3d MeasurementSet::GetCoords(unsigned int start_row, unsigned int nBaselines) const
    {
        Eigen::MatrixX3d matrix = Eigen::MatrixX3d::Zero(nBaselines, 3);
        icrar::ms_read_coords(
            *m_measurementSet,
            start_row,
            nBaselines,
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
        return GetVis(0, 0, num_channels, num_baselines, num_pols);
    }

    Eigen::Tensor<std::complex<double>, 3> MeasurementSet::GetVis(
        std::uint32_t startBaseline,
        std::uint32_t startChannel,
        std::uint32_t nChannels,
        std::uint32_t nBaselines,
        std::uint32_t nPolarizations) const
    {
        auto visibilities = Eigen::Tensor<std::complex<double>, 3>(nPolarizations, nBaselines, nChannels);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        icrar::ms_read_vis(*m_measurementSet, startBaseline, startChannel, nChannels, nBaselines, nPolarizations, "DATA", reinterpret_cast<double*>(visibilities.data()));
        return visibilities;
    }

    std::set<int32_t> MeasurementSet::CalculateUniqueAntennas() const
    {
        casacore::Vector<casacore::Int> a1 = m_msmc->antenna1().getColumn();
        casacore::Vector<casacore::Int> a2 = m_msmc->antenna1().getColumn();
        auto a1s = std::set<int32_t>(a1.cbegin(), a1.cend());
        auto a2s = std::set<int32_t>(a2.cbegin(), a2.cend());
        std::set<std::int32_t> antennas;
        std::set_union(a1.cbegin(), a1.cend(), a2.cbegin(), a2.cend(), std::inserter(antennas, antennas.begin()));
        return antennas; 
    }
} // namespace icrar