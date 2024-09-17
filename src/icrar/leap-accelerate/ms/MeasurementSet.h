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

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/common/Slice.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/optional.hpp>

#include <iterator>
#include <string>
#include <exception>
#include <memory>
#include <tuple>
#include <vector>

namespace icrar
{
    /**
     * @brief Provides an abstraction layer around a casacore MeasurementSet that provides all data
     * required for leap calibration. This class additionally stores runtime user
     * specificed variables and cached variabled calculated from the underlying measurement set.
     */
    class MeasurementSet
    {
        std::unique_ptr<casacore::MeasurementSet> m_measurementSet;
        std::unique_ptr<casacore::MSMainColumns> m_msmc;
        std::unique_ptr<casacore::MSColumns> m_msc;

        boost::optional<std::string> m_filepath;
        std::set<std::int32_t> m_antennas;
        int m_stations;
        bool m_readAutocorrelations;
        uint32_t m_numBaselines;
        uint32_t m_numRows;
        uint32_t m_numTimesteps;
        double m_timeInterval;
        uint32_t m_numPols;

    public:
        MeasurementSet(const std::string& filepath);

        boost::optional<std::string> GetFilepath() const { return m_filepath; }
        
        bool IsAutoCorrelationsEnabled() const { return m_readAutocorrelations; }

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

        /**
         * @brief Gets the total number of antennas including flagged antennas.
         * 
         */
        uint32_t GetTotalAntennas() const;

        /**
         * @brief Gets the number of stations excluding flagged stations.
         * 
         * @return uint32_t 
         */
        uint32_t GetNumStations() const;

        /**
         * @brief Get the number of baselines in the measurement set using the current autocorrelations setting
         * and including stations not recording rows.
         * @note TODO: baselines should always be n*(n-1) / 2 and without autocorrelations
         * @return uint32_t 
         */
        uint32_t GetNumBaselines() const;

        /**
         * @brief Get the number of polarizations in the measurement set
         * 
         * @return uint32_t 
         */
        uint32_t GetNumPols() const;

        /**
         * @brief Gets the number of channels in the measurement set
         * 
         * @return uint32_t 
         */
        uint32_t GetNumChannels() const;

        /**
         * @brief Gets the number of rows in the measurement set (non-flagged baselines * timesteps).
         * 
         * @return uint32_t
         */
        uint32_t GetNumRows() const;

        /**
         * @brief Gets the indexes of the first antenna in baselines
         * 
         * @return Eigen::VectorXi
         */
        Eigen::VectorXi GetAntenna1() const;

        /**
         * @brief Gets the indexes of the second antenna in baselines
         * 
         * @return Eigen::VectorXi
         */
        Eigen::VectorXi GetAntenna2() const;

        /**
         * @brief Gets the total number of timesteps in the measurement set
         * 
         * @return uint32_t 
         */
        uint32_t GetNumTimesteps() const;

        /**
         * @brief Gets the time interval of visibilities (assuming the same for all rows)
        */
        double GetTimeInterval() const;

        /**
         * @brief Get the Epochs object
         * 
         * @return std::vector<double> 
         */
        std::vector<double> GetEpochs() const;

        /**
         * @brief Gets a vector of size nBaselines with a true value at the index of unflagged baselines.
         * Flag is is logical and of channels and polarizations.
         * 
         * @return Eigen::VectorXb 
         */
        Eigen::VectorXb GetFlaggedBaselines() const;

        /**
         * @brief Get the number of baselines that are flagged by the measurement set
         * 
         * @return uint32_t 
         */
        uint32_t GetNumFlaggedBaselines() const;

        /**
         * @brief Gets a flag vector of short baselines
         * 
         * @param minimumBaselineThreshold baseline threshold
         * @return Eigen::VectorXb
         */
        Eigen::VectorXb GetShortBaselines(double minimumBaselineThreshold = 0.0) const;

        /**
         * @brief Get the number of baselines that below the @p minimumBaselineThreshold
         * 
         * @param minimumBaselineThreshold baseline threshold
         * @return uint32_t 
         */
        uint32_t GetNumShortBaselines(double minimumBaselineThreshold = 0.0) const;

        /**
         * @brief Gets flag vector of filtered baselines that are either flagged or short
         * 
         * @param minimumBaselineThreshold baseline threshold
         * @return Eigen::VectorXb 
         */
        Eigen::VectorXb GetFilteredBaselines(double minimumBaselineThreshold = 0.0) const;

        /**
         * @brief Gets the number of baselines that are flagged baselines or short baselines
         * 
         * @param minimumBaselineThreshold baseline threshold
         * @return uint32_t 
         */
        uint32_t GetNumFilteredBaselines(double minimumBaselineThreshold = 0.0) const;

        /**
         * @brief Reads UVW coordinates from the measurement set
         * 
         * @return Eigen::Tensor<double, 3> 
         */
        Eigen::Tensor<double, 3> ReadCoords() const;

        /**
         * @brief Gets the Coords/UVWs of a specified time interval
         * 
         * @param startTimestep 
         * @param intervalTimesteps 
         * @return Eigen::Tensor<double, 3> of dimensions (3, baselines, timesteps)
         */
        Eigen::Tensor<double, 3> ReadCoords(
            const Slice& timestepSlice) const;

        /**
         * @brief Gets the Coords/UVWs of a specified time interval
         * 
         * @param startTimestep 
         * @param intervalTimesteps 
         * @return Eigen::Tensor<double, 3> of dimensions (3, baselines, timesteps)
         */
        Eigen::Tensor<double, 3> ReadCoords(
            uint32_t startTimestep,
            uint32_t intervalTimesteps) const;

        /**
         * @brief Gets the visibilities from all baselines, channels and polarizations
         * for the first timestep
         * 
         * @return Eigen::Tensor<std::complex<double>, 4> of dimensions (polarizations, channels, baselines, timesteps)
         */
        Eigen::Tensor<std::complex<double>, 4> ReadVis() const;

        /**
         * @brief Gets visibilities from the specificed dimension slices
         * of a specified timestep slice
         * 
         * @param startTimestep start timestep index
         * @param intervalTimesteps number of timesteps
         * @return Eigen::Tensor<std::complex<double>, 4> of dimensions (polarizations, channels, baselines, timesteps)
         */
        Eigen::Tensor<std::complex<double>, 4> ReadVis(
            const Slice& timestepSlice,
            const Slice& polarizationSlice = Slice(0, boost::none, 1)) const;

        /**
         * @brief Gets visibilities from the specificed dimension slices
         * of a specified timestep slice
         * 
         * @param startTimestep start timestep index
         * @param intervalTimesteps number of timesteps
         * @return Eigen::Tensor<std::complex<double>, 4> of dimensions (polarizations, channels, baselines, timesteps)
         */
        Eigen::Tensor<std::complex<double>, 4> ReadVis(
            std::uint32_t startTimestep,
            std::uint32_t intervalTimesteps,
            const Slice& polarizationSlice = Slice(0, boost::none, 1)) const;

        /**
         * @brief Reads from file visibilities using specified dimension slices
         * 
         * @param startTimestep 
         * @param intervalTimesteps 
         * @param polarizationRange 
         * @param column 
         * @return Eigen::Tensor<std::complex<double>, 4> of dimensions (polarizations, channels, baselines, timesteps)
         */
        Eigen::Tensor<std::complex<double>, 4> ReadVis(
            uint32_t startTimestep,
            uint32_t intervalTimesteps,
            Range<int32_t> polarizationRange,
            const char* column) const;

        /**
         * @brief Gets the antennas that are not present in any baselines
         * 
         * @return std::set<int32_t>
         */
        std::set<int32_t> GetMissingAntennas() const;

        /**
         * @brief Gets the antenna indexes that are either not present in any baselines
         * or are flagged in all of it's baselines. 
         * 
         * Indexes are out of the total antennas 
         * 
         * @return std::set<int32_t> 
         */
        std::set<int32_t> GetFlaggedAntennas() const;

    private:

        void Validate() const;

        /**
         * @brief Calculates the number of baselines in the measurement set (e.g. (0,0), (1,1), (2,2))
         * 
         * @return uint32_t 
         */
        uint32_t CalculateNumBaselines(bool useAutocorrelations) const;

        /**
         * @brief Calculates the set of unique antenna present in baselines
         * 
         * @return uint32_t 
         */
        std::tuple<std::set<int32_t>, bool> CalculateUniqueAntennas() const;

    };
} // namespace icrar
