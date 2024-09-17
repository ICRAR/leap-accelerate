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

#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <Eigen/Core>

#include <boost/optional.hpp>
#include <utility>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Generate Phase Matrix
     * Given the antenna lists from an MS and (optionally) reference antenna and antenna flags:
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * If True Map is provided it returns the index map for the matrix (only useful if RefAnt set).
     *
     * This function generates and returns the linear matrix for the phase calibration (only)
     * @param a1 indexes vector of 1st antenna of each baselines
     * @param a2 indexes vector of 2nd antenna of each baselines
     * @param refAnt the reference antenna index
     * @param fg a flag vector of flagged baselines to ignore when true
     * @param allBaselines whether to generate phase matrix for all baselines or just ones with reference antenna 
     * @return std::pair<Matrixd, Matrixi>
     * for refAnt = none: first matrix is of size [baselines,stations] and seconds of size[baselines,1]
     * for 0 <= refAnt < stations: first matrix is of size [stations,stations] and seconds of size[stations,1]
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXi> PhaseMatrixFunction(
        const Eigen::VectorXi& a1,
        const Eigen::VectorXi& a2,
        const Eigen::VectorXb& fg,
        uint32_t refAnt,
        bool allBaselines);
}
}