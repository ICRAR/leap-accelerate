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

#include <casacore/casa/Arrays/Matrix.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>
#include <limits>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Calculates the PseudoInverse matrix of size N * M for a given M * N matrix.
     * Satisfies the equation A = A * Ah * A
     * @param A 
     * @param epsilon 
     * @return Matrix_T
     */
    template<typename Matrix_T>
    Matrix_T SVDPseudoInverse(const Matrix_T& a, double epsilon = std::numeric_limits<typename Matrix_T::Scalar>::epsilon())
    {
        // See https://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
        Eigen::BDCSVD<Matrix_T> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
        return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    }

    /**
     * @brief Invert as a function
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * 
     * This function generates and returns the inverse of the linear matrix to solve for the phase calibration (only)
     * given a MS.
     * The MS is used to fill the columns of the matrix, based on the baselines in the MS (and RefAnt if given)
     * 
     * The output will be the inverse matrix to cross with the observation vector.
     * 
     * @param A
     */
    template<typename Matrix_T>
    Matrix_T PseudoInverse(const Matrix_T& a)
    {
        #if EIGEN_VERSION_AT_LEAST(3,3,0)
        return a.completeOrthogonalDecomposition().pseudoInverse();
        #else
        return SVDPseudoInverse(a);
        #endif
    }

    /**
     * @see PseudoInverse
     */
    casacore::Matrix<double> PseudoInverse(const casacore::Matrix<double>& a);

    /**
     * @see SVDPseudoInverse
     */
    casacore::Matrix<double> SVDPseudoInverse(const casacore::Matrix<double>& a,  double epsilon = std::numeric_limits<Eigen::MatrixXd::Scalar>::epsilon());
}
}