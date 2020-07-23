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

#include "Invert.h"

#include <icrar/leap-accelerate/math/eigen_helper.h>

#include <casacore/casa/Arrays/Matrix.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>
#include <queue>

namespace icrar
{
namespace cpu
{
    Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& A)
    {
		#if EIGEN_VERSION_AT_LEAST(3,3,0)
        return A.completeOrthogonalDecomposition().pseudoInverse();
        #else
		throw std::runtime_error("completeOrthogonalDecomposition missing, use SVDPseudoInverse instead");
		#endif
    }

    casacore::Matrix<double> PseudoInverse(const casacore::Matrix<double>& a)
    {
        return ConvertMatrix(PseudoInverse(ConvertMatrix(a)));
    }

    Eigen::MatrixXd SVDPseudoInverse(const Eigen::MatrixXd& a, double epsilon)
    {
        // See https://eigen.tuxfamily.org/bz/show_bug.cgi?id=257
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
        return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    }

    casacore::Matrix<double> SVDPseudoInverse(const casacore::Matrix<double>& a,  double epsilon)
    {
        return ConvertMatrix(SVDPseudoInverse(ConvertMatrix(a), epsilon));
    }
}
}
