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

#include "svd.h"

#ifdef GSL_ENABLED
#include <gsl/gsl_linalg.h>
#endif

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SVD>

#include <utility>

namespace icrar
{
namespace cpu
{
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd> SVD(const Eigen::MatrixXd& mat)
    {
        auto bdc = Eigen::BDCSVD<Eigen::MatrixXd>(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return std::make_tuple(bdc.matrixU(), bdc.singularValues(), bdc.matrixV());
    }

    std::tuple<Eigen::MatrixXd, Eigen::SparseMatrix<double>, Eigen::MatrixXd> SVDSparse(const Eigen::MatrixXd& mat)
    {
        Eigen::MatrixXd u;
        Eigen::VectorXd s; //sigma
        Eigen::MatrixXd v;
        
        std::tie(u, s, v) = SVD(mat);
        auto sd = Eigen::SparseMatrix<double>(mat.rows(), mat.cols()); //sigma sparse matrix
        for(int i = 0; i < s.size(); i++)
        {
            sd.insert(i,i) = s(i);
        }

        return std::make_tuple(u, sd, v);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVDSigma(const Eigen::MatrixXd& mat)
    {
        Eigen::MatrixXd u;
        Eigen::VectorXd s; //sigma
        Eigen::MatrixXd v;
        
        std::tie(u, s, v) = SVD(mat);
        Eigen::MatrixXd sd = Eigen::MatrixXd::Zero(mat.rows(), mat.cols()); //sigma diagonal matrix
        for(int i = 0; i < s.size(); i++)
        {
            sd(i,i) = s(i);
        }

        return std::make_tuple(u, sd, v);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> SVDSigmaP(const Eigen::MatrixXd& mat)
    {
        Eigen::MatrixXd u;
        Eigen::VectorXd s; //sigma
        Eigen::MatrixXd v;
        
        std::tie(u, s, v) = SVD(mat);
        Eigen::MatrixXd sd = Eigen::MatrixXd::Zero(mat.rows(), mat.cols()); //sigma diagonal matrix
        for(int i = 0; i < s.size(); i++)
        {
            sd(i,i) = 1.0/s(i);
        }

        return std::make_tuple(u, sd.transpose(), v);
    }

#ifdef GSL_ENABLED
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> SVD_gsl(Eigen::MatrixXd& mat)
    {
        Eigen::MatrixXd matU(mat.size(), mat.size());
        gsl_matrix_view mat_view = gsl_matrix_view_array(mat.data(), mat.rows(), mat.cols());
        gsl_matrix_view mat_u = gsl_matrix_view_array(mat.data(), mat.rows(), mat.cols());

        gsl_matrix* mat1 = &mat_view.matrix;
        gsl_matrix* mat2 = &mat_u.matrix;
        gsl_vector* vec1 = gsl_vector_alloc(mat.rows());
        gsl_vector* vec2 = gsl_vector_alloc(mat.cols());

        gsl_linalg_SV_decomp(mat1, mat2, vec1, vec2);

        gsl_vector_free(vec2);
        gsl_vector_free(vec1);

        return std::make_pair(mat, matU);
    }
#endif
}
}