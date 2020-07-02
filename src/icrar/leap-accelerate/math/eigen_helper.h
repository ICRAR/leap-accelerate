
#pragma once
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <eigen3/Eigen/Core>

Eigen::MatrixXd ConvertMatrix(casacore::Matrix<double> value)
{
    auto shape = value.shape();
    auto m = Eigen::MatrixXd(shape[0], shape[1]);

    auto it = value.begin();
    for(int row = 0; row < shape[0]; ++row)
    {
        for(int col = 0; col < shape[1]; ++col)
        {
            m(row, col) = *it;
            it++;
        }
    }
    return m;
}

casacore::Matrix<double> ConvertMatrix(Eigen::MatrixXd value)
{
    Eigen::MatrixXd m(value.rows(), value.cols());
    for(int row = 0; row < value.rows(); ++row)
    {
        for(int col = 0; col < value.cols(); ++col)
        {
            m(row, col) = 0;
        }
    }
    return casacore::Matrix<double>(casacore::IPosition(value.rows(), value.cols()), m.data());
}

// Eigen::MatrixXcd ConvertMatrix(casacore::CMatrix<double> v)
// {

// }

Eigen::VectorXd ConvertVector(casacore::Array<double> value)
{
    auto v = Eigen::VectorXd(value.size());
    
    
}

// Eigen::VectorXcd ConvertVector(casacore::CArray<double> v)
// {

// }