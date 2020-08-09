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

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <casacore/casa/Arrays/Array.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/Eigen/Core>

#include <vector>
#include <array>

// C++ Style interface (templates not supported)

namespace icrar
{
namespace cuda
{
    void add(size_t n, const double* a, const double* b, double* c);
    void add(size_t n, const float* a, const float* b, float* c);
    void add(size_t n, const int* a, const int* b, int* c);

    void add(const device_vector<double>& a, const device_vector<double>& b, device_vector<double>& c);
    void add(const device_vector<float>& a, const device_vector<float>& b, device_vector<float>& c);
    void add(const device_vector<int>& a, const device_vector<int>& b, device_vector<int>& c);

    void add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c);
    void add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
    void add(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c);

    void add(const Eigen::VectorXd& a, const Eigen::VectorXd& b, Eigen::VectorXd& c);
    void add(const Eigen::VectorXf& a, const Eigen::VectorXf& b, Eigen::VectorXf& c);
    void add(const Eigen::VectorXi& a, const Eigen::VectorXi& b, Eigen::VectorXi& c);

    void add(const casacore::Array<double>& a, const casacore::Array<double>& b, casacore::Array<double>& c);
    void add(const casacore::Array<float>& a, const casacore::Array<float>& b, casacore::Array<float>& c);
    void add(const casacore::Array<int>& a, const casacore::Array<int>& b, casacore::Array<int>& c);

}
}