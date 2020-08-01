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

#include "test_helper.h"

template<typename T>
void assert_meq(const Eigen::Matrix<T, -1, -1>& expected, const Eigen::Matrix<T, -1, -1>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Matrix elements differ at:\n";
        
        for(int col = 0; col < actual.cols(); ++col)
        {
            for(int row = 0; row < actual.rows(); ++row)
            {
                if(abs(expected(row, col) - actual(row, col)) > tolerance)
                {
                    std::cerr << "expected(" << row << ", " << col << ") == " << expected(row, col) << "\n";
                    std::cerr << "actual(" << row << ", " << col << ") == " << actual(row, col) << "\n";
                }
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_meqi(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
{
    assert_meq<int>(expected, actual, tolerance, ln, rn, file, line);
}

void assert_meqd(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
{
    assert_meq<double>(expected, actual, tolerance, ln, rn, file, line);
}

void assert_meqd(const Eigen::Matrix3d& expected, const Eigen::Matrix3d& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Matrix elements differ at:\n";
        
        for(int col = 0; col < actual.cols(); ++col)
        {
            for(int row = 0; row < actual.rows(); ++row)
            {
                if(abs(expected(row, col) - actual(row, col)) > tolerance)
                {
                    std::cerr << "expected(" << row << ", " << col << ") == " << expected(row, col) << "\n";
                    std::cerr << "actual(" << row << ", " << col << ") == " << actual(row, col) << "\n";
                }
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_meqcd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
{
    assert_meq<std::complex<double>>(expected, actual, tolerance, ln, rn, file, line);
}

template<typename T>
void assert_veq(const Eigen::Matrix<T, -1, 1>& expected, const Eigen::Matrix<T, -1, 1>& actual, double tolerance, std::string file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << file << ":" << line << " Vector elements differ at:\n";
        
        for(int row = 0; row < actual.rows(); ++row)
        {
            if(abs(expected(row) - actual(row)) > tolerance)
            {
                std::cerr << "expected(" << row << ") == " << expected(row) << "\n";
                std::cerr << "actual(" << row << ") == " << actual(row) << "\n";
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_veqi(const Eigen::VectorXi& expected, const Eigen::VectorXi& actual, double tolerance, std::string file, int line)
{
    assert_veq<int>(expected, actual, tolerance, file, line);
}

void assert_veqd(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual, double tolerance, std::string file, int line)
{
    assert_veq<double>(expected, actual, tolerance, file, line);
}

void assert_metadataeq(const icrar::cuda::MetaDataCudaHost& expected, const icrar::cuda::MetaDataCudaHost& actual, std::string file, int line)
{
    std::cerr << "checking MetaData at " << file << ":" << line << std::endl;
    const double THRESHOLD = 0.001;
    //ASSERT_EQ(expectedIntegration.baselines, metadataOutput.avg_data.rows());
    
    ASSERT_EQ(expected.IsInitialized(), actual.IsInitialized());
    ASSERT_EQ(expected.GetConstants().nantennas, actual.GetConstants().nantennas);
    //ASSERT_EQ(expected.nbaseline, metadata.nbaseline);
    ASSERT_EQ(expected.GetConstants().channels, actual.GetConstants().channels);
    ASSERT_EQ(expected.GetConstants().num_pols, actual.GetConstants().num_pols);
    ASSERT_EQ(expected.GetConstants().stations, actual.GetConstants().stations);
    ASSERT_EQ(expected.GetConstants().rows, actual.GetConstants().rows);
    ASSERT_EQ(expected.GetConstants().freq_start_hz, actual.GetConstants().freq_start_hz);
    ASSERT_EQ(expected.GetConstants().freq_inc_hz, actual.GetConstants().freq_inc_hz);
    ASSERT_EQ(expected.GetConstants().solution_interval, actual.GetConstants().solution_interval);
    ASSERT_EQ(expected.GetConstants().channel_wavelength, actual.GetConstants().channel_wavelength);
    ASSERT_EQ(expected.GetConstants().phase_centre_ra_rad, actual.GetConstants().phase_centre_ra_rad);
    ASSERT_EQ(expected.GetConstants().phase_centre_dec_rad, actual.GetConstants().phase_centre_dec_rad);
    ASSERT_EQ(expected.GetConstants().dlm_ra, actual.GetConstants().dlm_ra);
    ASSERT_EQ(expected.GetConstants().dlm_dec, actual.GetConstants().dlm_dec);
    //ASSERT_EQ(expected.oldUVW, metadata.oldUVW); //TODO

    ASSERT_EQ(expected.avg_data.is_initialized(), actual.avg_data.is_initialized());
    ASSERT_MEQCD(expected.avg_data.get(), actual.avg_data.get(), THRESHOLD);

    ASSERT_EQ(expected.dd.is_initialized(), actual.dd.is_initialized());
    ASSERT_MEQ3D(expected.dd.get(), actual.dd.get(), THRESHOLD);
    ASSERT_MEQ(expected.A, actual.A, THRESHOLD);
    ASSERT_MEQI(expected.I, actual.I, THRESHOLD);
    ASSERT_MEQ(expected.Ad, actual.Ad, THRESHOLD);
    ASSERT_MEQ(expected.A1, actual.A1, THRESHOLD);
    ASSERT_MEQI(expected.I1, actual.I1, THRESHOLD);
    ASSERT_MEQ(expected.Ad1, actual.Ad1, THRESHOLD);
    
    //ASSERT_EQ(expecexpectedtedMetadata, metadata);
    //ASSERT_EQ(expectedIntegration, integration);
}