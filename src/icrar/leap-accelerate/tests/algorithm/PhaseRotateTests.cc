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


#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/linear_math_helper.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/cuda/MetaDataCuda.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <gtest/gtest.h>

#include <vector>

using namespace std::literals::complex_literals;

namespace icrar
{
    enum class Impl
    {
        casa,
        eigen,
        cuda
    };

    class PhaseRotateTests : public ::testing::Test
    {
        casacore::MeasurementSet ms;
        MetaData meta;

    protected:

        PhaseRotateTests() {

        }

        ~PhaseRotateTests() override
        {

        }

        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = casacore::MeasurementSet(filename);
            meta = MetaData(ms);
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(Impl impl)
        {
            
            MetaData metadata;
            casacore::MVDirection direction;
            std::queue<Integration> input;
            std::queue<IntegrationResult> output_integrations;
            std::queue<CalibrationResult> output_calibrations;

            if(impl == Impl::casa)
            {
                icrar::casa::PhaseRotate(metadata, direction, input, output_integrations, output_calibrations);
            }
            if(impl == Impl::eigen)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                icrar::cpu::PhaseRotate(metadatahost, direction, input, output_integrations, output_calibrations);
            }
            if(impl == Impl::cuda)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                auto metadatadevice = icrar::cuda::MetaDataCudaDevice(metadatahost);
                icrar::cuda::PhaseRotate(metadatadevice, direction, input, output_integrations, output_calibrations);
            }
            else
            {
                throw std::invalid_argument("impl");
            }
        }

        void RotateVisibilitiesTest(Impl impl)
        {
            const double THRESHOLD = 0.01;

            std::string filepath = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            auto ms = casacore::MeasurementSet(filepath);
            auto metadata = MetaData(ms);
            auto direction = casacore::MVDirection(-0.7129444556035118, -0.365286407171852);
            meta.SetDD(direction);
            auto integration = Integration();
            integration.uvw = std::vector<casacore::MVuvw> { casacore::MVuvw() };
            integration.baselines = integration.uvw.size();
            integration.data = Eigen::MatrixXcd(metadata.channels, integration.baselines);


            if(impl == Impl::casa)
            {
                icrar::casa::RotateVisibilities(integration, metadata, direction);
            }
            if(impl == Impl::eigen)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                icrar::cpu::RotateVisibilities(integration, metadatahost, direction);
            }
            if(impl == Impl::cuda)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                auto metadatadevice = icrar::cuda::MetaDataCudaDevice(metadatahost);
                icrar::cuda::RotateVisibilities(integration, metadatadevice, direction);
                metadatadevice.ToHost(metadatahost);
            }

            auto expectedIntegration = Integration();
            expectedIntegration.baselines = 1;

            auto expectedMetadata = MetaData(ms);
            expectedMetadata.SetDD(direction);
            expectedMetadata.SetWv();
            expectedIntegration.uvw = integration.uvw;
            expectedMetadata.oldUVW = expectedIntegration.uvw;
            expectedMetadata.init = false;

            //Test case specific
            ASSERT_EQ(1, expectedIntegration.baselines);
            ASSERT_EQ(4, expectedMetadata.num_pols);
            auto expectedAvg_data = Eigen::MatrixXcd(expectedIntegration.baselines, metadata.num_pols);
            expectedAvg_data <<
            0,
            0.877191 + 0.357205i, //-0.549283 + 0.773963i,
            0,
            0;
            expectedMetadata.avg_data = ConvertMatrix(expectedAvg_data);

            
            ASSERT_EQ(expectedMetadata.init, metadata.init);
            ASSERT_EQ(expectedMetadata.nantennas, metadata.nantennas);
            //ASSERT_EQ(expectedMetadata.nbaseline, metadata.nbaseline);
            ASSERT_EQ(expectedMetadata.channels, metadata.channels);
            ASSERT_EQ(expectedMetadata.num_pols, metadata.num_pols);
            ASSERT_EQ(expectedMetadata.stations, metadata.stations);
            ASSERT_EQ(expectedMetadata.rows, metadata.rows);
            ASSERT_EQ(expectedMetadata.freq_start_hz, metadata.freq_start_hz);
            ASSERT_EQ(expectedMetadata.freq_inc_hz, metadata.freq_inc_hz);
            ASSERT_EQ(expectedMetadata.solution_interval, metadata.solution_interval);
            ASSERT_EQ(expectedMetadata.channel_wavelength, metadata.channel_wavelength);
            ASSERT_EQ(expectedMetadata.phase_centre_ra_rad, metadata.phase_centre_ra_rad);
            ASSERT_EQ(expectedMetadata.phase_centre_dec_rad, metadata.phase_centre_dec_rad);
            ASSERT_EQ(expectedMetadata.dlm_ra, metadata.dlm_ra);
            ASSERT_EQ(expectedMetadata.dlm_dec, metadata.dlm_dec);
            //ASSERT_EQ(expectedMetadata.oldUVW, metadata.oldUVW); //TODO

            ASSERT_EQ(expectedIntegration.baselines, metadata.avg_data.shape()[0]);
            ASSERT_EQ(expectedMetadata.num_pols, metadata.avg_data.shape()[1]);
        
            ASSERT_MEQCD(ConvertMatrix(expectedMetadata.avg_data), ConvertMatrix(metadata.avg_data), THRESHOLD);

            ASSERT_TRUE(icrar::Equal(expectedMetadata.dd, metadata.dd));
            ASSERT_MEQ(ConvertMatrix(expectedMetadata.A), ConvertMatrix(metadata.A), THRESHOLD);
            ASSERT_MEQI(ConvertMatrix<int>(expectedMetadata.I), ConvertMatrix<int>(metadata.I), THRESHOLD);
            ASSERT_MEQ(ConvertMatrix(expectedMetadata.Ad), ConvertMatrix(metadata.Ad), THRESHOLD);
            ASSERT_MEQ(ConvertMatrix(expectedMetadata.A1), ConvertMatrix(metadata.A1), THRESHOLD);
            ASSERT_MEQI(ConvertMatrix<int>(expectedMetadata.I1), ConvertMatrix<int>(metadata.I1), THRESHOLD);
            ASSERT_MEQ(ConvertMatrix(expectedMetadata.Ad1), ConvertMatrix(metadata.Ad1), THRESHOLD);
            
            //ASSERT_EQ(expectedMetadata, metadata);
            //ASSERT_EQ(expectedIntegration, integration);
        }

        void PhaseMatrixFunctionTest(Impl impl)
        {
            int refAnt = 0;
            bool map = true;

            try
            {
                if(impl == Impl::casa)
                {
                    const casacore::Vector<int32_t> a1;
                    const casacore::Vector<int32_t> a2;
                    icrar::casa::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                if(impl == Impl::eigen)
                {
                    auto a1 = Eigen::VectorXi(1);
                    auto a2 = Eigen::VectorXi(1);
                    icrar::cpu::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                if(impl == Impl::cuda)
                {
                    const Eigen::VectorXi a1;
                    const Eigen::VectorXi a2;
                    icrar::cuda::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
            }
            catch(std::invalid_argument& e)
            {
                
            }
            catch(...)
            {
                FAIL() << "Expected std::invalid_argument";
            }
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionTestCasa) { PhaseMatrixFunctionTest(Impl::casa); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(Impl::casa); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCasa) { PhaseRotateTest(Impl::casa); }

    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCpu) { PhaseMatrixFunctionTest(Impl::eigen); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(Impl::eigen); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCpu) { PhaseRotateTest(Impl::eigen); }

    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCuda) { PhaseMatrixFunctionTest(Impl::cuda); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(Impl::cuda); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCuda) { PhaseRotateTest(Impl::cuda); }
}
