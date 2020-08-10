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


#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/linear_math_helper.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/model/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/MetaDataCuda.h>


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
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(Impl impl)
        {
            auto metadata = casalib::MetaData(ms);
            std::vector<casacore::MVDirection> directions =
            {
                casacore::MVDirection(-0.4606549305661674,-0.29719233792392513),
                casacore::MVDirection(-0.753231018062671,-0.44387635324622354),
                casacore::MVDirection(-0.6207547100721282,-0.2539086572881469),
                casacore::MVDirection(-0.41958660604621867,-0.03677626900108552),
                casacore::MVDirection(-0.41108685258900596,-0.08638012622791202),
                casacore::MVDirection(-0.7782459495668798,-0.4887860989684432),
                casacore::MVDirection(-0.17001324965728973,-0.28595644149463484),
                casacore::MVDirection(-0.7129444556035118,-0.365286407171852),
                casacore::MVDirection(-0.1512764129166089,-0.21161026349648748)
            };

            if(impl == Impl::casa)
            {
                icrar::casalib::Calibrate(metadata, directions, 126, 3600);
            }
            if(impl == Impl::eigen)
            {
                // auto metadatahost = icrar::cuda::MetaData(metadata);
                // icrar::cpu::Calibrate(metadatahost, direction, input, output_integrations, output_calibrations);
            }
            if(impl == Impl::cuda)
            {
                // auto metadatahost = icrar::cuda::MetaData(metadata);
                // auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                // icrar::cuda::Calibrate(metadatadevice, direction, input, output_integrations, output_calibrations);
            }
            else
            {
                throw std::invalid_argument("impl");
            }
        }

        void RotateVisibilitiesTest(Impl impl)
        {
            const double THRESHOLD = 0.01;

            auto metadata = casalib::MetaData(ms);
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            auto integration = Integration(0, metadata.channels, metadata.GetBaselines(), metadata.num_pols, 2);
            integration.uvw = std::vector<casacore::MVuvw> { casacore::MVuvw(0, 0, 0), casacore::MVuvw(0, 0, 0) };
            integration.baselines = integration.uvw.size();

            boost::optional<icrar::cuda::MetaData> metadataOptionalOutput;
            if(impl == Impl::casa)
            {
                icrar::casalib::RotateVisibilities(integration, metadata, direction);
                metadataOptionalOutput = icrar::cuda::MetaData(metadata);
            }
            if(impl == Impl::eigen)
            {
                auto metadatahost = icrar::cuda::MetaData(metadata, direction, integration.uvw);
                icrar::cpu::RotateVisibilities(integration, metadatahost);
                metadataOptionalOutput = metadatahost;
            }
            if(impl == Impl::cuda)
            {
                auto metadatahost = icrar::cuda::MetaData(metadata, direction, integration.uvw);
                auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                icrar::cuda::RotateVisibilities(integration, metadatadevice);
                metadatadevice.ToHost(metadatahost);
                metadataOptionalOutput = metadatahost;
            }
            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cuda::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = Integration(0, metadata.channels, metadata.GetBaselines(), metadata.num_pols,  metadata.GetBaselines());
            expectedIntegration.baselines = integration.uvw.size();
            expectedIntegration.uvw = integration.uvw;

            //TODO: don't rely on eigen implementation for expected values
            auto expectedMetadata = icrar::cuda::MetaData(casalib::MetaData(ms), direction, integration.uvw);
            expectedMetadata.oldUVW = metadataOutput.oldUVW;

            //Test case specific
            expectedMetadata.dd = Eigen::Matrix3d();
            expectedMetadata.dd <<
             0.46856701,  0.86068501, -0.19916391,
            -0.79210108,  0.50913781,  0.33668172,
             0.39117878,  0.0,         0.92031471;

            ASSERT_EQ(2, expectedIntegration.baselines);
            ASSERT_EQ(4, expectedMetadata.GetConstants().num_pols);
            expectedMetadata.avg_data = Eigen::MatrixXcd(expectedIntegration.baselines, metadata.num_pols);
            expectedMetadata.avg_data <<
            0, 0, 0, 0,
            0, 0, 0, 0;

            // ==========
            // ASSERT
            ASSERT_EQ(expectedMetadata.GetConstants().num_pols, metadataOutput.avg_data.cols());
            ASSERT_MDEQ(expectedMetadata, metadataOutput, THRESHOLD);

            //ASSERT_EQ(expectedMetadata, metadata);
            //ASSERT_EQ(expectedIntegration, integration);
        }

        void PhaseMatrixFunction0Test(Impl impl)
        {
            int refAnt = 0;
            bool map = true;

            try
            {
                if(impl == Impl::casa)
                {
                    const casacore::Vector<int32_t> a1;
                    const casacore::Vector<int32_t> a2;
                    icrar::casalib::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                if(impl == Impl::eigen)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
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

        Eigen::MatrixXd GetExpectedA1()
        {
            Eigen::MatrixXd A1Expected = Eigen::MatrixXd::Zero(4854, 128);
            A1Expected(0, 0) = 1;
            A1Expected(1, 0) = 1;
            A1Expected(2, 0) = 1;
            A1Expected(3, 0) = 1;
            A1Expected(4, 0) = 1;
            A1Expected(5, 0) = 1;
            A1Expected(6, 0) = 1;
            A1Expected(7, 0) = 1;
            A1Expected(8, 0) = 1;
            A1Expected(9, 0) = 1;
            A1Expected(10, 0) = 1;
            A1Expected(11, 0) = 1;
            A1Expected(12, 0) = 1;
            A1Expected(13, 0) = 1;
            A1Expected(14, 0) = 1;
            A1Expected(15, 0) = 1;
            A1Expected(16, 0) = 1;
            A1Expected(17, 0) = 1;
            A1Expected(18, 0) = 1;
            A1Expected(19, 0) = 1;
            A1Expected(20, 0) = 1;
            A1Expected(21, 0) = 1;
            A1Expected(22, 0) = 1;
            A1Expected(23, 0) = 1;
            A1Expected(24, 0) = 1;
            A1Expected(25, 0) = 1;
            A1Expected(26, 0) = 1;
            A1Expected(27, 0) = 1;
            A1Expected(28, 0) = 1;
            A1Expected(29, 0) = 1;
            A1Expected(30, 0) = 1;
            A1Expected(31, 0) = 1;
            A1Expected(32, 0) = 1;
            A1Expected(33, 0) = 1;
            A1Expected(34, 0) = 1;
            A1Expected(35, 0) = 1;
            A1Expected(36, 0) = 1;
            A1Expected(37, 0) = 1;
            A1Expected(38, 0) = 1;
            A1Expected(39, 0) = 1;
            A1Expected(40, 0) = 1;
            A1Expected(41, 0) = 1;
            A1Expected(42, 0) = 1;
            A1Expected(43, 0) = 1;
            A1Expected(44, 0) = 1;
            A1Expected(45, 0) = 1;
            A1Expected(46, 0) = 1;
            A1Expected(47, 0) = 1;
            A1Expected(48, 0) = 1;
            A1Expected(49, 0) = 1;
            A1Expected(50, 0) = 1;
            A1Expected(51, 0) = 1;
            A1Expected(52, 0) = 1;
            A1Expected(53, 0) = 1;
            A1Expected(54, 0) = 1;
            A1Expected(55, 0) = 1;
            A1Expected(56, 0) = 1;
            A1Expected(57, 0) = 1;
            A1Expected(58, 0) = 1;
            A1Expected(59, 0) = 1;
            A1Expected(60, 0) = 1;
            A1Expected(61, 0) = 1;
            A1Expected(62, 0) = 1;
            A1Expected(63, 0) = 1;
            A1Expected(64, 0) = 1;
            A1Expected(65, 0) = 1;
            A1Expected(66, 0) = 1;
            A1Expected(67, 0) = 1;
            A1Expected(68, 0) = 1;
            A1Expected(69, 0) = 1;
            A1Expected(70, 0) = 1;
            A1Expected(71, 0) = 1;
            A1Expected(72, 0) = 1;
            A1Expected(73, 0) = 1;
            A1Expected(74, 0) = 1;
            A1Expected(75, 0) = 1;
            A1Expected(76, 0) = 1;
            A1Expected(77, 0) = 1;
            A1Expected(78, 0) = 1;
            A1Expected(79, 0) = 1;
            A1Expected(80, 0) = 1;
            A1Expected(81, 0) = 1;
            A1Expected(82, 0) = 1;
            A1Expected(83, 0) = 1;
            A1Expected(84, 0) = 1;
            A1Expected(85, 0) = 1;
            A1Expected(86, 0) = 1;
            A1Expected(87, 0) = 1;
            A1Expected(88, 0) = 1;
            A1Expected(89, 0) = 1;
            A1Expected(90, 0) = 1;
            A1Expected(91, 0) = 1;
            A1Expected(92, 0) = 1;
            A1Expected(93, 0) = 1;
            A1Expected(94, 0) = 1;
            A1Expected(95, 0) = 1;
            A1Expected(96, 0) = 1;
            A1Expected(0, 1) = -1;
            A1Expected(1, 4) = -1;
            A1Expected(2, 5) = -1;
            A1Expected(3, 6) = -1;
            A1Expected(4, 7) = -1;
            A1Expected(5, 8) = -1;
            A1Expected(6, 12) = -1;
            A1Expected(7, 13) = -1;
            A1Expected(8, 14) = -1;
            A1Expected(9, 16) = -1;
            A1Expected(10, 17) = -1;
            A1Expected(11, 20) = -1;
            A1Expected(12, 21) = -1;
            A1Expected(13, 22) = -1;
            A1Expected(14, 23) = -1;
            A1Expected(15, 24) = -1;
            A1Expected(16, 25) = -1;
            A1Expected(17, 27) = -1;
            A1Expected(18, 28) = -1;
            A1Expected(19, 29) = -1;
            A1Expected(20, 31) = -1;
            A1Expected(21, 32) = -1;
            A1Expected(22, 33) = -1;
            A1Expected(23, 35) = -1;
            A1Expected(24, 36) = -1;
            A1Expected(25, 37) = -1;
            A1Expected(26, 40) = -1;
            A1Expected(27, 41) = -1;
            A1Expected(28, 42) = -1;
            A1Expected(29, 43) = -1;
            A1Expected(30, 44) = -1;
            A1Expected(31, 45) = -1;
            A1Expected(32, 46) = -1;
            A1Expected(33, 47) = -1;
            A1Expected(34, 48) = -1;
            A1Expected(35, 50) = -1;
            A1Expected(36, 52) = -1;
            A1Expected(37, 53) = -1;
            A1Expected(38, 54) = -1;
            A1Expected(39, 55) = -1;
            A1Expected(40, 56) = -1;
            A1Expected(41, 57) = -1;
            A1Expected(42, 58) = -1;
            A1Expected(43, 59) = -1;
            A1Expected(44, 65) = -1;
            A1Expected(45, 66) = -1;
            A1Expected(46, 67) = -1;
            A1Expected(47, 68) = -1;
            A1Expected(48, 69) = -1;
            A1Expected(49, 72) = -1;
            A1Expected(50, 73) = -1;
            A1Expected(51, 74) = -1;
            A1Expected(52, 75) = -1;
            A1Expected(53, 76) = -1;
            A1Expected(54, 77) = -1;
            A1Expected(55, 78) = -1;
            A1Expected(56, 79) = -1;
            A1Expected(57, 81) = -1;
            A1Expected(58, 82) = -1;
            A1Expected(59, 83) = -1;
            A1Expected(60, 84) = -1;
            A1Expected(61, 85) = -1;
            A1Expected(62, 86) = -1;
            A1Expected(63, 88) = -1;
            A1Expected(64, 89) = -1;
            A1Expected(65, 90) = -1;
            A1Expected(66, 92) = -1;
            A1Expected(67, 96) = -1;
            A1Expected(68, 97) = -1;
            A1Expected(69, 98) = -1;
            A1Expected(70, 99) = -1;
            A1Expected(71, 100) = -1;
            A1Expected(72, 101) = -1;
            A1Expected(73, 102) = -1;
            A1Expected(74, 103) = -1;
            A1Expected(75, 104) = -1;
            A1Expected(76, 105) = -1;
            A1Expected(77, 106) = -1;
            A1Expected(78, 107) = -1;
            A1Expected(79, 108) = -1;
            A1Expected(80, 109) = -1;
            A1Expected(81, 110) = -1;
            A1Expected(82, 112) = -1;
            A1Expected(83, 113) = -1;
            A1Expected(84, 114) = -1;
            A1Expected(85, 116) = -1;
            A1Expected(86, 117) = -1;
            A1Expected(87, 118) = -1;
            A1Expected(88, 119) = -1;
            A1Expected(89, 120) = -1;
            A1Expected(90, 121) = -1;
            A1Expected(91, 122) = -1;
            A1Expected(92, 123) = -1;
            A1Expected(93, 124) = -1;
            A1Expected(94, 125) = -1;
            A1Expected(95, 126) = -1;
            A1Expected(96, 127) = -1;
            return A1Expected;
        }

        Eigen::VectorXi GetExpectedI1()
        {
            auto I1Expected = Eigen::VectorXi(4854);
            I1Expected.setConstant(1);
            for(int i = 1; i < 97; ++i)
            {
                I1Expected(i) = i + 1;
            }
            return I1Expected;
        }

        void PhaseMatrixFunctionDataTest(Impl impl)
        {
            //int nantennas = 10;
            //int nstations = 1;

            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            auto ms = casacore::MeasurementSet(filename);
            auto msmc = std::make_unique<casacore::MSMainColumns>(ms);

            int nstations = ms.antenna().nrow(); //128

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int nEpochs = 0;
            for(int i = 0; i < time.size(); i++)
            {
                if(time[i] == time[0]) nEpochs++;
            }
            auto epochIndices = casacore::Slice(0, nEpochs, 1); //TODO assuming epoch indices are sorted

            int nantennas = nEpochs;

            casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
            casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

            //Start calculations
            int refAnt = 0;
            bool map = true;

            //output
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            if(impl == Impl::casa)
            {
                casacore::Matrix<double> casaA1;
                casacore::Array<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = icrar::casalib::PhaseMatrixFunction(a1, a2, refAnt, map);

                A1 = ConvertMatrix(casaA1);
                I1 = ConvertVector(casaI1);
            }
            if(impl == Impl::eigen)
            {
                auto ea1 = ConvertVector(a1);
                auto ea2 = ConvertVector(a2);
                std::tie(A1, I1) = icrar::cpu::PhaseMatrixFunction(ea1, ea2, refAnt, map);

            }
            if(impl == Impl::cuda)
            {
                auto ea1 = ConvertVector(a1);
                auto ea2 = ConvertVector(a2);
                std::tie(A1, I1) = icrar::cuda::PhaseMatrixFunction(ea1, ea2, refAnt, map);
            }

            ASSERT_EQ(nantennas + 1, A1.rows());
            ASSERT_EQ(nstations, A1.cols());

            Eigen::MatrixXd A1Expected = GetExpectedA1();
            ASSERT_MEQ(A1Expected, A1, 0.001);

            auto I1Expected = GetExpectedI1();
            ASSERT_VEQI(I1Expected, I1, 0.001);
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCasa) { PhaseMatrixFunction0Test(Impl::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(Impl::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCuda) { PhaseMatrixFunction0Test(Impl::cuda); }

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCasa) { PhaseMatrixFunctionDataTest(Impl::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(Impl::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCuda) { PhaseMatrixFunctionDataTest(Impl::cuda); }

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(Impl::casa); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(Impl::eigen); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(Impl::cuda); }
    
    TEST_F(PhaseRotateTests, PhaseRotateTestCasa) { PhaseRotateTest(Impl::casa); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCpu) { PhaseRotateTest(Impl::eigen); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCuda) { PhaseRotateTest(Impl::cuda); }
}
