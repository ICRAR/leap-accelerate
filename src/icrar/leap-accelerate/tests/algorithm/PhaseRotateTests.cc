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

#include "PhaseRotateTestCaseData.h"

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>


#include <icrar/leap-accelerate/algorithm/casa/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseMatrixFunction.h>
#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>
#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

#include <boost/log/trivial.hpp>

#include <vector>
#include <set>
#include <unordered_map>

using namespace std::literals::complex_literals;

namespace icrar
{
    class PhaseRotateTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;

    protected:

        PhaseRotateTests() {

        }

        ~PhaseRotateTests() override
        {

        }

        void SetUp() override
        {
            std::string filename = std::string(TEST_DATA_DIR) + "/mwa/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126, true);
            std::cout << std::setprecision(15);
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 1e-12;

            auto metadata = icrar::casalib::MetaData(*ms);

            std::vector<casacore::MVDirection> directions =
            {
                { -0.4606549305661674,-0.29719233792392513 },
                { -0.753231018062671,-0.44387635324622354 },
                { -0.6207547100721282,-0.2539086572881469 },
                { -0.41958660604621867,-0.03677626900108552 },
                { -0.41108685258900596,-0.08638012622791202 },
                { -0.7782459495668798,-0.4887860989684432 },
                //{ -0.17001324965728973,-0.28595644149463484 },
                //{ -0.7129444556035118,-0.365286407171852 },
                //{ -0.1512764129166089,-0.21161026349648748 }

            };

            std::vector<std::vector<cpu::IntegrationResult>> integrations;
            std::vector<std::vector<cpu::CalibrationResult>> calibrations;
            if(impl == ComputeImplementation::casa)
            {
                auto pair = icrar::casalib::Calibrate(*ms, directions);
                std::tie(integrations, calibrations) = cpu::ToCalibrateResult(pair);
            }
            else if(impl == ComputeImplementation::cpu)
            {
                std::tie(integrations, calibrations) = cpu::Calibrate(*ms, ToDirectionVector(directions));
            }
            else if(impl == ComputeImplementation::cuda)
            {
#ifdef CUDA_ENABLED
                std::tie(integrations, calibrations) = cuda::Calibrate(*ms, ToDirectionVector(directions));
#endif // CUDA_ENABLED
            }
            else
            {
                throw std::invalid_argument("impl");
            }

            auto expected = GetExpectedCalibration();

            ASSERT_EQ(directions.size(), calibrations.size());
            for(size_t i = 0; i < expected.size(); i++)
            {
                casacore::MVDirection expectedDirection;
                std::vector<double> expectedCalibration;
                std::tie(expectedDirection, expectedCalibration) = expected[i];

                ASSERT_EQ(1, calibrations[i].size());
                const auto& result = calibrations[i].front();

                ASSERT_EQ(expectedDirection(0), result.GetDirection()(0));
                ASSERT_EQ(expectedDirection(1), result.GetDirection()(1));

                if(!ToVector(expectedCalibration).isApprox(result.GetData(), THRESHOLD))
                {
                    std::cout << i+1 << "/" << expected.size() << " got:\n" << result.GetData() << std::endl;
                }
                ASSERT_MEQD(ToVector(expectedCalibration), result.GetData(), THRESHOLD);
            }
        }

        void RotateVisibilitiesTest(ComputeImplementation impl)
        {
            using namespace std::complex_literals;
            const double THRESHOLD = 0.01;
            
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            boost::optional<icrar::cpu::MetaData> metadataOptionalOutput;
            if(impl == ComputeImplementation::casa)
            {
                auto metadata = casalib::MetaData(*ms);
                auto integration = casalib::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                icrar::casalib::RotateVisibilities(integration, metadata, direction);
                metadataOptionalOutput = icrar::cpu::MetaData(metadata);
            }
            if(impl == ComputeImplementation::cpu)
            {
                
                auto integration = cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto hostMetadata = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                icrar::cpu::RotateVisibilities(integration, hostMetadata);

                metadataOptionalOutput = hostMetadata;
            }
            if(impl == ComputeImplementation::cuda)
            {
#ifdef CUDA_ENABLED
                auto integration = icrar::cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto hostMetadata = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                auto constantMetadata = std::make_shared<icrar::cuda::ConstantMetaData>(
                    hostMetadata.GetConstants(),
                    hostMetadata.GetA(),
                    hostMetadata.GetI(),
                    hostMetadata.GetAd(),
                    hostMetadata.GetA1(),
                    hostMetadata.GetI1(),
                    hostMetadata.GetAd1()
                );
                auto deviceMetadata = icrar::cuda::DeviceMetaData(constantMetadata, hostMetadata);
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                icrar::cuda::RotateVisibilities(deviceIntegration, deviceMetadata);
                deviceMetadata.ToHost(hostMetadata);
                metadataOptionalOutput = hostMetadata;
#endif // CUDA_ENABLED
            }

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = icrar::casalib::Integration(0, *ms, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
            expectedIntegration.uvw = ToCasaUVWVector(ms->GetCoords());

            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nbaselines = 8001;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 126;
            expectedConstants.rows = 63089;
            expectedConstants.freq_start_hz = 1.39195e+08;
            expectedConstants.freq_inc_hz = 640000;
            expectedConstants.phase_centre_ra_rad = 0.57595865315812877;
            expectedConstants.phase_centre_dec_rad = 0.10471975511965978;
            expectedConstants.dlm_ra = -1.0366135837242962;
            expectedConstants.dlm_dec = -0.40191209304358488;
            auto expectedDD = Eigen::Matrix3d();
            expectedDD <<
            0.46856701307821974, 0.860685013060222, -0.19916390874975543,
            -0.792101075276669, 0.509137808744868, 0.336681716539552,
            0.39117878367889541, 0, 0.920314706608288;

            //========
            // ASSERT
            //========
            EXPECT_DOUBLE_EQ(expectedConstants.nbaselines, metadataOutput.GetConstants().nbaselines);
            EXPECT_DOUBLE_EQ(expectedConstants.channels, metadataOutput.GetConstants().channels);
            EXPECT_DOUBLE_EQ(expectedConstants.num_pols, metadataOutput.GetConstants().num_pols);
            EXPECT_DOUBLE_EQ(expectedConstants.stations, metadataOutput.GetConstants().stations);
            EXPECT_DOUBLE_EQ(expectedConstants.rows, metadataOutput.GetConstants().rows);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_start_hz, metadataOutput.GetConstants().freq_start_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_inc_hz, metadataOutput.GetConstants().freq_inc_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_ra_rad, metadataOutput.GetConstants().phase_centre_ra_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_dec_rad, metadataOutput.GetConstants().phase_centre_dec_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_ra, metadataOutput.GetConstants().dlm_ra);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_dec, metadataOutput.GetConstants().dlm_dec);
            ASSERT_TRUE(expectedConstants == metadataOutput.GetConstants());        
            
            EXPECT_DOUBLE_EQ(expectedDD(0,0), metadataOutput.GetDD()(0,0));
            EXPECT_DOUBLE_EQ(expectedDD(0,1), metadataOutput.GetDD()(0,1));
            EXPECT_DOUBLE_EQ(expectedDD(0,2), metadataOutput.GetDD()(0,2));
            EXPECT_DOUBLE_EQ(expectedDD(1,0), metadataOutput.GetDD()(1,0));
            EXPECT_DOUBLE_EQ(expectedDD(1,1), metadataOutput.GetDD()(1,1));
            EXPECT_DOUBLE_EQ(expectedDD(1,2), metadataOutput.GetDD()(1,2));
            EXPECT_DOUBLE_EQ(expectedDD(2,0), metadataOutput.GetDD()(2,0));
            EXPECT_DOUBLE_EQ(expectedDD(2,1), metadataOutput.GetDD()(2,1));
            EXPECT_DOUBLE_EQ(expectedDD(2,2), metadataOutput.GetDD()(2,2));

            ASSERT_EQ(8001, metadataOutput.GetAvgData().rows());
            ASSERT_EQ(4, metadataOutput.GetAvgData().cols());
            ASSERT_EQCD(152.207482222774 + 157.780854994143i, metadataOutput.GetAvgData()(1,0), THRESHOLD);
            ASSERT_EQCD(237.735520799299 + 123.628127794715i, metadataOutput.GetAvgData()(1,1), THRESHOLD);
            ASSERT_EQCD(3.57682429815259 + -75.3381937487565i, metadataOutput.GetAvgData()(1,2), THRESHOLD);
            ASSERT_EQCD(-168.342543770758 + -87.1917020804175i, metadataOutput.GetAvgData()(1,3), THRESHOLD);
        }

        void PhaseMatrixFunction0Test(ComputeImplementation impl)
        {
            int refAnt = 0;

            try
            {
                if(impl == ComputeImplementation::casa)
                {
                    const casacore::Vector<int32_t> a1;
                    const casacore::Vector<int32_t> a2;
                    icrar::casalib::PhaseMatrixFunction(a1, a2, refAnt);
                }
                if(impl == ComputeImplementation::cpu)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
                    icrar::cpu::PhaseMatrixFunction(a1, a2, refAnt);
                }
                else
                {
                    throw icrar::invalid_argument_exception("invalid PhaseMatrixFunction implementation", "impl", __FILE__, __LINE__);
                }
            }
            catch(std::invalid_argument& e)
            {
                SUCCEED();
            }
            catch(...)
            {
                FAIL() << "Expected std::invalid_argument";
            }
        }

        void PhaseMatrixFunctionDataTest(ComputeImplementation impl)
        {
            auto msmc = ms->GetMSMainColumns();

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int epochRows = 0;
            for(size_t i = 0; i < time.size(); i++)
            {
                if(time[i] == epoch) epochRows++;
            }
            auto epochIndices = casacore::Slice(0, epochRows, 1); //TODO assuming epoch indices are sorted

            casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
            casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

            //Start calculations

            //output
            Eigen::MatrixXd A;
            Eigen::VectorXi I;
            Eigen::MatrixXd Ad;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            Eigen::MatrixXd Ad1;

            if(impl == ComputeImplementation::casa)
            {
                casacore::Matrix<double> casaA;
                casacore::Vector<std::int32_t> casaI;
                std::tie(casaA, casaI) = casalib::PhaseMatrixFunction(a1, a2, -1);
                Ad = ToMatrix(icrar::casalib::PseudoInverse(casaA));

                casacore::Matrix<double> casaA1;
                casacore::Vector<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = casalib::PhaseMatrixFunction(a1, a2, 0);
                Ad1 = ToMatrix(icrar::casalib::PseudoInverse(casaA1));

                A = ToMatrix(casaA);
                I = ToVector(casaI);
                A1 = ToMatrix(casaA1);
                I1 = ToVector(casaI1);
            }
            else if(impl == ComputeImplementation::cpu)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                std::tie(A, I) = cpu::PhaseMatrixFunction(ea1, ea2, -1);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, 0);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }
            else
            {
                throw std::invalid_argument("impl");
            }

            double TOLERANCE = 0.00001;

            // A
            ASSERT_DOUBLE_EQ(4754, A.rows()); //-32=4754, -split=5152
            ASSERT_DOUBLE_EQ(128, A.cols());
            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(4753,125), TOLERANCE);
            EXPECT_NEAR(0.00, A(4753,126), TOLERANCE);
            EXPECT_NEAR(0.00, A(4753,127), TOLERANCE);

            // I
            const int nBaselines = 4753;
            ASSERT_DOUBLE_EQ(nBaselines+1, I.size());
            ASSERT_EQ(4754, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(2.00, I(1));
            EXPECT_EQ(3.00, I(2));
            //...
            EXPECT_EQ(4849, I(4751));
            EXPECT_EQ(4851, I(4752));
            EXPECT_EQ(-1, I(4753));

            // Ad
            ASSERT_DOUBLE_EQ(128, Ad.rows());
            ASSERT_DOUBLE_EQ(4754, Ad.cols());
            // EXPECT_NEAR(2.62531368e-15, Ad(0,0), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(2.04033520e-15, Ad(0,1), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(3.25648083e-16, Ad(0,2), TOLERANCE); // TODO: emergent
            // //...
            // EXPECT_NEAR(-1.02040816e-02, Ad(127,95), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-0.020408163265312793, Ad(127,96), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-8.9737257304377696e-16, Ad(127,97), TOLERANCE); // TODO: emergent
            ASSERT_MEQD(A, A * Ad * A, TOLERANCE);

            //A1
            ASSERT_DOUBLE_EQ(98, A1.rows()); //-32=98, -split=102
            ASSERT_DOUBLE_EQ(128, A1.cols());
            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR(0.00, A1(97,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,126), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,127), TOLERANCE);

            //I1
            ASSERT_DOUBLE_EQ(98, I1.size());
            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(2.00, I1(1));
            EXPECT_DOUBLE_EQ(3.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(96.00, I1(95));
            EXPECT_DOUBLE_EQ(97.00, I1(96));
            EXPECT_DOUBLE_EQ(-1.00, I1(97));

            //Ad1
            ASSERT_DOUBLE_EQ(98, Ad1.cols());
            ASSERT_DOUBLE_EQ(128, Ad1.rows());
            //TODO: Ad1 not identical
            // EXPECT_DOUBLE_EQ(-9.8130778667735933e-18, Ad1(0,0)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(6.3742385976163974e-17, Ad1(0,1)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(3.68124219034074e-19, Ad1(0,2)); // TODO: emergent
            // //...
            // EXPECT_DOUBLE_EQ(5.4194040934156436e-17, Ad1(127,95)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(-1.0, Ad1(127,96)); // TODO: emergent
            // EXPECT_DOUBLE_EQ(1.0, Ad1(127,97)); // TODO: emergent
            ASSERT_MEQD(A1, A1 * Ad1 * A1, TOLERANCE);
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCasa) { PhaseMatrixFunction0Test(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::cpu); }

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCasa) { PhaseMatrixFunctionDataTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::cpu); }

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(ComputeImplementation::cuda); }
    
    TEST_F(PhaseRotateTests, PhaseRotateTestCasa) { PhaseRotateTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(ComputeImplementation::cpu); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCuda) { PhaseRotateTest(ComputeImplementation::cuda); }
}
