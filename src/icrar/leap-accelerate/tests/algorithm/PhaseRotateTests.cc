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

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>

#include <icrar/leap-accelerate/model/casa/Integration.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <icrar/leap-accelerate/model/cuda/DeviceIntegration.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <gtest/gtest.h>

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
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 0.01;
            
            auto metadata = icrar::casalib::MetaData(*ms);

            std::vector<casacore::MVDirection> directions =
            {
                { -0.4606549305661674,-0.29719233792392513 },
                //{ -0.753231018062671,-0.44387635324622354 },
                //{ -0.6207547100721282,-0.2539086572881469 },
                //{ -0.41958660604621867,-0.03677626900108552 },
                //{ -0.41108685258900596,-0.08638012622791202 },
                //{ -0.7782459495668798,-0.4887860989684432 },
                //{ -0.17001324965728973,-0.28595644149463484 },
                //{ -0.7129444556035118,-0.365286407171852 },
                //{ -0.1512764129166089,-0.21161026349648748 }

            };

            std::vector<std::vector<cpu::IntegrationResult>> integrations;
            std::vector<std::vector<cpu::CalibrationResult>> calibrations;
            if(impl == ComputeImplementation::casa)
            {
                auto pair = icrar::casalib::Calibrate(*ms, directions, 3600);
                std::tie(integrations, calibrations) = cpu::ToCalibrateResult(pair);
            }
            else if(impl == ComputeImplementation::eigen)
            {
                std::tie(integrations, calibrations) = cpu::Calibrate(*ms, ToDirectionVector(directions), 3600);
            }
            else if(impl == ComputeImplementation::cuda)
            {
                std::tie(integrations, calibrations) = cuda::Calibrate(*ms, ToDirectionVector(directions), 3600);
            }
            else
            {
                throw std::invalid_argument("impl");
            }

            auto expected = GetExpectedCalibration();

            ASSERT_EQ(directions.size(), calibrations.size());
            for(int i = 0; i < calibrations.size(); i++)
            {
                casacore::MVDirection direction;
                std::vector<double> calibration;
                std::tie(direction, calibration) = expected[0];

                ASSERT_EQ(1, calibrations[i].size());
                const auto& result = calibrations[i].front();
                
                ASSERT_EQ(1, result.GetData().size());
#ifndef NDEBUG
                std::cout << std::setprecision(15) << "calibration result: " << result.GetData()[0] << std::endl;
#endif

                //TODO: assert with LEAP-Cal
                ASSERT_MEQ(ToVector(calibration), ToMatrix(result.GetData()[0]), THRESHOLD);
            }
        }

        void RotateVisibilitiesTest(ComputeImplementation impl)
        {
            using namespace std::complex_literals;
            const double THRESHOLD = 0.01;
            
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            //auto eigenuvw = ToUVWVector(ms->GetCoords(index, baselines));
            //auto casauvw = ToCasaUVWVector(ms->GetCoords(index, baselines));
            //std::cout << "eigenuvw[1]" << eigenuvw[1] << std::endl;
            //std::cout << "casauvw[1]" << casauvw[1] << std::endl;
            //assert(uvw[1](0) == -75.219106714973222);

            boost::optional<icrar::cpu::Integration> integrationOptionalOutput;
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
#ifdef PROFILING
                auto startTime = std::chrono::high_resolution_clock::now();
#endif
                icrar::casalib::RotateVisibilities(integration, metadata, direction);
#ifdef PROFILING
                auto endTime = std::chrono::high_resolution_clock::now();
#endif
                integrationOptionalOutput = icrar::cpu::Integration(integration);
                metadataOptionalOutput = icrar::cpu::MetaData(metadata);

#ifdef PROFILING
                std::cout << "casa time" << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
#endif
            }
            if(impl == ComputeImplementation::eigen)
            {
                
                auto integration = cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto metadatahost = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
#ifdef PROFILING
                auto startTime = std::chrono::high_resolution_clock::now();
#endif
                icrar::cpu::RotateVisibilities(integration, metadatahost);
#ifdef PROFILING
                auto endTime = std::chrono::high_resolution_clock::now(); //remove the polar conversion?
#endif
                integrationOptionalOutput = integration;
                metadataOptionalOutput = metadatahost;
#ifdef PROFILING
                std::cout << "eigen time" << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
#endif          
            }
            if(impl == ComputeImplementation::cuda)
            {
                auto integration = icrar::cpu::Integration(
                    0,
                    *ms,
                    0,
                    ms->GetNumChannels(),
                    ms->GetNumBaselines(),
                    ms->GetNumPols());

                auto metadatahost = icrar::cpu::MetaData(*ms, ToDirection(direction), integration.GetUVW());
                auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                icrar::cuda::RotateVisibilities(deviceIntegration, metadatadevice);
                metadatadevice.ToHost(metadatahost);
                integrationOptionalOutput = integration;
                metadataOptionalOutput = metadatahost;
            }
            ASSERT_TRUE(integrationOptionalOutput.is_initialized());
            icrar::cpu::Integration& integrationOutput = integrationOptionalOutput.get();

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = icrar::casalib::Integration(0, *ms, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
            expectedIntegration.uvw = ToCasaUVWVector(ms->GetCoords());

            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nantennas = 0;
            expectedConstants.nbaselines = 8001;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 126;
            expectedConstants.rows = 63089;
            expectedConstants.solution_interval = 3601;
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
            EXPECT_DOUBLE_EQ(expectedConstants.nantennas, metadataOutput.GetConstants().nantennas);
            EXPECT_DOUBLE_EQ(expectedConstants.nbaselines, metadataOutput.GetConstants().nbaselines);
            EXPECT_DOUBLE_EQ(expectedConstants.channels, metadataOutput.GetConstants().channels);
            EXPECT_DOUBLE_EQ(expectedConstants.num_pols, metadataOutput.GetConstants().num_pols);
            EXPECT_DOUBLE_EQ(expectedConstants.stations, metadataOutput.GetConstants().stations);
            EXPECT_DOUBLE_EQ(expectedConstants.rows, metadataOutput.GetConstants().rows);
            EXPECT_DOUBLE_EQ(expectedConstants.solution_interval, metadataOutput.GetConstants().solution_interval);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_start_hz, metadataOutput.GetConstants().freq_start_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.freq_inc_hz, metadataOutput.GetConstants().freq_inc_hz);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_ra_rad, metadataOutput.GetConstants().phase_centre_ra_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.phase_centre_dec_rad, metadataOutput.GetConstants().phase_centre_dec_rad);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_ra, metadataOutput.GetConstants().dlm_ra);
            EXPECT_DOUBLE_EQ(expectedConstants.dlm_dec, metadataOutput.GetConstants().dlm_dec);
            ASSERT_TRUE(expectedConstants == metadataOutput.GetConstants());        
            
            EXPECT_DOUBLE_EQ(expectedDD(0,0), metadataOutput.dd(0,0));
            EXPECT_DOUBLE_EQ(expectedDD(0,1), metadataOutput.dd(0,1));
            EXPECT_DOUBLE_EQ(expectedDD(0,2), metadataOutput.dd(0,2));
            EXPECT_DOUBLE_EQ(expectedDD(1,0), metadataOutput.dd(1,0));
            EXPECT_DOUBLE_EQ(expectedDD(1,1), metadataOutput.dd(1,1));
            EXPECT_DOUBLE_EQ(expectedDD(1,2), metadataOutput.dd(1,2));
            EXPECT_DOUBLE_EQ(expectedDD(2,0), metadataOutput.dd(2,0));
            EXPECT_DOUBLE_EQ(expectedDD(2,1), metadataOutput.dd(2,1));
            EXPECT_DOUBLE_EQ(expectedDD(2,2), metadataOutput.dd(2,2));

            auto cthreshold = std::complex<double>(0.001, 0.001);

            // ASSERT_EQCD(0.0+0.0i, integrationOutput.GetData()(0,0,0), THRESHOLD);
            // ASSERT_EQCD(-24.9622-30.7819i, integrationOutput.GetData()(0,1,0), THRESHOLD);
            // ASSERT_EQCD(-16.0242+31.1452i, integrationOutput.GetData()(0,2,0), THRESHOLD);

            //ASSERT_EQCD(0.0+0.0i, integrationOutput.GetData()(0,0,0), THRESHOLD);
            //ASSERT_EQCD(35.6735722091204 + 17.2635476789549i, integrationOutput.GetData()(0,1,0), THRESHOLD);
            //ASSERT_EQCD(25.3136137725085 + -24.2077854859281i, integrationOutput.GetData()(0,2,0), THRESHOLD);

            ASSERT_EQ(8001, metadataOutput.avg_data.rows());
            ASSERT_EQ(4, metadataOutput.avg_data.cols());
            ASSERT_EQCD(152.207482222774 + 157.780854994143i, metadataOutput.avg_data(1,0), THRESHOLD);
            ASSERT_EQCD(237.735520799299 + 123.628127794715i, metadataOutput.avg_data(1,1), THRESHOLD);
            ASSERT_EQCD(3.57682429815259 + -75.3381937487565i, metadataOutput.avg_data(1,2), THRESHOLD);
            ASSERT_EQCD(-168.342543770758 + -87.1917020804175i, metadataOutput.avg_data(1,3), THRESHOLD);
        }

        void PhaseMatrixFunction0Test(ComputeImplementation impl)
        {
            int refAnt = 0;
            bool map = true;

            try
            {
                if(impl == ComputeImplementation::casa)
                {
                    const casacore::Vector<int32_t> a1;
                    const casacore::Vector<int32_t> a2;
                    icrar::casalib::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                if(impl == ComputeImplementation::eigen)
                {
                    auto a1 = Eigen::VectorXi();
                    auto a2 = Eigen::VectorXi();
                    icrar::cpu::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                if(impl == ComputeImplementation::cuda)
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

        std::vector<std::pair<casacore::MVDirection, std::vector<double>>> GetExpectedCalibration()
        {
            std::vector<std::pair<casacore::MVDirection, std::vector<double>>> output;
            output.push_back(std::make_pair(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>{
                -0.728771850537932,
                -0.272984982217124,
                0,
                0,
                -0.165700172596898,
                -0.229753294024841,
                -0.573810560316764,
                -0.612707257340495,
                -0.553899256476972,
                0,
                0,
                0,
                -0.44068499390976,
                -0.409222601326422,
                -0.571310475749061,
                0,
                -0.353810473560959,
                -0.567253122480653,
                0,
                0,
                -0.390115870959069,
                -0.27066056626084,
                -0.553722448531889,
                -0.456395891052881,
                -0.018552177789207,
                -0.276622901244882,
                0,
                -0.288017778301478,
                -0.646669510918547,
                -0.579349816207018,
                0,
                -0.403442642627818,
                -0.66971997908397,
                -0.216615288790623,
                0,
                -0.466098473086471,
                -0.312644385625222,
                -0.560453321783426,
                0,
                0,
                -0.297359362580634,
                -0.25700167376058,
                -0.538472962780001,
                -0.268184096901752,
                -0.115719471775436,
                -0.46649624464272,
                -0.486119424331862,
                -0.527341765199047,
                -0.662436501411798,
                0,
                -0.52538966707018,
                0,
                -0.591959624959538,
                -0.293389709057398,
                -0.571769546208154,
                -0.461431071036542,
                -0.69827220309119,
                -0.596764537546079,
                -0.557914581432327,
                -0.60585573996508,
                0,
                0,
                0,
                0,
                0,
                -0.576064123662643,
                -0.578117694374516,
                -0.736233357704992,
                -0.602098333525665,
                -0.608274968533843,
                0,
                0,
                -0.879002815749142,
                -0.148596828802831,
                -0.572404457009006,
                -0.468371935399828,
                -0.422146642185461,
                -0.327880572287607,
                -0.0915081715901103,
                -0.158511591522468,
                0,
                -0.426389264447865,
                -0.139521775004381,
                -0.337312794195202,
                -0.544457292447768,
                -0.511656054195138,
                -0.164048900126062,
                0,
                -0.656303639745102,
                -0.335344067087277,
                -0.82309375869757,
                0,
                -0.371477243004447,
                0,
                0,
                0,
                -0.410837107807203,
                -0.32513727097287,
                -0.328328761630409,
                -0.444960762933782,
                -0.509293189841975,
                -0.369724821544758,
                -0.725307982947541,
                -0.335940533875519,
                -0.462416690296085,
                -0.419349193489786,
                -0.441903126642277,
                -0.407528165019852,
                -0.273432501400391,
                -0.524227735137418,
                -0.490111247600731,
                0,
                -0.703262102152449,
                -0.180113486322149,
                -0.0964145756719589,
                0,
                -0.0976397203984036,
                -0.467975697984875,
                -0.0510049879991166,
                -0.490338003258752,
                -0.870678300688149,
                -0.426715995938114,
                -0.555159229452293,
                -0.0743951211822176,
                -0.583101273144654,
                -0.589907136674013,
                -0.435703316716115,
                0.0205611886165418,
            }));

            return output;
        }

        std::vector<std::pair<casacore::MVDirection, std::vector<double>>> GetPythonCalibration()
        {
            std::vector<std::pair<casacore::MVDirection, std::vector<double>>> output;
            output.push_back(std::make_pair(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>{
                2.82837057e+00,  3.00762994e+00,  8.53843848e-16, -1.68355222e-15,
                2.96613576e+00,  3.26952537e+00,  2.73594265e+00,  2.88927258e+00,
                2.76638911e+00, -4.88427424e-17, -1.67191438e-15, -1.00382254e-15,
                3.09779782e+00,  3.02964471e+00,  2.77083490e+00,  2.94276942e-15,
                2.70048391e+00,  2.68046281e+00,  4.44100967e-15, -9.65911598e-17,
                2.92247690e+00,  2.94246184e+00,  2.99540724e+00,  3.00046475e+00,
                3.16929796e+00,  3.05537337e+00,  1.40429478e-15,  3.27306372e+00,
                2.89033018e+00,  2.77328291e+00, -1.20200661e-15,  3.03302677e+00,
                2.62320436e+00,  3.03649653e+00,  2.06945403e-16,  2.79987633e+00,
                2.94891315e+00,  2.64359818e+00, -1.25519247e-15, -3.34064905e-15,
                3.17691873e+00,  3.00708043e+00,  2.84658764e+00,  2.92237786e+00,
                3.18199545e+00,  2.87549313e+00,  2.95959874e+00,  2.68938577e+00,
                2.73051736e+00, -7.51094703e-16,  2.89901500e+00,  1.28481563e-15,
                2.82179574e+00,  3.14853577e+00,  2.98364742e+00,  2.98765950e+00,
                2.87233251e+00,  2.79630304e+00,  2.88152321e+00,  2.71366573e+00,
                1.25165078e-15,  3.45550475e-15,  1.01808361e-15, -3.11987190e-15,
                -5.45209305e-15,  2.76527545e+00,  2.67869650e+00,  2.68752433e+00,
                2.86361153e+00,  2.72944922e+00, -1.75943920e-16,  1.17337357e-15,
                2.55548928e+00,  3.14283381e+00,  2.81851527e+00,  2.84267830e+00,
                2.84820770e+00,  2.69509426e+00,  3.21021192e+00,  3.06505159e+00,
                -3.89732041e-16,  2.79503995e+00,  2.99170790e+00,  2.92197148e+00,
                2.80478198e+00,  2.67396709e+00,  3.01408558e+00,  2.01418090e-16,
                2.62260434e+00,  3.11520286e+00,  2.61278811e+00,  1.43483145e-15,
                3.13077821e+00, -1.06094630e-15, -6.47003411e-16, -1.20230509e-15,
                2.97224865e+00,  2.86369811e+00,  3.25868192e+00,  3.01513895e+00,
                2.68918716e+00,  2.79656781e+00,  2.70057553e+00,  3.11158050e+00,
                2.91778673e+00,  2.63628022e+00,  3.07275702e+00,  3.16980398e+00,
                3.00434587e+00,  2.65428333e+00,  2.85211245e+00,  5.67741669e-16,
                3.03357082e+00,  2.93599456e+00,  3.18832249e+00, -3.17886427e-16,
                3.22383359e+00,  2.98925674e+00,  3.02415049e+00,  2.94730568e+00,
                2.66804119e+00,  2.91573887e+00,  2.76595422e+00,  3.15652661e+00,
                2.96306376e+00,  2.78359436e+00,  2.95934717e+00,  3.19582259e+00}));

            return output;
        }

        Eigen::MatrixXd GetExpectedA()
        {
            Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(4754, 128); //TODO: emergent, validate with LEAP-Cal
            return expected;
        }

        Eigen::VectorXi GetExpectedI()
        {
            auto expected = Eigen::VectorXi(4754);
            expected.setConstant(1);
            for(int i = 1; i < 4753; ++i)
            {
                expected(i) = i + 39;
            }
            return expected;
        }

        Eigen::MatrixXd GetExpectedA1()
        {
            Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(98, 128);
            expected(0, 0) = 1;
            expected(1, 0) = 1;
            expected(2, 0) = 1;
            expected(3, 0) = 1;
            expected(4, 0) = 1;
            expected(5, 0) = 1;
            expected(6, 0) = 1;
            expected(7, 0) = 1;
            expected(8, 0) = 1;
            expected(9, 0) = 1;
            expected(10, 0) = 1;
            expected(11, 0) = 1;
            expected(12, 0) = 1;
            expected(13, 0) = 1;
            expected(14, 0) = 1;
            expected(15, 0) = 1;
            expected(16, 0) = 1;
            expected(17, 0) = 1;
            expected(18, 0) = 1;
            expected(19, 0) = 1;
            expected(20, 0) = 1;
            expected(21, 0) = 1;
            expected(22, 0) = 1;
            expected(23, 0) = 1;
            expected(24, 0) = 1;
            expected(25, 0) = 1;
            expected(26, 0) = 1;
            expected(27, 0) = 1;
            expected(28, 0) = 1;
            expected(29, 0) = 1;
            expected(30, 0) = 1;
            expected(31, 0) = 1;
            expected(32, 0) = 1;
            expected(33, 0) = 1;
            expected(34, 0) = 1;
            expected(35, 0) = 1;
            expected(36, 0) = 1;
            expected(37, 0) = 1;
            expected(38, 0) = 1;
            expected(39, 0) = 1;
            expected(40, 0) = 1;
            expected(41, 0) = 1;
            expected(42, 0) = 1;
            expected(43, 0) = 1;
            expected(44, 0) = 1;
            expected(45, 0) = 1;
            expected(46, 0) = 1;
            expected(47, 0) = 1;
            expected(48, 0) = 1;
            expected(49, 0) = 1;
            expected(50, 0) = 1;
            expected(51, 0) = 1;
            expected(52, 0) = 1;
            expected(53, 0) = 1;
            expected(54, 0) = 1;
            expected(55, 0) = 1;
            expected(56, 0) = 1;
            expected(57, 0) = 1;
            expected(58, 0) = 1;
            expected(59, 0) = 1;
            expected(60, 0) = 1;
            expected(61, 0) = 1;
            expected(62, 0) = 1;
            expected(63, 0) = 1;
            expected(64, 0) = 1;
            expected(65, 0) = 1;
            expected(66, 0) = 1;
            expected(67, 0) = 1;
            expected(68, 0) = 1;
            expected(69, 0) = 1;
            expected(70, 0) = 1;
            expected(71, 0) = 1;
            expected(72, 0) = 1;
            expected(73, 0) = 1;
            expected(74, 0) = 1;
            expected(75, 0) = 1;
            expected(76, 0) = 1;
            expected(77, 0) = 1;
            expected(78, 0) = 1;
            expected(79, 0) = 1;
            expected(80, 0) = 1;
            expected(81, 0) = 1;
            expected(82, 0) = 1;
            expected(83, 0) = 1;
            expected(84, 0) = 1;
            expected(85, 0) = 1;
            expected(86, 0) = 1;
            expected(87, 0) = 1;
            expected(88, 0) = 1;
            expected(89, 0) = 1;
            expected(90, 0) = 1;
            expected(91, 0) = 1;
            expected(92, 0) = 1;
            expected(93, 0) = 1;
            expected(94, 0) = 1;
            expected(95, 0) = 1;
            expected(96, 0) = 1;
            expected(97, 0) = 1;
            expected(0, 1) = -1;
            expected(1, 4) = -1;
            expected(2, 5) = -1;
            expected(3, 6) = -1;
            expected(4, 7) = -1;
            expected(5, 8) = -1;
            expected(6, 12) = -1;
            expected(7, 13) = -1;
            expected(8, 14) = -1;
            expected(9, 16) = -1;
            expected(10, 17) = -1;
            expected(11, 20) = -1;
            expected(12, 21) = -1;
            expected(13, 22) = -1;
            expected(14, 23) = -1;
            expected(15, 24) = -1;
            expected(16, 25) = -1;
            expected(17, 27) = -1;
            expected(18, 28) = -1;
            expected(19, 29) = -1;
            expected(20, 31) = -1;
            expected(21, 32) = -1;
            expected(22, 33) = -1;
            expected(23, 35) = -1;
            expected(24, 36) = -1;
            expected(25, 37) = -1;
            expected(26, 40) = -1;
            expected(27, 41) = -1;
            expected(28, 42) = -1;
            expected(29, 43) = -1;
            expected(30, 44) = -1;
            expected(31, 45) = -1;
            expected(32, 46) = -1;
            expected(33, 47) = -1;
            expected(34, 48) = -1;
            expected(35, 50) = -1;
            expected(36, 52) = -1;
            expected(37, 53) = -1;
            expected(38, 54) = -1;
            expected(39, 55) = -1;
            expected(40, 56) = -1;
            expected(41, 57) = -1;
            expected(42, 58) = -1;
            expected(43, 59) = -1;
            expected(44, 65) = -1;
            expected(45, 66) = -1;
            expected(46, 67) = -1;
            expected(47, 68) = -1;
            expected(48, 69) = -1;
            expected(49, 72) = -1;
            expected(50, 73) = -1;
            expected(51, 74) = -1;
            expected(52, 75) = -1;
            expected(53, 76) = -1;
            expected(54, 77) = -1;
            expected(55, 78) = -1;
            expected(56, 79) = -1;
            expected(57, 81) = -1;
            expected(58, 82) = -1;
            expected(59, 83) = -1;
            expected(60, 84) = -1;
            expected(61, 85) = -1;
            expected(62, 86) = -1;
            expected(63, 88) = -1;
            expected(64, 89) = -1;
            expected(65, 90) = -1;
            expected(66, 92) = -1;
            expected(67, 96) = -1;
            expected(68, 97) = -1;
            expected(69, 98) = -1;
            expected(70, 99) = -1;
            expected(71, 100) = -1;
            expected(72, 101) = -1;
            expected(73, 102) = -1;
            expected(74, 103) = -1;
            expected(75, 104) = -1;
            expected(76, 105) = -1;
            expected(77, 106) = -1;
            expected(78, 107) = -1;
            expected(79, 108) = -1;
            expected(80, 109) = -1;
            expected(81, 110) = -1;
            expected(82, 112) = -1;
            expected(83, 113) = -1;
            expected(84, 114) = -1;
            expected(85, 116) = -1;
            expected(86, 117) = -1;
            expected(87, 118) = -1;
            expected(88, 119) = -1;
            expected(89, 120) = -1;
            expected(90, 121) = -1;
            expected(91, 122) = -1;
            expected(92, 123) = -1;
            expected(93, 124) = -1;
            expected(94, 125) = -1;
            expected(95, 126) = -1;
            expected(96, 127) = -1;
            return expected;
        }

        Eigen::VectorXi GetExpectedI1()
        {
            auto expected = Eigen::VectorXi(98);
            expected.setConstant(1);
            for(int i = 1; i < 97; ++i)
            {
                expected(i) = i + 1;
            }
            return expected;
        }

        void PhaseMatrixFunctionDataTest(ComputeImplementation impl)
        {
            auto msmc = ms->GetMSMainColumns();

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
            bool map = true;

            //output
            Eigen::MatrixXd A;
            Eigen::VectorXi I;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            if(impl == ComputeImplementation::casa)
            {
                casacore::Matrix<double> casaA;
                casacore::Array<std::int32_t> casaI;
                std::tie(casaA, casaI) = casalib::PhaseMatrixFunction(a1, a2, -1, map);

                casacore::Matrix<double> casaA1;
                casacore::Array<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = casalib::PhaseMatrixFunction(a1, a2, 0, map);

                A = ToMatrix(casaA);
                I = ToVector(casaI);
                A1 = ToMatrix(casaA1);
                I1 = ToVector(casaI1);
            }
            if(impl == ComputeImplementation::eigen)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                std::tie(A, I) = cpu::PhaseMatrixFunction(ea1, ea2, -1, map);
                std::tie(A1, I1) = cpu::Pcd .haseMatrixFunction(ea1, ea2, 0, map);

            }
            if(impl == ComputeImplementation::cuda)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                std::tie(A, I) = cuda::PhaseMatrixFunction(ea1, ea2, -1, map);
                std::tie(A1, I1) = cuda::PhaseMatrixFunction(ea1, ea2, 0, map);
            }

            auto IExpected = GetExpectedI();

            ASSERT_EQ(4754, A.rows()); //-32=4754, -split=5152
            ASSERT_EQ(128, A.cols());
            ASSERT_EQ(4754, I.size());

            ASSERT_EQ(98, A1.rows()); //-32=98, -split=102
            ASSERT_EQ(128, A1.cols());
            ASSERT_EQ(98, I1.size());

            //TODO: print out to comma seperated row major form
            //ASSERT_MEQ(GetExpectedA(), A, 0.001);
            //ASSERT_VEQI(GetExpectedI(), I, 0.001);
            //ASSERT_MEQ(GetExpectedA1(), A1, 0.001);
            //ASSERT_VEQI(GetExpectedI1(), I1, 0.001);clearclea
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCasa) { PhaseMatrixFunction0Test(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCuda) { PhaseMatrixFunction0Test(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCasa) { PhaseMatrixFunctionDataTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCuda) { PhaseMatrixFunctionDataTest(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(ComputeImplementation::cuda); }
    
    TEST_F(PhaseRotateTests, PhaseRotateTestCasa) { PhaseRotateTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCuda) { PhaseRotateTest(ComputeImplementation::cuda); }
}
