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
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>
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
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-split.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 102, true);
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
                auto pair = icrar::casalib::Calibrate(*ms, directions);
                std::tie(integrations, calibrations) = cpu::ToCalibrateResult(pair);
            }
            else if(impl == ComputeImplementation::cpu)
            {
                std::tie(integrations, calibrations) = cpu::Calibrate(*ms, ToDirectionVector(directions));
            }
            else if(impl == ComputeImplementation::cuda)
            {
                std::tie(integrations, calibrations) = cuda::Calibrate(*ms, ToDirectionVector(directions));
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
                ASSERT_EQ(1, result.GetData().size());

                //TODO: assert with LEAP-Cal
                ASSERT_EQ(expectedDirection(0), result.GetDirection()(0));
                ASSERT_EQ(expectedDirection(1), result.GetDirection()(1));

                if(!ToVector(expectedCalibration).isApprox(ToMatrix(result.GetData()[0]), THRESHOLD))
                {
                    std::cout << i+1 << "/" << expected.size() << " got:\n" << ToMatrix(result.GetData()[0]) << std::endl;
                }
                ASSERT_MEQD(ToVector(expectedCalibration), ToMatrix(result.GetData()[0]), THRESHOLD);
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
            }

            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = icrar::casalib::Integration(0, *ms, 0, ms->GetNumChannels(), ms->GetNumBaselines(), ms->GetNumPols());
            expectedIntegration.uvw = ToCasaUVWVector(ms->GetCoords());

            auto expectedConstants = icrar::cpu::Constants();
            expectedConstants.nbaselines = 5253;
            expectedConstants.channels = 48;
            expectedConstants.num_pols = 4;
            expectedConstants.stations = 102;
            expectedConstants.rows = 73542;
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

            ASSERT_EQ(5253, metadataOutput.GetAvgData().rows());
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
                if(impl == ComputeImplementation::cuda)
                {
                    throw not_implemented_exception(__FILE__, __LINE__);
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

        /**
         * @brief Get the Expected Calibration object validated the output of LEAP-Cal:ported
         * 
         * @return a vector of direction and antenna calibration pairs
         */
        std::vector<std::pair<casacore::MVDirection, std::vector<double>>> GetExpectedCalibration()
        {
            std::vector<std::pair<casacore::MVDirection, std::vector<double>>> output;

            output.push_back(std::make_pair(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>{
                     -2.2464544573217,
                    -2.25268450303484,
                -7.99149120051432e-15,
                    -2.28085413869221,
                    -2.08744690746661,
                    -2.22125774656327,
                    -2.20382331186683,
                    -2.40554001014245,
                    -2.27096989448176,
                -6.87072871626993e-14,
                -6.15740790257879e-15,
                 2.63137693418615e-15,
                    -2.16439457625486,
                     -2.3897695035554,
                     -2.2716491166884,
                  1.6498989383413e-14,
                    -2.07073747055065,
                    -2.28856306466009,
                 5.71849839907065e-15,
                -2.40839790898758e-15,
                    -2.28016224771222,
                    -2.07147552365793,
                    -1.87838332969977,
                    -2.39057324559791,
                    -2.29600595665444,
                    -2.40388249054584,
                   4.060290369018e-16,
                    -2.52918529989379,
                    -2.34075384347618,
                     -2.2892767092113,
                -2.34423108829509e-14,
                     -1.9558876056415,
                    -2.44963898452163,
                    -2.12263555830268,
                 4.69373000910169e-15,
                    -2.30536565919049,
                    -2.47010639500095,
                    -2.37249345637269,
                -1.08555112935755e-15,
                -5.23294516958133e-15,
                    -2.17929354820691,
                    -1.99625262506932,
                    -2.40344173464626,
                    -2.17372211262134,
                    -2.41345020116907,
                    -2.50496987109197,
                    -2.05730783389154,
                    -2.15390037834714,
                    -2.18517923557774,
                -2.64080014398127e-15,
                    -2.01392493966765,
                 2.58273985495831e-15,
                    -2.41479445667966,
                    -2.25785659610493,
                    -2.35873888395757,
                    -2.64751384819797,
                    -2.15799241882255,
                    -2.44401314174358,
                    -2.30029663807052,
                    -2.34932244296201,
                 1.66629066815671e-15,
                -7.47348082025638e-16,
                 2.02442206520529e-15,
                     -2.2808541386922,
                  1.6562617894823e-15,
                    -2.31760625824056,
                    -2.07054151161223,
                    -2.50480923642944,
                    -2.30942979363268,
                    -2.39583196069428,
                 2.65938657193246e-15,
                     -2.2808541386922,
                    -2.21756550893483,
                    -2.31222455300615,
                    -2.47836649053767,
                    -2.63890309657809,
                    -2.25697164898982,
                    -2.27733447618182,
                    -1.94476276445518,
                     -2.0380868490611,
                     -2.2808541386922,
                    -2.42947488209336,
                    -2.64561285040629,
                     -2.2889164888374,
                     -2.0316694609478,
                    -2.47850774670753,
                    -2.30144031235213,
                 2.03334843457002e-15,
                    -2.08746585296568,
                    -2.34181415903187,
                    -2.18743821568213,
                -9.29138108032271e-16,
                    -2.50242962332828,
                -1.81152753926548e-15,
                 -8.9425012597919e-16,
                -6.99522470962986e-16,
                    -2.15328340167755,
                    -2.12139513226519,
                     -2.0935906345951,
                     -2.4755455920179,
                    -2.41671057827871,
                    -2.09365369580635,
                    -2.28616156639316,
                    -2.53185104656145,
                    -2.37138924898075,
                    -2.08116998586843,
                    -2.15610534980253,
                    -2.17594355061059,
                    -1.93041684359688,
                    -2.15931587505569,
                    -2.62108897324237,
                -7.24120491176135e-16,
                    -2.12378496417687,
                    -2.00302740224217,
                    -2.48215638053328,
                 3.51767478388053e-16,
                    -2.16233343508881,
                    -2.40133712275933,
                    -2.49573627672211,
                    -2.46963935260233,
                    -2.24094017631893,
                    -2.52428226288756,
                    -2.22375587981504,
                    -2.14069083924131,
                    -2.58472310170016,
                    -2.24660970012184,
                    -2.27003192406292,
                    -2.38874378321015,
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>{
                    -2.24645445732163,
                    -2.39942919540456,
                 6.69158596401081e-15,
                    -2.28481441190287,
                    -2.36071034170287,
                    -2.45091296002682,
                    -2.35331836551524,
                    -2.18572950787915,
                    -2.28979626929608,
                -6.56472200462919e-15,
                 2.40678651664832e-14,
                -3.77746433807938e-15,
                    -2.32983994004414,
                    -2.16497303429026,
                    -2.46729959286749,
                 7.21105937992211e-15,
                    -2.28040219943039,
                    -2.13886267603946,
                -2.10713364899708e-14,
                -2.34301399657545e-14,
                    -2.29541940031801,
                    -2.32932986187068,
                    -2.26929009847917,
                    -2.10172255906846,
                    -2.30848204182455,
                    -2.55920646092714,
                -1.08225059808072e-15,
                    -2.40637806884057,
                    -2.16406930894269,
                    -2.26973545658513,
                 7.36860719418808e-15,
                    -2.43837456998623,
                    -2.27246833797271,
                    -2.28910589061989,
                -1.19692407777619e-14,
                    -2.24632379307519,
                    -2.11335880636053,
                    -2.58532102792776,
                 1.08170265560879e-15,
                -1.75017508378097e-15,
                     -2.7389784302316,
                    -2.38525708235067,
                     -2.0984297804373,
                    -2.24472552382048,
                    -2.19077314351311,
                     -2.6044176003912,
                    -2.23786271365648,
                    -2.41309387493137,
                    -1.94624231053842,
                -1.41590938945073e-15,
                    -2.23358903270702,
                -3.51738595556155e-15,
                    -2.13554435532368,
                    -2.26897275410573,
                    -2.45997008040407,
                    -2.14498368500592,
                    -2.36424651289069,
                    -2.24254472150386,
                    -2.20535305088269,
                    -2.36209304502385,
                 6.86916093171363e-16,
                -4.51404130285021e-15,
                -1.51520199852542e-15,
                    -2.28481441190288,
                -2.68647608917573e-16,
                    -2.17629611661759,
                    -2.30382245490081,
                    -2.15466078149971,
                    -1.87716185167249,
                    -2.19841490800391,
                 1.24280206790167e-15,
                    -2.28481441190288,
                    -2.51226983090434,
                    -2.23839161407637,
                    -2.27642919100971,
                    -2.19328234987411,
                    -2.32994013462281,
                    -2.01484135843394,
                    -2.21311326173144,
                    -1.99846411041896,
                    -2.28481441190288,
                    -2.34088572154223,
                    -2.15328716047391,
                    -2.33571455838866,
                    -2.27408033587233,
                    -2.40219280640502,
                    -2.03680939957709,
                -9.59254499997349e-16,
                    -2.20643087205977,
                    -2.56985294441806,
                    -2.23047362317861,
                -2.12270610282996e-15,
                     -2.2312220733232,
                -2.85483071900052e-15,
                 2.11059824409351e-16,
                -2.37377165464544e-15,
                    -2.50214886790546,
                    -2.51827092931728,
                     -2.2520884296852,
                    -2.18164274576045,
                    -2.17612415906236,
                    -2.15078390295606,
                    -2.79586505558423,
                    -2.16920838409695,
                    -2.35029924315989,
                    -2.36037545472698,
                    -2.33517451836348,
                    -2.53522302391866,
                    -2.42034763781368,
                    -2.51439293843604,
                    -2.37277722674156,
                -7.80397740261735e-16,
                    -2.00629451133654,
                    -2.26930431701593,
                    -2.43710061951244,
                 2.41205426871548e-16,
                    -2.11404413210047,
                    -2.00676879325475,
                    -2.18887819873357,
                    -2.17020930469314,
                    -2.27522069306038,
                    -2.42463805106543,
                    -2.09636210076875,
                    -2.31840998354117,
                    -2.38689969468074,
                    -2.34038745408167,
                    -2.15862604658096,
                    -2.22251656918798,
            }));

            return output;
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
                casacore::Array<std::int32_t> casaI;
                std::tie(casaA, casaI) = casalib::PhaseMatrixFunction(a1, a2, -1);
                Ad = ToMatrix(icrar::casalib::PseudoInverse(casaA));

                casacore::Matrix<double> casaA1;
                casacore::Array<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = casalib::PhaseMatrixFunction(a1, a2, 0);
                Ad1 = ToMatrix(icrar::casalib::PseudoInverse(casaA1));

                A = ToMatrix(casaA);
                I = ToVector(casaI);
                A1 = ToMatrix(casaA1);
                I1 = ToVector(casaI1);
            }
            if(impl == ComputeImplementation::cpu)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                std::tie(A, I) = cpu::PhaseMatrixFunction(ea1, ea2, -1);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, 0);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }
            if(impl == ComputeImplementation::cuda)
            {
                throw not_implemented_exception(__FILE__, __LINE__);
            }

            double TOLERANCE = 0.00001;

            // A
            const int aRows = 5152; 
            ASSERT_DOUBLE_EQ(aRows, A.rows());
            ASSERT_DOUBLE_EQ(128, A.cols());
            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(aRows-1, 125), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 126), TOLERANCE);
            EXPECT_NEAR(0.00, A(aRows-1, 127), TOLERANCE);

            // I
            const int nBaselines = 5152;
            ASSERT_EQ(nBaselines, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(2.00, I(1));
            EXPECT_EQ(3.00, I(2));
            //...
            EXPECT_EQ(5249, I(nBaselines-3));
            EXPECT_EQ(5251, I(nBaselines-2));
            EXPECT_EQ(-1, I(nBaselines-1));

            // Ad
            ASSERT_DOUBLE_EQ(128, Ad.rows());
            ASSERT_DOUBLE_EQ(nBaselines, Ad.cols());
            // EXPECT_NEAR(2.62531368e-15, Ad(0,0), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(2.04033520e-15, Ad(0,1), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(3.25648083e-16, Ad(0,2), TOLERANCE); // TODO: emergent
            // //...
            // EXPECT_NEAR(-1.02040816e-02, Ad(127,95), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-0.020408163265312793, Ad(127,96), TOLERANCE); // TODO: emergent
            // EXPECT_NEAR(-8.9737257304377696e-16, Ad(127,97), TOLERANCE); // TODO: emergent
            ASSERT_MEQD(A, A * Ad * A, TOLERANCE);

            //A1
            ASSERT_DOUBLE_EQ(102, A1.rows()); //-32=98, -split=102
            ASSERT_DOUBLE_EQ(128, A1.cols());
            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR(0.00, A1(97,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,126), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,127), TOLERANCE);

            //I1
            ASSERT_DOUBLE_EQ(102, I1.size());
            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(2.00, I1(1));
            EXPECT_DOUBLE_EQ(3.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(96.00, I1(95));
            EXPECT_DOUBLE_EQ(97.00, I1(96));
            EXPECT_DOUBLE_EQ(98.00, I1(97));

            //Ad1
            ASSERT_DOUBLE_EQ(102, Ad1.cols());
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
