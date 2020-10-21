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
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = std::make_unique<icrar::MeasurementSet>(filename, 126);
            std::cout << std::setprecision(15);
        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 0.00000001;
            
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

                icrar::casalib::RotateVisibilities(integration, metadata, direction);
                integrationOptionalOutput = icrar::cpu::Integration(integration);
                metadataOptionalOutput = icrar::cpu::MetaData(metadata);
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
                icrar::cpu::RotateVisibilities(integration, metadatahost);

                integrationOptionalOutput = integration;
                metadataOptionalOutput = metadatahost;
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
            //icrar::cpu::Integration& integrationOutput = integrationOptionalOutput.get();

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
                1.29703834432374e-13,
                 0.4557868683214,
                                0,
                                0,
                0.563071677941698,
                0.499018556513769,
                0.154961290221771,
                0.11606459319824,
                0.174872594061634,
                                0,
                                0,
                                0,
                0.288086856628776,
                0.319549249212282,
                0.15746137478952,
                                0,
                0.374961376977472,
                0.161518728058037,
                                0,
                                0,
                0.338655979579404,
                0.458111284277748,
                0.175049402006757,
                0.272375959485712,
                0.710219672749268,
                0.452148949293648,
                                0,
                0.440754072237396,
                0.0821023396200791,
                0.149422034331564,
                                0,
                0.325329207910692,
                0.0590518714546786,
                0.512156561748077,
                                0,
                0.262673377452112,
                0.416127464913364,
                0.168318528755103,
                                0,
                                0,
                0.431412487957835,
                0.4717701767782,
                0.190298887758615,
                0.460587753636922,
                0.613052378763357,
                0.262275605896076,
                0.242652426206737,
                0.201430085339613,
                0.066335349126879,
                                0,
                0.203382183468439,
                                0,
                0.136812225579107,
                0.435382141481257,
                0.157002304330411,
                0.267340779502116,
                0.0304996474474444,
                0.132007312992534,
                0.170857269106293,
                0.122916110573544,
                                0,
                                0,
                                0,
                                0,
                                0,
                0.152707726876016,
                0.150654156164,
            -0.0074615071664309,
                0.126673517012974,
                0.120496882004718,
                                0,
                                0,
                -0.1502309652106,
                0.580175021735789,
                0.156367393529562,
                0.260399915138713,
                0.306625208353136,
                0.400891278250942,
                0.637263678948471,
                0.570260259016083,
                                0,
                0.302382586090506,
                0.589250075534229,
                0.391459056343483,
                0.184314558091003,
                0.217115796343502,
                0.5647229504126,
                                0,
            0.0724682107935088,
                0.393427783451258,
            -0.0943219081588267,
                                0,
                0.357294607534211,
                                0,
                                0,
                                0,
                0.317934742731497,
                0.403634579565768,
                0.400443088908263,
                0.28381108760486,
                0.219478660696612,
                0.359047028993927,
            0.00346386759106387,
                0.392831316663167,
                0.26635516024252,
                0.309422657048775,
                0.286868723896515,
                0.321243685518811,
                0.455339349138046,
                0.204544115401106,
                0.238660602937876,
                                0,
            0.0255097483863038,
                0.548658364216509,
                0.632357274866713,
                                0,
                0.631132130140091,
                0.260796152553571,
                0.677766862539649,
                0.238433847279804,
            -0.141906450149862,
                0.302055854600596,
                0.173612621086205,
                0.654376729356363,
                0.145670577393995,
                0.138864713864514,
                0.293068533822486,
                0.749333039155198,
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>{
                -1.45023693739608e-13,
                -0.372380935480676,
                                0,
                                0,
                -0.296543435052146,
                -0.0377163462601289,
                0.0539844639188343,
                -0.190610593880375,
                -0.106741763237148,
                                0,
                                0,
                                0,
                -0.319596296068871,
                -0.0612864023383246,
                0.0421475975194592,
                                0,
                -0.137010146846931,
                -0.0730096066479224,
                                0,
                                0,
                0.0187480024446911,
                -0.246600641812304,
                -0.285588780168926,
                -0.490435342895829,
                -0.0779783531412394,
                -0.217648847309205,
                                0,
                -0.332824746082458,
                -0.0443876680324879,
                -0.16095183319616,
                                0,
                -0.211404859986684,
                -0.552501174562647,
                -0.125943708743122,
                                0,
                0.106556694738386,
                -0.0730820533011332,
                -0.0799899410618461,
                                0,
                                0,
                -0.18986366300981,
                0.119681240198038,
                -0.0808547955010281,
                -0.321923066960288,
                    0.17973156382389,
                -0.54187636464451,
                -0.610883095999685,
                -0.401295140095826,
                0.00428709425166695,
                                0,
                0.0813413381354254,
                                0,
                -0.0845610640191043,
                -0.402443978341347,
                -0.272248891947365,
                0.0803320795981685,
                -0.158313834207248,
                -0.0862640825984842,
                -0.214774910383813,
                -0.376295499281725,
                                0,
                                0,
                                0,
                                0,
                                0,
                0.0650934482812801,
                -0.330924071223905,
                -0.178942763038737,
                -0.175702775150928,
                -0.217643368877848,
                                0,
                                0,
                -0.336023091698855,
                -0.0200879280354931,
                -0.25795230799457,
                -0.10999756064612,
                -0.222118783013718,
                -0.0129830856384392,
                -0.144880432642877,
                -0.26379748732335,
                                0,
                -0.551571319787739,
                -0.17675336844034,
                -0.332382068576546,
                -0.447095256218844,
                -0.0785677423463558,
                -0.111864034121067,
                                0,
                -0.230484057884808,
                -0.0616957835569586,
                -0.461484443857918,
                                0,
                -0.106112153518787,
                                0,
                                0,
                                0,
                -0.0471948651831711,
                -0.102873272048931,
                -0.00644231082028934,
                -0.101638246864899,
                -0.108779647295359,
                0.160390166552816,
                -0.236927293105129,
                -0.141480811166867,
                -0.357509256798716,
                0.0153064169157122,
                -0.147966857924222,
                0.147009259407973,
                -0.00817188234745458,
                -0.294827837660907,
                0.164924264944643,
                                0,
                -0.0597497931380759,
                -0.126700302028749,
                0.200531527129648,
                                0,
                0.0388331393735739,
                -0.236614691364464,
                -0.27049549072081,
                -0.287284026475242,
                -0.276492714312735,
                -0.418417088160443,
                -0.0295111760722802,
                -0.23776680784717,
                -0.0461877892320186,
                -0.160134625537264,
                -0.255784146464508,
                0.0949964505491674,
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.6207547100721282,-0.2539086572881469), std::vector<double>{
                -6.49050493004999e-14,
                    0.0812835903018461,
                                    0,
                                    0,
                    -0.214368961699482,
                    -0.140269989021146,
                    0.179930168534337,
                    -0.201932627266316,
                -0.000736127363613059,
                                    0,
                                    0,
                                    0,
                    0.120397805158596,
                    0.389992118956922,
                    -0.213758748674936,
                                    0,
                    -0.0726000032325249,
                    -0.0670053177771388,
                                    0,
                                    0,
                    -0.118168478315619,
                    0.128997642424678,
                    -0.0649143177248188,
                        0.11932041721434,
                    -0.0838782252414894,
                    0.330644647973898,
                                    0,
                    -0.0334822715265966,
                        0.29980614964414,
                    0.111700224992917,
                                    0,
                    0.0291737554446816,
                    0.123261831766712,
                    -0.132367553244237,
                                    0,
                    -0.113227882346593,
                    -0.24472332002041,
                    0.0744925080678464,
                                    0,
                                    0,
                    0.00169322440532871,
                    0.0525848536486448,
                    -0.098903820735338,
                    -0.0460186740652723,
                    0.145243218367971,
                    -0.0524703484320435,
                    -0.371277404759483,
                    -0.024157674122544,
                    -0.393864946450815,
                                    0,
                    -0.11265768322694,
                                    0,
                    -0.0871927489151674,
                    -0.157762196852741,
                    -0.063209223994096,
                    0.120705258292862,
                    0.133502291963931,
                    -0.0107651725848408,
                        -0.1141900399081,
                    -0.203469233943038,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                    0.0908684150121053,
                -7.56307886305985e-05,
                    0.170979115769999,
                    0.0485838318454808,
                    -0.302928969252086,
                                    0,
                                    0,
                    0.0362816506132535,
                    -0.129897498614016,
                    0.133910536755496,
                    -0.158835906378044,
                    -0.0549061145541076,
                    0.160522286370787,
                    0.011832072954729,
                    -0.0962458578808266,
                                    0,
                    -0.238710236237216,
                    0.00967552139850558,
                    0.267349226896155,
                    -0.107771827828061,
                    -0.283848538581059,
                    0.125404232518817,
                                    0,
                    0.0787530292596079,
                    -0.0964785356465745,
                    -0.0706672118334202,
                                    0,
                    -0.0176572695647608,
                                    0,
                                    0,
                                    0,
                    -0.224288734622073,
                    -0.169151652682866,
                    0.0757860009536973,
                    -0.0101606406881722,
                        0.03454778825745,
                    -0.184905566758737,
                    -0.346989542193591,
                    0.0555650481460059,
                    -0.332924078524026,
                    -0.35877164115928,
                    -0.0609714243545161,
                    0.0144289156834507,
                    -0.0719313410844673,
                    0.0838825759248927,
                    0.241575388345085,
                                    0,
                    0.218694639652044,
                    -0.321869113405006,
                    0.0955665374497641,
                                    0,
                    -0.0322041461290379,
                    -0.247486694380045,
                    -0.0856629235993993,
                    0.0175569125104602,
                    -0.109257891282038,
                    -0.379830129753801,
                    -0.14095356600388,
                    0.217740373289704,
                    -0.0957821526823217,
                    -0.0997426745624919,
                    -0.0015404845283804,
                    0.0539268039588294,
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.41958660604621867,-0.03677626900108552), std::vector<double>{
                -1.08634998754267e-13,
                0.155675953316536,
                                0,
                                0,
                0.104088207033636,
               -0.108279251253433,
               -0.294509526810762,
                0.281514244809755,
               -0.132130014524794,
                                0,
                                0,
                                0,
                 0.20931888295011,
              -0.0158765048956826,
                0.150425218845961,
                                0,
                0.232579345952286,
               0.0852522525412409,
                                0,
                                0,
                 0.16905474581781,
                0.289154054145234,
                0.277351745701442,
              0.00390543045665148,
                0.139032743561484,
               -0.420438305768883,
                                0,
                0.386327694334056,
               -0.159516891468512,
               0.0729943389865353,
                                0,
                0.113365921587549,
                0.279475470941734,
                0.284537306145519,
                                0,
                0.345407038178954,
              -0.0611928924773415,
                0.189998669972107,
                                0,
                                0,
               -0.146327460811618,
                0.342765250938491,
              -0.0669850398603644,
               0.0556437783684567,
                0.130548347975614,
              -0.0470226271077561,
              -0.0605677095413488,
                0.091216970592666,
               -0.111771903926096,
                                0,
                0.237497432154955,
                                0,
                0.188976487676663,
               -0.126653007559191,
                0.268371618814625,
              -0.0283535557649615,
               -0.068306697267671,
                0.109152587241898,
                0.193034223390015,
                 0.30860523424843,
                                0,
                                0,
                                0,
                                0,
                                0,
               0.0553124475828521,
                -0.16962701781678,
                -0.32069277090118,
               0.0696820961333839,
                0.239565841610411,
                                0,
                                0,
              -0.0361019794667446,
               0.0583290641785483,
               0.0195022473394344,
              -0.0584630808211077,
                0.303852494456187,
              -0.0360041838408487,
               -0.039851137001496,
              -0.0699808973145541,
                                0,
               0.0817232137274144,
               -0.100330034144144,
               -0.217225043573148,
                0.105669573119421,
                0.270351496523093,
              -0.0654783033864579,
                                0,
                0.162622406850469,
               -0.211042324285579,
              -0.0947281783798282,
                                0,
               -0.169759912376501,
                                0,
                                0,
                                0,
               -0.129131958707811,
             -0.00363777730816739,
               -0.259938079424316,
                0.237639853077853,
              -0.0996398005428287,
              -0.0441384161375784,
              -0.0233743711217396,
              -0.0306823352661789,
                0.195363086272967,
                0.128096801981772,
                0.232687033820471,
               -0.205042386071209,
                0.325588478040415,
                -0.15561709054922,
                0.297921503120075,
                                0,
                0.102173125057511,
               0.0294787682009598,
                  0.2009228291636,
                                0,
               0.0798340829941031,
                0.288844987199894,
                0.282362598893007,
                0.185983349650182,
             -0.00300891818683224,
                0.168213773456753,
               -0.334445999925388,
               0.0250816466910888,
               0.0358210368839591,
               0.0693180800320472,
                0.351156584107946,
                0.162889265914072,
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

        void PhaseMatrixFunctionDataTest(ComputeImplementation impl)
        {
            auto msmc = ms->GetMSMainColumns();

            //select the first epoch only
            casacore::Vector<double> time = msmc->time().getColumn();
            double epoch = time[0];
            int nEpochs = 0;
            for(size_t i = 0; i < time.size(); i++)
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
            Eigen::MatrixXd Ad;
            Eigen::MatrixXd A1;
            Eigen::VectorXi I1;
            Eigen::MatrixXd Ad1;

            if(impl == ComputeImplementation::casa)
            {
                casacore::Matrix<double> casaA;
                casacore::Array<std::int32_t> casaI;
                std::tie(casaA, casaI) = casalib::PhaseMatrixFunction(a1, a2, -1, map);
                Ad = ToMatrix(icrar::casalib::PseudoInverse(casaA));

                casacore::Matrix<double> casaA1;
                casacore::Array<std::int32_t> casaI1;
                std::tie(casaA1, casaI1) = casalib::PhaseMatrixFunction(a1, a2, 0, map);
                Ad1 = ToMatrix(icrar::casalib::PseudoInverse(casaA1));

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
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, 0, map);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }
            if(impl == ComputeImplementation::cuda)
            {
                auto ea1 = ToVector(a1);
                auto ea2 = ToVector(a2);
                std::tie(A, I) = cuda::PhaseMatrixFunction(ea1, ea2, -1, map);
                Ad = icrar::cpu::PseudoInverse(A);

                std::tie(A1, I1) = cuda::PhaseMatrixFunction(ea1, ea2, 0, map);
                Ad1 = icrar::cpu::PseudoInverse(A1);
            }

            const int nBaselines = 4753;
            ASSERT_DOUBLE_EQ(4754, A.rows()); //-32=4754, -split=5152
            ASSERT_DOUBLE_EQ(128, A.cols());
            ASSERT_DOUBLE_EQ(nBaselines+1, I.size());
            ASSERT_DOUBLE_EQ(128, Ad.rows());
            ASSERT_DOUBLE_EQ(4754, Ad.cols());

            double TOLERANCE = 0.00001;

            EXPECT_EQ(1.00, A(0,0));
            EXPECT_EQ(-1.00, A(0,1));
            EXPECT_EQ(0.00, A(0,2));
            //...
            EXPECT_NEAR(0.00, A(4753,125), TOLERANCE);
            EXPECT_NEAR(0.00, A(4753,126), TOLERANCE);
            EXPECT_NEAR(0.00, A(4753,127), TOLERANCE);

            ASSERT_EQ(4754, I.size());
            EXPECT_EQ(1.00, I(0));
            EXPECT_EQ(2.00, I(1));
            EXPECT_EQ(3.00, I(2));
            //...
            EXPECT_EQ(4849, I(4751));
            EXPECT_EQ(4851, I(4752));
            EXPECT_EQ(-1, I(4753));

            //TODO: Ad not identical
            EXPECT_NEAR(2.62531368e-15, Ad(0,0), TOLERANCE); // TODO: emergent
            EXPECT_NEAR(2.04033520e-15, Ad(0,1), TOLERANCE); // TODO: emergent
            EXPECT_NEAR(3.25648083e-16, Ad(0,2), TOLERANCE); // TODO: emergent
            //...
            EXPECT_NEAR(-1.02040816e-02, Ad(127,95), TOLERANCE); // TODO: emergent
            EXPECT_NEAR(-0.020408163265312793, Ad(127,96), TOLERANCE); // TODO: emergent
            EXPECT_NEAR(-8.9737257304377696e-16, Ad(127,97), TOLERANCE); // TODO: emergent

            ASSERT_DOUBLE_EQ(98, A1.rows()); //-32=98, -split=102
            ASSERT_DOUBLE_EQ(128, A1.cols());
            ASSERT_DOUBLE_EQ(98, I1.size());
            ASSERT_DOUBLE_EQ(128, Ad1.rows());
            ASSERT_DOUBLE_EQ(98, Ad1.cols());

            EXPECT_DOUBLE_EQ(1.0, A1(0,0));
            EXPECT_DOUBLE_EQ(-1.0, A1(0,1));
            EXPECT_DOUBLE_EQ(0.0, A1(0,2));
            //...
            EXPECT_NEAR(0.00, A1(97,125), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,126), TOLERANCE);
            EXPECT_NEAR(0.00, A1(97,127), TOLERANCE);

            EXPECT_DOUBLE_EQ(1.00, I1(0));
            EXPECT_DOUBLE_EQ(2.00, I1(1));
            EXPECT_DOUBLE_EQ(3.00, I1(2));
            //...
            EXPECT_DOUBLE_EQ(96.00, I1(95));
            EXPECT_DOUBLE_EQ(97.00, I1(96));
            EXPECT_DOUBLE_EQ(-1.00, I1(97));

            //TODO: Ad not identical
            EXPECT_DOUBLE_EQ(-9.8130778667735933e-18, Ad1(0,0)); // TODO: emergent
            EXPECT_DOUBLE_EQ(6.3742385976163974e-17, Ad1(0,1)); // TODO: emergent
            EXPECT_DOUBLE_EQ(3.68124219034074e-19, Ad1(0,2)); // TODO: emergent
            //...
            EXPECT_DOUBLE_EQ(5.4194040934156436e-17, Ad1(127,95)); // TODO: emergent
            EXPECT_DOUBLE_EQ(-1.0, Ad1(127,96)); // TODO: emergent
            EXPECT_DOUBLE_EQ(1.0, Ad1(127,97)); // TODO: emergent
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
    TEST_F(PhaseRotateTests, PhaseRotateTestCuda) { PhaseRotateTest(ComputeImplementation::cuda); }
}
