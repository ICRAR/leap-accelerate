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
    cpu::CalibrateResult ToCalibrateResult(casalib::CalibrateResult& result)
    {
        auto output_integrations = std::vector<std::queue<cpu::IntegrationResult>>();
        auto output_calibrations = std::vector<std::queue<cpu::CalibrationResult>>();

        for(auto& queues : result.first)
        {
            int index = output_integrations.size();
            output_integrations.push_back(std::queue<cpu::IntegrationResult>());
            while(!queues.empty())
            {
                auto& integrationResult = queues.front();
                output_integrations[index].emplace(
                    ToDirection(integrationResult.GetDirection()),
                    integrationResult.GetIntegrationNumber(),
                    integrationResult.GetData()
                );
                queues.pop();
            }
        }

        for(auto& queues : result.second)
        {
            int index = output_calibrations.size();
            output_calibrations.push_back(std::queue<cpu::CalibrationResult>());
            while(!queues.empty())
            {
                auto& calibrationResult = queues.front();
                output_calibrations[index].emplace(
                    ToDirection(calibrationResult.GetDirection()),
                    calibrationResult.GetData()
                );
                queues.pop();
            }
        }

        return std::make_pair(std::move(output_integrations), std::move(output_calibrations));
    }

    class PhaseRotateTests : public ::testing::Test
    {
        std::unique_ptr<icrar::MeasurementSet> ms;
        //std::unique_ptr<icrar::MeasurementSet> ms2;

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

            //std::string filename2 = std::string(TEST_DATA_DIR) + "/1197638568-split.ms";
            //ms2 = std::make_unique<icrar::MeasurementSet>(filename2, 126);
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

            std::vector<std::queue<cpu::IntegrationResult>> integrations;
            std::vector<std::queue<cpu::CalibrationResult>> calibrations;
            if(impl == ComputeImplementation::casa)
            {
                auto pair = icrar::casalib::Calibrate(*ms, directions, 3600);
                std::tie(integrations, calibrations) = ToCalibrateResult(pair);
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

            //auto expected = GetExpectedCalibration();

            // ASSERT_EQ(directions.size(), calibrations.size());
            // for(int i = 0; i < calibrations.size(); i++)
            // {
            //     casacore::MVDirection direction;
            //     std::vector<double> calibration;
            //     std::tie(direction, calibration) = expected[0];

            //     ASSERT_EQ(1, calibrations[i].size());
            //     const auto& result = calibrations[i].front();
                
            //     ASSERT_EQ(1, result.GetData().size());
            //     std::cout << result.GetData()[0] << std::endl;
            //     //ASSERT_MEQ(ToVector(calibration), ToMatrix(result.GetData()[0]), THRESHOLD); //TODO: assert with LEAP-Cal
            // }
        }

        void RotateVisibilitiesTest(ComputeImplementation impl)
        {
            const double THRESHOLD = 0.01;

            auto metadata = casalib::MetaData(*ms);
            metadata.stations = 126;
            auto direction = casacore::MVDirection(-0.4606549305661674, -0.29719233792392513);

            boost::optional<icrar::cpu::MetaData> metadataOptionalOutput;
            if(impl == ComputeImplementation::casa)
            {
                auto integration = casalib::Integration(
                    *ms,
                    0,
                    metadata.channels,
                    metadata.GetBaselines(),
                    metadata.num_pols);

                icrar::casalib::RotateVisibilities(integration, metadata, direction);
                metadataOptionalOutput = icrar::cpu::MetaData(metadata);
            }
            if(impl == ComputeImplementation::eigen)
            {
                auto integration = cpu::Integration(
                    *ms,
                    0,
                    metadata.channels,
                    metadata.GetBaselines(),
                    metadata.num_pols);

                auto metadatahost = icrar::cpu::MetaData(metadata, ToDirection(direction), integration.GetUVW());
                icrar::cpu::RotateVisibilities(integration, metadatahost);
                metadataOptionalOutput = metadatahost;
            }
            if(impl == ComputeImplementation::cuda)
            {
                auto integration = icrar::cpu::Integration(
                    *ms,
                    0,
                    metadata.channels,
                    metadata.GetBaselines(),
                    metadata.num_pols);

                auto metadatahost = icrar::cpu::MetaData(metadata, ToDirection(direction), integration.GetUVW());
                auto metadatadevice = icrar::cuda::DeviceMetaData(metadatahost);
                auto deviceIntegration = icrar::cuda::DeviceIntegration(integration);
                icrar::cuda::RotateVisibilities(deviceIntegration, metadatadevice);
                metadatadevice.ToHost(metadatahost);
                metadataOptionalOutput = metadatahost;
            }
            ASSERT_TRUE(metadataOptionalOutput.is_initialized());
            icrar::cpu::MetaData& metadataOutput = metadataOptionalOutput.get();

            // =======================
            // Build expected results
            // Test case generic
            auto expectedIntegration = icrar::casalib::Integration(*ms, 0, metadata.channels, metadata.GetBaselines(), metadata.num_pols);
            expectedIntegration.baselines = metadata.GetBaselines();
            expectedIntegration.uvw = ToCasaUVWVector(ms->GetCoords(0, metadata.GetBaselines()));

            //TODO: don't rely on eigen implementation for expected values
            auto expectedMetadata = icrar::cpu::MetaData(casalib::MetaData(*ms), ToDirection(direction), ToUVWVector(expectedIntegration.uvw));

            //Test case specific
            expectedMetadata.dd = Eigen::Matrix3d();
            expectedMetadata.dd <<
             0.46856701,  0.86068501, -0.19916391,
            -0.79210108,  0.50913781,  0.33668172,
             0.39117878,  0.0,         0.92031471;

            ASSERT_EQ(8001, expectedIntegration.baselines);
            ASSERT_EQ(4, expectedMetadata.GetConstants().num_pols);
            expectedMetadata.avg_data = Eigen::MatrixXcd::Zero(expectedIntegration.baselines, metadata.num_pols);


            // ==========
            // ASSERT
            auto cthreshold = std::complex<double>(0.001, 0.001);
            ASSERT_EQ(8001, metadataOutput.avg_data.rows());
            ASSERT_EQ(4, metadataOutput.avg_data.cols());
            ASSERT_EQCD(std::complex<double>(-18.733685278333724,  149.59337317943573), metadataOutput.avg_data(0,0), THRESHOLD);
            ASSERT_EQCD(std::complex<double>( 383.91613554954529, -272.36856329441071), metadataOutput.avg_data(0,1), THRESHOLD);
            ASSERT_EQCD(std::complex<double>(-32.724725462496281,  681.10801546275616), metadataOutput.avg_data(0,2), THRESHOLD);
            ASSERT_EQCD(std::complex<double>( 206.11409425735474, -244.23817884922028), metadataOutput.avg_data(0,3), THRESHOLD);
            
            // =============
            // ASSERT OBJECT
            ASSERT_EQ(expectedMetadata.GetConstants().num_pols, metadataOutput.avg_data.cols());
            // TODO: too much data to hard code
            //ASSERT_MDEQ(expectedMetadata, metadataOutput, THRESHOLD);
            //ASSERT_EQ(expectedIntegration, integration);
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

            output.push_back(std::make_pair(casacore::MVDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>{
                2.73980498e+00,  2.51482925e+00, -2.28497196e-16,  2.22914519e-15,
                2.60255235e+00,  2.49255324e+00,  2.72070986e+00,  2.79123560e+00,
                2.58884833e+00, -1.68528381e-15, -1.18505178e-16, -1.34205246e-15,
                2.44297662e+00,  2.47746594e+00,  2.77099212e+00, -6.80043189e-16,
                2.66318822e+00,  2.53448500e+00, -3.35545431e-15, -4.89613803e-15,
                2.74583720e+00,  2.49707877e+00,  2.61392899e+00,  2.56026463e+00,
                2.67147863e+00,  2.71611331e+00,  2.91391798e-15,  2.45093023e+00,
                2.60719923e+00,  2.52147177e+00, -3.11555776e-16,  2.58113320e+00,
                2.32675127e+00,  2.55556133e+00, -9.06930570e-16,  2.92140116e+00,
                2.50447595e+00,  2.50704029e+00, -1.43036643e-16, -3.27673789e-15,
                2.68224610e+00,  2.79431807e+00,  2.64255969e+00,  2.43478916e+00,
                2.77566491e+00,  2.35802640e+00,  2.40382889e+00,  2.38861154e+00,
                2.77985215e+00, -4.93762115e-16,  2.61430883e+00,  7.79571372e-16,
                2.75870940e+00,  2.43305947e+00,  2.53098551e+00,  2.61012527e+00,
                2.57354870e+00,  2.64478672e+00,  2.36311381e+00,  2.25146410e+00,
                7.27235492e-16,  8.62577338e-16,  1.71498409e-15, -3.70838155e-15,
                1.99251943e-15,  2.91655658e+00,  2.49654168e+00,  2.57531591e+00,
                2.49617488e+00,  2.56658098e+00,  1.05600566e-15,  1.67553045e-16,
                2.60601257e+00,  2.54422330e+00,  2.51501678e+00,  2.67644619e+00,
                2.66112036e+00,  2.85862785e+00,  2.72826695e+00,  2.42857556e+00,
                -1.11681682e-15,  2.38429862e+00,  2.62549005e+00,  2.42908640e+00,
                2.53418571e+00,  2.61898925e+00,  2.67908317e+00, -2.65356204e-16,
                2.60624929e+00,  2.70855280e+00,  2.43149108e+00, -4.11450512e-16,
                2.67753264e+00, -4.49878204e-16, -2.42694132e-15,  2.56557360e-17,
                2.65447931e+00,  2.64758474e+00,  2.73543649e+00,  2.62503750e+00,
                2.64936396e+00,  2.88468459e+00,  2.40485419e+00,  3.00854309e+00,
                2.63567772e+00,  2.65233163e+00,  2.54735002e+00,  3.05526237e+00,
                2.84107853e+00,  2.64775745e+00,  3.06883580e+00, -4.13666983e-19,
                2.62231240e+00,  2.59269180e+00,  2.92538806e+00, -5.02286686e-16,
                2.63153817e+00,  2.45829823e+00,  2.81125635e+00,  2.45622417e+00,
                2.59542388e+00,  2.42223246e+00,  2.79754317e+00,  2.64009344e+00,
                2.58537188e+00,  2.72767696e+00,  2.75064112e+00,  2.77773665e+00}));

            output.push_back(std::make_pair(casacore::MVDirection(-0.6207547100721282,-0.2539086572881469), std::vector<double>{
                2.29227686e+00,  2.58107099e+00, -4.15197135e-15,  4.62557369e-16,
                2.25031521e+00,  2.28442800e+00,  2.55607092e+00,  2.38497081e+00,
                2.29657739e+00, -9.13907846e-16, -2.19994480e-16, -1.41525506e-15,
                2.49923439e+00,  2.77332071e+00,  2.44097779e+00,  6.76484457e-15,
                2.29212629e+00,  2.35147205e+00, -1.51968193e-15,  3.10220928e-15,
                2.43655057e+00,  2.65334974e+00,  2.16069009e+00,  2.60029103e+00,
                2.28415215e+00,  2.53801926e+00, -1.71829059e-15,  2.33791792e+00,
                2.58966087e+00,  2.50921970e+00,  1.95034497e-15,  2.52914653e+00,
                2.53113654e+00,  2.12911568e+00, -9.66661956e-16,  2.18186827e+00,
                2.35727215e+00,  2.28872710e+00,  1.18234139e-15, -1.94577782e-15,
                2.36992926e+00,  2.56428979e+00,  2.22313558e+00,  2.37475030e+00,
                2.16864007e+00,  2.37641571e+00,  1.99868744e+00,  2.15047809e+00,
                2.21822817e+00, -1.72985736e-17,  2.21645495e+00, -1.70497880e-15,
                2.47082247e+00,  2.01266015e+00,  2.43893153e+00,  2.36771162e+00,
                2.53762970e+00,  2.48555273e+00,  2.18637343e+00,  2.20936616e+00,
                -2.10341048e-15, -8.43595505e-15, -3.68848056e-15,  1.10518757e-14,
                -2.19506884e-15,  2.37280888e+00,  2.37597420e+00,  2.65902413e+00,
                2.26534798e+00,  2.11622251e+00, -1.24090619e-15, -1.15028098e-15,
                2.55393191e+00,  2.40434008e+00,  2.32951890e+00,  2.32883599e+00,
                2.22634421e+00,  2.41637752e+00,  2.28045533e+00,  2.42036681e+00,
                2.51742526e-15,  2.20767381e+00,  2.43382411e+00,  2.71627220e+00,
                2.28460254e+00,  2.25979618e+00,  2.20060646e+00,  3.35037242e-16,
                2.48407502e+00,  2.12474858e+00,  2.39297448e+00, -4.91650851e-16,
                2.68716759e+00,  1.69940811e-16,  1.76476920e-15, -5.00534973e-16,
                2.23019392e+00,  2.27032590e+00,  2.36902525e+00,  2.58892101e+00,
                2.42158435e+00,  2.41585887e+00,  2.28042567e+00,  2.48711474e+00,
                2.17531039e+00,  2.43028956e+00,  2.22803199e+00,  2.35308803e+00,
                2.41774675e+00,  2.55054173e+00,  2.41055097e+00,  3.19223859e-15,
                2.53583798e+00,  2.14914610e+00,  2.65969161e+00, -4.76200124e-16,
                2.33428233e+00,  2.25809115e+00,  2.30449429e+00,  2.30305275e+00,
                2.41747045e+00,  1.90454755e+00,  2.24756303e+00,  2.91795834e+00,
                2.17792139e+00,  2.43606237e+00,  2.43585559e+00,  2.34511245e+00}));

            output.push_back(std::make_pair(casacore::MVDirection(-0.41958660604621867,-0.03677626900108552), std::vector<double>{
                2.15961409e+00,  2.43010640e+00, -3.92941215e-15, -3.30692390e-16,
                2.43961803e+00,  2.34011487e+00,  1.83813002e+00,  2.61907349e+00,
                2.28694894e+00, -3.38158377e-16,  1.13797162e-15, -5.88316180e-16,
                2.44994470e+00,  2.16905619e+00,  2.14304105e+00, -2.97923639e-16,
                2.24899968e+00,  2.21657597e+00, -2.14452387e-15, -2.01188650e-15,
                2.57246286e+00,  2.52244068e+00,  2.41106850e+00,  2.14138395e+00,
                2.46543622e+00,  1.92906801e+00,  1.32678441e-15,  2.44563320e+00,
                2.08130716e+00,  2.47380521e+00, -3.14955595e-15,  2.38969597e+00,
                2.41601517e+00,  2.59910274e+00, -2.05592346e-15,  2.54736098e+00,
                2.18549170e+00,  2.47287497e+00, -2.10034382e-15,  2.07380501e-15,
                2.18593987e+00,  2.70012676e+00,  2.12665593e+00,  2.45555759e+00,
                2.40393019e+00,  2.40339688e+00,  2.22581111e+00,  2.48627454e+00,
                2.13898754e+00, -1.58444807e-16,  2.59635616e+00,  1.10157977e-15,
                2.31612443e+00,  2.35806073e+00,  2.41953125e+00,  2.40187146e+00,
                2.19274419e+00,  2.34290386e+00,  2.34540367e+00,  2.46490644e+00,
                1.36406702e-15, -1.82436341e-15,  2.47422224e-15,  5.47850196e-16,
                -1.09080898e-15,  2.17493731e+00,  2.07792119e+00,  2.00945913e+00,
                2.58764983e+00,  2.52462751e+00, -1.08343859e-15, -1.61021543e-15,
                2.28981968e+00,  2.61844626e+00,  2.34774154e+00,  2.40410473e+00,
                2.55631157e+00,  2.34465041e+00,  2.13931309e+00,  2.20023476e+00,
                -1.92334074e-15,  2.10545469e+00,  2.05219073e+00,  2.16364745e+00,
                2.52044719e+00,  2.45576811e+00,  2.11598556e+00,  1.51319281e-15,
                2.26943401e+00,  2.23734000e+00,  2.40052406e+00,  1.29189882e-15,
                2.23862815e+00, -1.14340545e-15, -1.20677130e-15, -1.04009377e-15,
                2.24811245e+00,  2.00614798e+00,  2.29877181e+00,  2.38672439e+00,
                2.51098058e+00,  2.27363709e+00,  2.28381387e+00,  2.13600586e+00,
                2.62815903e+00,  2.29542580e+00,  2.42378627e+00,  2.07516173e+00,
                2.73276865e+00,  2.27148068e+00,  2.66222344e+00,  6.84767136e-17,
                2.35324916e+00,  2.44343339e+00,  2.23713084e+00, -5.80189593e-16,
                2.37581605e+00,  2.44039557e+00,  2.48988475e+00,  2.43860462e+00,
                2.41749738e+00,  2.39298297e+00,  2.02921070e+00,  2.44882630e+00,
                2.35046770e+00,  2.32184194e+00,  2.38342155e+00,  2.35396835e+00
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.41108685258900596,-0.08638012622791202), std::vector<double>{
                -1.14150786e+00, -1.42866427e+00, -5.64002246e-16, -2.97919352e-15,
                -1.27996348e+00, -1.21982103e+00, -1.07916653e+00, -1.59959662e+00,
                -1.50072502e+00, -2.39404510e-15, -1.99309181e-15,  2.55515830e-15,
                -1.23324292e+00, -1.33856588e+00, -1.18116854e+00, -1.71726369e-15,
                -1.24193886e+00, -1.47305174e+00, -4.21107266e-15,  7.07865409e-15,
                -1.61912818e+00, -1.46557326e+00, -1.56237883e+00, -1.34996420e+00,
                -1.13260380e+00, -1.79381109e+00, -3.92388174e-15, -1.58286723e+00,
                -1.37777500e+00, -1.52211694e+00,  9.96796048e-16, -1.36999744e+00,
                -1.38845301e+00, -1.70634927e+00,  1.90406423e-15, -1.31017343e+00,
                -1.20761121e+00, -1.71652323e+00, -1.56053577e-15, -1.90713636e-15,
                -1.56112400e+00, -1.38286129e+00, -1.28848016e+00, -1.35875430e+00,
                -1.53011410e+00, -1.76198471e+00, -1.35658226e+00, -1.50238020e+00,
                -1.47786353e+00, -2.24094765e-15, -1.24844691e+00,  1.11911755e-15,
                -1.16027298e+00, -1.38416384e+00, -1.39775203e+00, -1.23676148e+00,
                -1.31830877e+00, -1.75675142e+00, -1.13551836e+00, -1.61922030e+00,
                 1.38390808e-15,  1.52891945e-15,  9.53280875e-16, -1.41630035e-15,
                 2.89072975e-15, -1.24698069e+00, -1.01128837e+00, -1.11590303e+00,
                -9.86611270e-01, -1.45204600e+00, -2.50555438e-15,  5.01552208e-16,
                -1.25863147e+00, -1.01873323e+00, -1.42620866e+00, -1.36633173e+00,
                -1.55344135e+00, -1.16205677e+00, -1.27269213e+00, -1.40346837e+00,
                -8.14135680e-16, -1.60980676e+00, -1.54958098e+00, -1.53832165e+00,
                -1.17952872e+00, -1.09357292e+00, -1.53319119e+00, -4.95350503e-16,
                -1.29570368e+00, -1.51929822e+00, -1.44906694e+00, -1.12584941e-15,
                -1.42328992e+00,  5.65731064e-16,  1.20364090e-17,  7.16090749e-16,
                -1.18074360e+00, -1.08054997e+00, -1.22992963e+00, -1.18638199e+00,
                -1.38754297e+00, -1.37194756e+00, -1.34503704e+00, -1.38457849e+00,
                -1.41379934e+00, -8.94694186e-01, -1.28836392e+00, -1.18370022e+00,
                -1.49688822e+00, -1.25244391e+00, -1.37488936e+00, -8.52768518e-16,
                -1.28503633e+00, -1.51756820e+00, -1.13134634e+00,  4.12846576e-16,
                -1.49618499e+00, -1.53095957e+00, -1.47512361e+00, -1.62817227e+00,
                -1.16015806e+00, -1.52080387e+00, -1.46459141e+00, -1.54113200e+00,
                -1.69859179e+00, -1.41576944e+00, -1.48268031e+00, -1.32222914e+00
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.7782459495668798,-0.4887860989684432), std::vector<double>{
                3.00556588e+00,  2.82460549e+00,  2.86686026e-15, -6.25290438e-16,
                2.86353158e+00,  2.60470146e+00,  2.60311594e+00,  2.45812688e+00,
                2.60496223e+00, -3.03587418e-16,  5.57110793e-16,  1.69414286e-15,
                2.60174880e+00,  2.97731034e+00,  2.95841499e+00, -1.47694790e-15,
                2.61560965e+00,  2.75894744e+00,  3.18353146e-15, -1.56162130e-15,
                2.74850123e+00,  2.66975582e+00,  2.73392719e+00,  2.52702352e+00,
                2.73043243e+00,  2.65014480e+00,  3.48366707e-15,  2.80733252e+00,
                2.87379673e+00,  2.25702167e+00, -2.89943449e-16,  2.56885435e+00,
                2.93037448e+00,  2.65790091e+00, -2.22580347e-15,  2.52770387e+00,
                2.51124853e+00,  2.63382233e+00,  3.20665372e-17, -1.76930396e-15,
                2.73979364e+00,  2.75679716e+00,  2.87046784e+00,  2.63721330e+00,
                3.11205718e+00,  2.51246399e+00,  2.95760459e+00,  2.78493119e+00,
                2.69895454e+00,  3.69516475e-16,  2.75316117e+00,  1.87898450e-15,
                2.49607140e+00,  2.91383077e+00,  2.36924983e+00,  2.48624803e+00,
                2.82108604e+00,  2.69817280e+00,  2.39862088e+00,  2.85494040e+00,
                6.68348471e-16, -1.61654088e-15,  2.27120245e-15, -3.67451159e-15,
                -1.44369313e-17,  2.44907315e+00,  2.26704393e+00,  2.38046589e+00,
                2.87409849e+00,  3.04002458e+00, -6.79594943e-16,  4.00776009e-15,
                2.65190867e+00,  2.21456544e+00,  2.59895041e+00,  2.59923788e+00,
                2.79124971e+00,  2.90565681e+00,  2.61342962e+00,  2.56663244e+00,
                -1.29882596e-15,  2.39605080e+00,  2.52100385e+00,  2.61142317e+00,
                2.35234295e+00,  2.69175540e+00,  2.92705167e+00, -2.94046944e-16,
                2.72649159e+00,  2.79752665e+00,  2.61850061e+00,  1.12373980e-16,
                2.98950325e+00,  2.37864150e-16, -4.76225824e-16,  1.06666453e-15,
                2.62880976e+00,  2.58162280e+00,  2.69560309e+00,  2.44628393e+00,
                2.78945768e+00,  2.73131470e+00,  2.61246232e+00,  2.60397086e+00,
                2.30202006e+00,  2.47760399e+00,  2.97381504e+00,  2.52192883e+00,
                2.63825717e+00,  2.58472165e+00,  2.97917692e+00,  1.05624292e-15,
                2.73730736e+00,  2.66677596e+00,  2.54337216e+00, -1.43672532e-15,
                2.51334573e+00,  2.65472168e+00,  2.69768015e+00,  2.87478144e+00,
                2.76666383e+00,  2.55334176e+00,  2.88827160e+00,  2.55955509e+00,
                2.43129284e+00,  2.67393020e+00,  2.85783924e+00,  2.63251015e+00
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.17001324965728973,-0.28595644149463484), std::vector<double>{
                1.77283883e+00,  2.32963970e+00,  3.20198500e-15, -1.53774746e-16,
                2.10104785e+00,  1.92382610e+00,  1.91802384e+00,  1.84085612e+00,
                1.56314274e+00,  4.23907517e-16, -3.99820627e-15, -2.49158985e-15,
                1.72807587e+00,  1.91769414e+00,  1.61253692e+00, -2.01743195e-16,
                1.88434205e+00,  1.88244342e+00,  8.22982907e-15,  9.82702387e-17,
                1.78406945e+00,  1.50426566e+00,  1.82703898e+00,  1.79149029e+00,
                2.06491651e+00,  1.72252303e+00, -7.90829262e-16,  2.15420556e+00,
                2.14105181e+00,  1.98370596e+00, -5.42010986e-16,  1.98047633e+00,
                2.01343003e+00,  1.94403799e+00, -8.38182525e-16,  1.94267584e+00,
                2.07768485e+00,  1.80165033e+00, -1.43761637e-16, -1.40407461e-15,
                1.50081007e+00,  1.78961809e+00,  1.81950091e+00,  1.67118960e+00,
                1.94992217e+00,  1.91203136e+00,  1.79338226e+00,  1.69073783e+00,
                1.77772714e+00,  6.60355227e-16,  1.59342997e+00,  5.94630759e-16,
                1.77472251e+00,  2.21474464e+00,  1.84960282e+00,  2.07413942e+00,
                1.93260712e+00,  2.04380860e+00,  2.00155018e+00,  2.09220021e+00,
                -9.33768372e-16, -1.08823567e-16, -2.42249495e-15,  1.66805503e-15,
                5.14110406e-15,  1.88062686e+00,  1.91917944e+00,  1.75437771e+00,
                1.87142651e+00,  1.94493267e+00, -3.97485130e-15, -6.80111269e-16,
                1.67818419e+00,  1.91279927e+00,  1.65384172e+00,  2.10478954e+00,
                2.39625600e+00,  2.01962375e+00,  2.00620819e+00,  1.84056327e+00,
                -2.12988059e-15,  1.81304636e+00,  2.07154719e+00,  1.93735904e+00,
                1.98259315e+00,  1.69785669e+00,  1.84712704e+00, -1.54118989e-15,
                2.09159455e+00,  1.93659752e+00,  1.88767811e+00,  6.94839978e-16,
                2.05015970e+00,  1.02084758e-15, -9.26271776e-16,  4.50030924e-16,
                1.95773103e+00,  2.11563818e+00,  1.74795403e+00,  1.96193640e+00,
                2.15722180e+00,  2.16853069e+00,  1.83161587e+00,  1.92888622e+00,
                2.02345312e+00,  1.94985155e+00,  1.75096734e+00,  1.94785268e+00,
                1.77847268e+00,  1.73670987e+00,  1.95246299e+00, -4.83318133e-16,
                1.65730425e+00,  2.04345583e+00,  1.75167935e+00, -5.08381562e-18,
                1.66358843e+00,  1.77048617e+00,  1.90941605e+00,  2.00416358e+00,
                1.84065463e+00,  2.08354199e+00,  1.75284110e+00,  1.89325862e+00,
                1.66893037e+00,  1.78597660e+00,  1.98124703e+00,  1.65193364e+00
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.7129444556035118,-0.365286407171852), std::vector<double>{
                2.61675262e+00,  2.63570814e+00,  2.00360301e-15,  4.22838251e-16,
                3.09257547e+00,  3.16163895e+00,  2.85650720e+00,  2.67504006e+00,
                2.94895855e+00, -6.87555413e-16,  6.23734138e-16, -1.12008577e-15,
                2.81877444e+00,  2.98005035e+00,  2.70583818e+00, -2.86118205e-15,
                2.94878166e+00,  2.58468569e+00,  1.03739467e-15, -2.91007393e-15,
                2.88223296e+00,  2.45733439e+00,  2.98168685e+00,  2.45243512e+00,
                2.97230600e+00,  3.03764878e+00,  4.79116089e-16,  3.17536346e+00,
                2.71948913e+00,  2.57360988e+00, -1.01633598e-15,  2.86046452e+00,
                2.91839616e+00,  3.04960347e+00, -2.14786144e-15,  2.73989024e+00,
                2.88500637e+00,  2.68598449e+00, -8.93647349e-16, -1.28563482e-15,
                2.69127842e+00,  3.10569671e+00,  2.77144836e+00,  2.75121917e+00,
                3.18758908e+00,  3.03016583e+00,  2.92588772e+00,  2.62290148e+00,
                3.20604533e+00,  5.81554218e-16,  3.04068998e+00,  1.31225215e-15,
                2.76961004e+00,  3.01128517e+00,  2.83953931e+00,  2.72981567e+00,
                2.69148344e+00,  2.85269049e+00,  2.78252309e+00,  2.87845487e+00,
                7.44512756e-16,  3.27309405e-15,  3.37207433e-15, -4.04555815e-15,
                -2.20526792e-15,  2.93223443e+00,  3.19781148e+00,  2.95817478e+00,
                2.60808116e+00,  2.84035427e+00,  1.84706284e-15, -1.11964016e-15,
                2.57640840e+00,  3.24635568e+00,  2.87908916e+00,  2.69883687e+00,
                2.82716832e+00,  3.09292215e+00,  2.83653065e+00,  3.16830249e+00,
                1.47307741e-15,  2.72738985e+00,  2.87994805e+00,  2.86343315e+00,
                2.89508538e+00,  2.91431576e+00,  3.07129842e+00, -9.43178561e-16,
                3.03467341e+00,  2.74862029e+00,  2.86164248e+00,  6.18471348e-16,
                3.01998475e+00,  1.40041311e-15,  2.50477831e-15, -1.16287113e-16,
                3.12223121e+00,  2.68178483e+00,  2.99546597e+00,  2.83591107e+00,
                2.51821433e+00,  3.17222000e+00,  2.69979137e+00,  2.94235765e+00,
                2.78638958e+00,  2.63143823e+00,  3.10146003e+00,  2.69836638e+00,
                2.95853204e+00,  3.02061788e+00,  2.55741797e+00,  1.65690860e-15,
                3.03332517e+00,  2.74257129e+00,  2.53687371e+00, -6.22696516e-16,
                2.74640323e+00,  2.97563311e+00,  2.86068259e+00,  3.07687536e+00,
                3.06143617e+00,  2.96200252e+00,  2.67958469e+00,  2.55180951e+00,
                2.87445301e+00,  2.66008449e+00,  3.10381512e+00,  2.94629896e+00
            }));

            output.push_back(std::make_pair(casacore::MVDirection(-0.1512764129166089,-0.21161026349648748), std::vector<double>{
                2.00935507e+00,  2.41465371e+00, -2.33928209e-15,  2.18095798e-15,
                2.17769081e+00,  2.28819989e+00,  2.19272809e+00,  1.96725379e+00,
                2.03799016e+00, -7.79421331e-16, -4.29310076e-15,  7.08703364e-16,
                1.82339241e+00,  2.23669149e+00,  1.93996069e+00,  1.73070418e-15,
                2.06977343e+00,  1.99790052e+00,  6.61587680e-16,  1.19610532e-15,
                1.97958701e+00,  1.92177353e+00,  2.16240592e+00,  2.31980759e+00,
                2.34836976e+00,  2.30645182e+00,  1.46539085e-15,  1.69133061e+00,
                2.16904626e+00,  2.25134864e+00, -1.38655638e-15,  2.25407897e+00,
                2.16075722e+00,  1.93661482e+00,  1.61635331e-15,  1.77408642e+00,
                2.04650053e+00,  1.91605091e+00,  3.06944128e-17,  2.05819697e-16,
                2.15538116e+00,  2.30011221e+00,  2.03799372e+00,  2.05138828e+00,
                2.30370371e+00,  2.35390355e+00,  2.01142060e+00,  1.83003145e+00,
                1.76707035e+00, -2.82968638e-16,  1.65561330e+00,  2.08847495e-15,
                1.89995395e+00,  2.20680619e+00,  2.13532407e+00,  1.97865394e+00,
                2.06787000e+00,  2.00872947e+00,  1.98474654e+00,  2.22113652e+00,
                -8.24032998e-16, -1.06503847e-15, -3.03917194e-16,  2.45936362e-15,
                -1.25624786e-15,  2.23413433e+00,  2.00119573e+00,  1.84833263e+00,
                2.34559976e+00,  2.16399692e+00,  9.61916874e-18, -4.69489136e-16,
                2.23500298e+00,  2.31747691e+00,  2.16147630e+00,  1.79004634e+00,
                2.05057462e+00,  2.16559987e+00,  1.83985922e+00,  2.18012744e+00,
                7.83018759e-16,  2.38012917e+00,  1.92261976e+00,  2.10579916e+00,
                2.15530166e+00,  1.69697406e+00,  1.85230807e+00, -1.75216784e-16,
                2.22450630e+00,  1.88912718e+00,  1.89359028e+00,  7.08787467e-17,
                1.86513584e+00, -3.98722725e-16,  1.64959758e-15, -1.39548367e-16,
                1.85389863e+00,  1.71234463e+00,  1.98835161e+00,  2.14244318e+00,
                2.41054420e+00,  1.88175899e+00,  2.17384278e+00,  2.06668230e+00,
                2.15732576e+00,  1.73196127e+00,  1.92685818e+00,  1.95021670e+00,
                2.31997122e+00,  2.14474925e+00,  1.95053897e+00, -7.81409117e-16,
                2.13316104e+00,  1.93187795e+00,  1.94338387e+00,  8.15283928e-16,
                1.95693981e+00,  1.96317687e+00,  1.78094744e+00,  1.87909478e+00,
                1.82946237e+00,  2.09130040e+00,  2.07623703e+00,  2.48244875e+00,
                2.15323697e+00,  2.23156623e+00,  2.09554321e+00,  2.22075569e+00
            }));

            return output;
        }

        Eigen::MatrixXd GetExpectedA()
        {
            Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(4754, 128);
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
                std::tie(A1, I1) = cpu::PhaseMatrixFunction(ea1, ea2, 0, map);

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

            //TODO update
            //ASSERT_MEQ(GetExpectedA(), A, 0.001);
            //ASSERT_VEQI(GetExpectedI(), I, 0.001);
            //ASSERT_MEQ(GetExpectedA1(), A1, 0.001);
            //ASSERT_VEQI(GetExpectedI1(), I1, 0.001);
        }
    };

    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCasa) { PhaseMatrixFunction0Test(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCpu) { PhaseMatrixFunction0Test(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunction0TestCuda) { PhaseMatrixFunction0Test(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCasa) { PhaseMatrixFunctionDataTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCpu) { PhaseMatrixFunctionDataTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionDataTestCuda) { PhaseMatrixFunctionDataTest(ComputeImplementation::cuda); }

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCasa) { RotateVisibilitiesTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(ComputeImplementation::cuda); }
    
    TEST_F(PhaseRotateTests, PhaseRotateTestCasa) { PhaseRotateTest(ComputeImplementation::casa); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(ComputeImplementation::eigen); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCuda) { PhaseRotateTest(ComputeImplementation::cuda); }
}
