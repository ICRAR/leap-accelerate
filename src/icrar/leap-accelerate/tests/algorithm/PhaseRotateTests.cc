
#include <gtest/gtest.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/cuda/MetaDataCuda.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <casacore/casa/Quanta/MVDirection.h>

#include <vector>

namespace icrar
{
    class PhaseRotateTests : public ::testing::Test
    {
    protected:

        PhaseRotateTests() {

        }

        ~PhaseRotateTests() override
        {

        }

        void SetUp() override
        {

        }

        void TearDown() override
        {
            
        }

        void PhaseRotateTest(bool useCuda)
        {
            
            MetaData metadata;
            casacore::MVDirection direction;
            std::queue<Integration> input;
            std::queue<IntegrationResult> output_integrations;
            std::queue<CalibrationResult> output_calibrations;

            if(useCuda)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                icrar::cuda::PhaseRotate(metadatahost, direction, input, output_integrations, output_calibrations); //TODO: exception
            }
            else
            {
                icrar::cpu::PhaseRotate(metadata, direction, input, output_integrations, output_calibrations); //TODO: exception
            }
        }

        void RotateVisibilitiesTest(bool useCuda)
        {
            std::string filepath = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            auto ms = casacore::MeasurementSet(filepath);
            auto metadata = MetaData(ms);
            auto direction = casacore::MVDirection(-0.7129444556035118, -0.365286407171852);
            auto integration = Integration();

            if(useCuda)
            {
                auto metadatahost = icrar::cuda::MetaDataCudaHost(metadata);
                icrar::cuda::RotateVisibilities(integration, metadatahost, direction);
            }
            else
            {
                icrar::cpu::RotateVisibilities(integration, metadata, direction);
            }
        }

        void PhaseMatrixFunctionTest(bool useCuda)
        {
            const casacore::Vector<int32_t> a1;
            const casacore::Vector<int32_t> a2;
            int refAnt = 0;
            bool map = true;

            try
            {
                if(useCuda)
                {
                    icrar::cuda::PhaseMatrixFunction(a1, a2, refAnt, map);
                }
                else
                {
                    icrar::cpu::PhaseMatrixFunction(a1, a2, refAnt, map);
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

    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCpu) { PhaseRotateTest(false); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCuda) { PhaseRotateTest(true); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(false); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(true); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCpu) { PhaseMatrixFunctionTest(false); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCuda) { PhaseMatrixFunctionTest(true); }
}
