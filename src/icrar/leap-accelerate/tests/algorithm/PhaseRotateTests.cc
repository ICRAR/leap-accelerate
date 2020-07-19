
#include <gtest/gtest.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/MetaData.h>
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
                icrar::cuda::PhaseRotate(metadata, direction, input, output_integrations, output_calibrations); //TODO: exception
            }
            else
            {
                icrar::cpu::PhaseRotate(metadata, direction, input, output_integrations, output_calibrations); //TODO: exception
            }
        }

        void RotateVisibilitiesTest(bool useCuda)
        {
            Integration integration;
            MetaData metadata;
            casacore::MVDirection direction;

            if(useCuda)
            {
                icrar::cuda::RotateVisibilities(integration, metadata, direction);
            }
            else
            {
                icrar::cpu::RotateVisibilities(integration, metadata, direction); //TODO: segfault
            }
        }

        void PhaseMatrixFunctionTest(bool useCuda)
        {
            const casacore::Array<int32_t> a1; //TODO: populate
            const casacore::Array<int32_t> a2;
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
                FAIL() << "Excpected std::invalid_argument";
            }
        }
    };

    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(false); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCuda) { PhaseRotateTest(true); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(false); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCuda) { RotateVisibilitiesTest(true); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionTestCpu) { PhaseMatrixFunctionTest(false); }
    TEST_F(PhaseRotateTests, PhaseMatrixFunctionTestCuda) { PhaseMatrixFunctionTest(true); }
}
