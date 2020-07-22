#include <utility>

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/config.h>

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#ifdef USE_CUDA
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>
#endif // USE_CUDA

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>

#include <casacore/casa/Quanta/MVDirection.h>

namespace icrar
{

    /// Templated version of operations implemented in both CPU and CUDA
    template<bool use_cuda=false>
    struct operations
    {

        template<typename ... Args>
        static void PhaseRotate(Args && ... args)
        {
            icrar::cpu::PhaseRotate(std::forward<Args>(args)...);
        }

        template<typename ... Args>
        static void RotateVisibilities(Args && ... args)
        {
            icrar::cpu::RotateVisibilities(std::forward<Args>(args)...);
        }

        template<typename ... Args>
        static void PhaseMatrixFunction(Args && ... args)
        {
            icrar::cpu::PhaseMatrixFunction(std::forward<Args>(args)...);
        }
    };

    #ifdef USE_CUDA
    template<>
    struct operations<true>
    {

        template<typename ... Args>
        static void PhaseRotate(Args && ... args)
        {
            icrar::cuda::PhaseRotate(std::forward<Args>(args)...);
        }

        template<typename ... Args>
        static void RotateVisibilities(Args && ... args)
        {
            icrar::cuda::RotateVisibilities(std::forward<Args>(args)...);
        }

        template<typename ... Args>
        static void PhaseMatrixFunction(Args && ... args)
        {
            icrar::cuda::PhaseMatrixFunction(std::forward<Args>(args)...);
        }
    };
    #endif // USE_CUDA

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

        template<bool useCuda>
        void PhaseRotateTest()
        {
            
            MetaData metadata;
            casacore::MVDirection direction;
            std::queue<Integration> input;
            std::queue<IntegrationResult> output_integrations;
            std::queue<CalibrationResult> output_calibrations;

            operations<useCuda>::PhaseRotate(metadata, direction, input, output_integrations, output_calibrations); //TODO: exception
        }

        template<bool useCuda>
        void RotateVisibilitiesTest()
        {
            Integration integration;
            MetaData metadata;
            casacore::MVDirection direction;

            operations<useCuda>::RotateVisibilities(integration, metadata, direction); //TODO: segfault with cpu
        }

        template<bool useCuda>
        void PhaseMatrixFunctionTest()
        {
            const casacore::Array<int32_t> a1; //TODO: populate
            const casacore::Array<int32_t> a2;
            int refAnt = 0;
            bool map = true;

            try
            {
                operations<useCuda>::PhaseMatrixFunction(a1, a2, refAnt, map);
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

    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCpu) { PhaseRotateTest<false>(); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCpu) { RotateVisibilitiesTest<false>(); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCpu) { PhaseMatrixFunctionTest<false>(); }

#ifdef USE_CUDA
    TEST_F(PhaseRotateTests, DISABLED_PhaseRotateTestCuda) { PhaseRotateTest<true>(); }
    TEST_F(PhaseRotateTests, DISABLED_RotateVisibilitiesTestCuda) { RotateVisibilitiesTest<true>(); }
    TEST_F(PhaseRotateTests, DISABLED_PhaseMatrixFunctionTestCuda) { PhaseMatrixFunctionTest<true>(); }
#endif // USE_CUDA
}
