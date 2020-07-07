
#include <gtest/gtest.h>
#include <icrar/leap-accelerate/algorithm/PhaseRotate.h>

#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <vector>

using namespace casacore;

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

        void PhaseMatrixFunctionTest(bool useCuda)
        {
            const Array<int32_t> a1;
            const Array<int32_t> a2;
            int refAnt = 0;
            bool map = true;
            PhaseMatrixFunction(a1, a2, refAnt, map);
        }

        void RotateVisibilitiesTest(bool useCuda)
        {
            MetaData metadata;
            std::vector<casacore::MVDirection> directions;
            std::queue<Integration> input;

            PhaseRotate(metadata, directions, input);
        }

        void PhaseRotateTest(bool useCuda)
        {
            Integration integration;
            MetaData metadata;
            casacore::MVDirection direction;

            RotateVisibilities(integration, metadata, direction);
        }
    };

    TEST_F(PhaseRotateTests, RotateVisibilitiesTestCpu) { RotateVisibilitiesTest(false); }
    TEST_F(PhaseRotateTests, RotateVisibilitiesTestGpu) { RotateVisibilitiesTest(true); }
    TEST_F(PhaseRotateTests, PhaseRotateTestCpu) { PhaseRotateTest(false); }
    TEST_F(PhaseRotateTests, PhaseRotateTestGpu) { PhaseRotateTest(true); }
}
