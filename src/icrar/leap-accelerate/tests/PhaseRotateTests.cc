
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

        void RotateVisibilitiesTest()
        {
            MetaData metadata;
            std::vector<casacore::MVDirection> directions;
            std::queue<Integration> input;

            PhaseRotate(metadata, directions, input);
        }

        void PhaseRotateTest()
        {
            Integration integration;
            MetaData metadata;
            casacore::MVDirection direction;

            RotateVisibilities(integration, metadata, direction);
        }
    };

    TEST_F(PhaseRotateTests, PhaseRotate) { PhaseRotateTest(); }
}
