
#include <gtest/gtest.h>
#include <icrar/leap-accelerate/algorithm/PhaseRotate.h>

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

        void PhaseRotate()
        {
            //icrar::helloworld::wave f;
            //EXPECT_EQ(f.greeting(), "I am waving hello");
        }
    };

    TEST_F(PhaseRotateTests, PhaseRotate) { PhaseRotate(); }
}
