
//#include <icrar/leap-accelerate/chgcentre.h>
#include <gtest/gtest.h>

namespace icrar
{
    class WaveTest : public ::testing::Test
    {
    protected:

        WaveTest() {

        }

        ~WaveTest() override
        {

        }

        void SetUp() override
        {

        }

        void TearDown() override
        {

        }

        void Wave()
        {
            //icrar::helloworld::wave f;
            //EXPECT_EQ(f.greeting(), "I am waving hello");
        }

        void FailTest()
        {
            ASSERT_EQ(1, 1);
        }
    };

    TEST_F(WaveTest, FailTest) { FailTest(); }
}
