
#include <icrar/leap-accelerate/chgcentre.h>
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
    };

    TEST_F(WaveTest, MethodWave)
    {
        //icrar::helloworld::wave f;
        //EXPECT_EQ(f.greeting(), "I am waving hello");
    }

    // Tests that Foo does Xyz.
    // TEST_F(WaveTest, DoesXyz) {
      // Exercises the Xyz feature of Foo.
    // }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
