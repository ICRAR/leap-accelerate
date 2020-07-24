
#include <gtest/gtest.h>

#include <icrar/leap-accelerate/MetaData.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>

#include <vector>

namespace icrar
{
    class MetaDataTests : public ::testing::Test
    {
        const double PRECISION = 0.0001;
    protected:
        MetaDataTests() {

        }

        ~MetaDataTests() override
        {

        }

        void SetUp() override
        {

        }

        void TearDown() override
        {

        }

        void TestMeasurementSet()
        {
            std::string filename = "../../../../../../../leap-accelerate/testdata/1197638568-32.ms";
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto msmc = std::make_unique<casacore::MSMainColumns>(*ms);
            casacore::Vector<double> time = msmc->time().getColumn();

            ASSERT_EQ(5020320156, time[0]);
            ASSERT_EQ(5020320156, time(casacore::IPosition(1,0)));

        }

        void TestReadFromFile()
        {
            std::string filename = "../../../../../../../leap-accelerate/testdata/1197638568-32.ms";
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto meta = MetaData(*ms);

            ASSERT_EQ(meta.init, true);
            ASSERT_EQ(4853, meta.nantennas);
            ASSERT_EQ(48, meta.channels);
            ASSERT_EQ(4, meta.num_pols);
            ASSERT_EQ(128, meta.stations);
            ASSERT_EQ(1, meta.rows);
            ASSERT_EQ(1.39195e+08, meta.freq_start_hz);
            ASSERT_EQ(640000, meta.freq_inc_hz);
            ASSERT_EQ(3601, meta.solution_interval);

            ASSERT_NEAR(5.759587e-01, meta.phase_centre_ra_rad, PRECISION);
            ASSERT_NEAR(1.047198e-01, meta.phase_centre_dec_rad, PRECISION);

            //TODO: verify these values
            ASSERT_EQ(4754, meta.A.shape()[0]); //4854?
            ASSERT_EQ(127, meta.A.shape()[1]); //128?
            ASSERT_EQ(127, meta.Ad.shape()[0]); //128?
            ASSERT_EQ(4754, meta.Ad.shape()[1]); //4854?
            ASSERT_EQ(4754, meta.I.shape()[0]); //4854?

            ASSERT_EQ(4854, meta.A1.shape()[0]);
            ASSERT_EQ(128, meta.A1.shape()[1]);
            ASSERT_EQ(128, meta.Ad1.shape()[0]);
            ASSERT_EQ(4854, meta.Ad1.shape()[1]);
            ASSERT_EQ(4854, meta.I1.shape()[0]);



        }
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestReadFromFile) { TestReadFromFile(); }
}
