
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
            std::string filename = "/mnt/d/dev/icrar/leap-accelerate/test_data/1197638568-32.ms";
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto msmc = std::make_unique<casacore::MSMainColumns>(*ms);
            casacore::Vector<double> time = msmc->time().getColumn();

            ASSERT_EQ(5020320156, time[0]);
            ASSERT_EQ(5020320156, time(casacore::IPosition(1,0)));

        }

        void TestReadFromFile()
        {
            std::string filename = "/mnt/d/dev/icrar/leap-accelerate/test_data/1197638568-32.ms";
            auto ms = std::make_unique<casacore::MeasurementSet>(filename);
            auto meta = MetaData(*ms);

            ASSERT_EQ(meta.init, true);
            ASSERT_EQ(meta.nantennas, 0);
            ASSERT_EQ(meta.channels, 48);
            ASSERT_EQ(meta.num_pols, 4);
            ASSERT_EQ(meta.stations, 128);
            ASSERT_EQ(meta.rows, 1);
            ASSERT_EQ(meta.freq_start_hz, 1.39195e+08);
            ASSERT_EQ(meta.freq_inc_hz, 640000);
            ASSERT_EQ(meta.solution_interval, 3601);

            ASSERT_NEAR(meta.phase_centre_ra_rad, 5.759587e-01, PRECISION);
            ASSERT_NEAR(meta.phase_centre_dec_rad, 1.047198e-01, PRECISION);
        }
    };

    TEST_F(MetaDataTests, TestMeasurementSet) { TestMeasurementSet(); }
    TEST_F(MetaDataTests, TestReadFromFile) { TestReadFromFile(); }
}
