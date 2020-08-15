/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#include <icrar/leap-accelerate/ms/utils.h>
#include <icrar/leap-accelerate/common/stream_extensions.h>


#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <icrar/leap-accelerate/tests/test_helper.h>
#include <gtest/gtest.h>

class MSUtilsTests : public testing::Test
{
    const double TOLERANCE = 0.0001;
    
    casacore::MeasurementSet ms;

public:
    MSUtilsTests()
    {

    }

    void SetUp() override
    {
            std::string filename = std::string(TEST_DATA_DIR) + "/1197638568-32.ms";
            ms = casacore::MeasurementSet(filename);
    }

    void TearDown() override
    {

    }

    void test_read_coords()
    {
        unsigned int start_row = 0;
        unsigned int num_baselines = 196;

        auto uu = std::vector<double>(num_baselines);
        auto ww = std::vector<double>(num_baselines);
        auto vv = std::vector<double>(num_baselines);

        icrar::ms_read_coords(ms,
            start_row,
            num_baselines,
            uu.data(),
            vv.data(),
            ww.data());

        auto expectedUu = GetExpectedUU();
        auto expectedVv = GetExpectedVV();
        auto expectedWw = GetExpectedWW();

        ASSERT_VEQD(expectedUu, uu, TOLERANCE);
        //ASSERT_VEQD(expectedVv, vv, TOLERANCE);
        //ASSERT_VEQD(expectedWw, ww, TOLERANCE);
    }

    void test_read_vis()
    {
        unsigned int start_row = 0;
        unsigned int start_channel = 0;

        unsigned int num_channels = 100;
        unsigned int num_baselines = 196;
        unsigned int num_pols = 3;

        auto visibilities = Eigen::Tensor<double, 3>(num_channels, num_baselines, num_pols);

        // ms_read_vis(ms,
        //     start_row,
        //     start_channel,
        //     num_channels,
        //     num_baselines,
        //     num_pols,
        //     "DATA",
        //     visibilities.data());
    }
private:
    std::vector<double> GetExpectedUU()
    {
        return std::vector<double>
        {
            0, -213.234574834057, -171.088059584787, 24.8832968787911, 75.9030116816415, 31.245674509303, 131.830257530978,
            401.958029973594, 400.081348645036, 297.869416811878, 409.233549684089, 384.100584883511, 584.903582556051, 609.706162817685,
            1006.99336853124, 912.026587829394, 138.127819921149, 100.479896312256, 130.26521840149, 80.8504201109847, 137.198986319818,
            733.459486795056, -869.239479202946, -380.056059687537, -338.953745657986, -175.475837483463, -56.9845612182013, -953.365320135896,
            -1272.29364709852, -1378.64886953924, -609.290318352972, -634.568642438031, -559.413564178326, -608.414011892997, -474.108230109677,
            -1134.52854769092, -508.174225421552, -402.897460266823, -568.196979552495, -539.171313039116, -562.682107740621, -591.616091685183,
            -403.097170078042, 59.4834065277692, -43.5280601988355, 133.316296154809, 499.616982347651, 597.122982982116, 673.835856340046,
            597.274536789626, 1899.07135788141, 1687.22102336035, 1530.8731095236, 1417.71549188342, 1337.04067536405, 1156.82291048554,
            1097.79627442992, 897.425527655708, 31.4666531727684, 223.908871230464, -104.71195303456, -180.429491317839, -1028.5247700206,
            -761.437176404545, -535.411523281294, -1094.18257680237, -1054.65386184022, -1884.30406453937, -2022.96371820072, -2214.56145126538,
            -2591.94620847363, -2594.18848000359, -2463.63501821009, -2927.70876308412, -3232.15662459114, -2987.02319024555, -1738.71402963224,
            -2053.31670359623, -2414.8051724614, -2272.28486972627, -2769.99609315878, -2710.39517135088, -2402.07729503966, -757.731497251641,
            -1500.48317313861, -1135.39745012848, -1487.88972981907, -1876.4044263105, -2259.72310840189, -1910.10104366261, -117.236467034634,
            -411.920936461338, -190.954541027847, -707.33161853491, -1025.51807814974, -780.496734909115, -1186.5352909369, -1553.06453862741,
            0, 42.14651524927, 238.117871712848, 289.137586515699, 244.48024934336, 345.064832365035, 615.192604807651,
            613.315923479093, 511.103991645935, 622.468124518147, 597.335159717568, 798.138157390108, 822.940737651742, 1220.2279433653,
            1125.26116266345, 351.362394755206, 313.714471146313, 343.499793235547, 294.084994945042, 350.433561153875, 946.694061629113,
            -656.004904368889, -166.82148485348, -125.719170823929, 37.7587373505943, 156.250013615856, -740.130745301839, -1059.05907226446,
            -1165.41429470518, -396.055743518915, -421.334067603974, -346.178989344268, -395.17943705894, -260.873655275619, -921.293972856864,
            -294.939650587495, -189.662885432766, -354.962404718438, -325.936738205058, -349.447532906564, -378.381516851126, -189.862595243985,
            272.717981361826, 169.706514635222, 346.550870988866, 712.851557181708, 810.357557816174, 887.070431174103, 810.509111623683,
            2112.30593271546, 1900.45559819441, 1744.10768435766, 1630.95006671748, 1550.27525019811, 1370.05748531959, 1311.03084926398,
            1110.66010248976, 244.701228006825, 437.143446064521, 108.522621799498, 32.8050835162184, -815.290195186547, -548.202601570488,
            -322.176948447237, -880.948001968315, -841.419287006162, -1671.06948970531, -1809.72914336667, -2001.32687643132, -2378.71163363957,
            -2380.95390516953, -2250.40044337603, -2714.47418825006, -3018.92204975708, -2773.78861541149, -1525.47945479818, -1840.08212876218, 
            -2201.57059762734, -2059.05029489222, -2556.76151832472, -2497.16059651682, -2188.8427202056, -544.496922417584, -1287.24859830456,
            -922.162875294422, -1274.65515498501, -1663.16985147644, -2046.48853356783, -1696.86646882855, 95.9981077994236, -198.686361627281,
            22.2800338062104, -494.097043700853, -812.283503315686, -567.262160075058, -973.300716102838, -1339.82996379335, 0
        };
    }
    std::vector<double> GetExpectedVV()
    {
        return std::vector<double>{};
    }
    std::vector<double> GetExpectedWW()
    {
        return std::vector<double>{};
    }
};

TEST_F(MSUtilsTests, test_read_coords) { test_read_coords(); }
TEST_F(MSUtilsTests, test_read_vis) { test_read_vis(); }
