
#include "utils.h"

using namespace casacore;

namespace icrar
{
    std::unique_ptr<MeasurementSet> ParseMeasurementSet(std::istream& input)
    {
        // don't skip the whitespace while reading
        std::cin >> std::noskipws;

        // use stream iterators to copy the stream to a string
        std::istream_iterator<char> it(std::cin);
        std::istream_iterator<char> end;
        std::string results = std::string(it, end);

        std::cout << results;

        return std::make_unique<MeasurementSet>(results);
    }

    std::unique_ptr<MeasurementSet> ParseMeasurementSet(std::filesystem::path& path)
    {
        auto ms = std::make_unique<MeasurementSet>();
        ms->openTable(path.generic_string());
        return ms;
    }
}