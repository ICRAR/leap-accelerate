
#include <memory>
#include <vector>

class channel_info
{
    double frequency;
    double width;

public:
    channel_info(double frequency, double width)
    : frequency(frequency)
    , width(width)
    {}
};

class band_data
{
    size_t channelCount;
    double frequencyStep;
    std::unique_ptr<std::vector<double>> channelFrequencies;

public:
    band_data()
    : channelCount(0)
    , channelFrequencies()
    , frequencyStep(0.0)
    { }
};