
#include <boost/python.hpp>

namespace icrar
{
    struct UVWData
    {
        std::array<double, 3> uu;
        std::array<double, 3> vv;
        std::array<double, 3> ww;
        double interval;
        double exposure;
    };

    class PlasmaTM
    {
        int _num_channels;
        double _freq_start_hz;
        double _freq_inc_hz;
        int _num_stations;
        int _num_baselines;
        bool _is_autocorrelated;
        int _num_pols;
        int _phase_centre_ra_rad;
        int _phase_centre_dec_rad;
        int _uvw;
        int _interval;
        int _exposure;
        int _time_idx;

    public:
        PlasmaTM(boost::python::object& obj)
        {
            obj.attr("");
        }

        int GetNumChannels() const;
        double GetFreqStartHz() const;
        double GetFreqIncHz() const;
        int GetNumStations() const;
        int GetNumBaselines() const;
        bool IsAutocorrelated() const;
        int GetNumPols() const;
        std::pair<int, int> GetPhaseCentreRadecRad() const;
        void* GetNearestData(double time) const;
        std::pair<int, UVWData> GetMatchingData(double current_mjd_utc) const;
    };
} // namespace icrar