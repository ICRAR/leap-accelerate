

#pragma once

namespace casacore
{
    class MSField;
    class MDirection;
    class MeasurementSet;
}

//namespace wsclean
//{
    casacore::MDirection ZenithDirection(casacore::MeasurementSet& set);

    void getShift(casacore::MSField& fieldTable, double& dl, double& dm);
//}

