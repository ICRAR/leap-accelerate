
#include "Calibrate.h"


#include <icrar/leap-accelerate/model/cpu/calibration/CalibrationCollection.h>
#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/common/config/Arguments.h>

namespace icrar
{
    void RunCalibration(const Arguments& args)
    {
        if(IsImmediateMode(args.GetStreamOutType()))
        {
            auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());

            auto outputCallback = [&](const cpu::Calibration& cal)
            {
                cal.Serialize(*args.CreateOutputStream(cal.GetStartEpoch()));
            };

            calibrator->Calibrate(
                outputCallback,
                args.GetMeasurementSet(),
                args.GetDirections(),
                args.GetSolutionInterval(),
                args.GetMinimumBaselineThreshold(),
                args.ComputeCal1(),
                args.GetReferenceAntenna(),
                args.GetComputeOptions());
        }
        else
        {
            auto calibrator = LeapCalibratorFactory::Create(args.GetComputeImplementation());

            std::vector<cpu::Calibration> calibrations;
            auto outputCallback = [&](const cpu::Calibration& cal)
            {
                calibrations.push_back(cal);
            };
            
            calibrator->Calibrate(
                outputCallback,
                args.GetMeasurementSet(),
                args.GetDirections(),
                args.GetSolutionInterval(),
                args.GetMinimumBaselineThreshold(),
                args.ComputeCal1(),
                args.GetReferenceAntenna(),
                args.GetComputeOptions());
            
            auto calibrationCollection = cpu::CalibrationCollection(std::move(calibrations));
            calibrationCollection.Serialize(*args.CreateOutputStream());
        }
    }
} // namespace icrar
