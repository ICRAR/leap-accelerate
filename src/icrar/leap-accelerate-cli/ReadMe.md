# leap-accelerate-cli

leap-accelerate-cli is a command line interface to performing leap calibration.

## Arguments

* --config - config file path

* --filepath - measurement set file path

* --output - Calibration output file path

* --directions - directions for calibration in polar coordinates, e.g. "[[1.2,0.8],[0.5,0.7]]"

* --stations - Overrides number of stations to use in the specified measurement set

* --implementation (cpu, cuda) - compute implementation type

* --useFileSystemCache (true, false) - Whether filesystem caching is used between system calls

* --autoCorrelations (true, false) - True if measurement set rows store autocorrelations

* --minimumBaselineThreshold (0.0 - inf) - Minimum antenna baeline length in meters

* --verbosity (true, false) - Verbosity (0=fatal, 1=error, 2=warn, 3=info, 4=debug, 5=trace), defaults to info

* (unsupported)--mwa-support (true, false) - negates baseline readings

## Logging

Log files are produced in the current working directory at ./log/leap_YYYY_MM_dd_{number}.log Log files rotate per day and store a maximum of 10MiB.

## Config File

Config files can be specified via the --config argument to specify runtime arguments as an alternative to command line arguments.

Config files currently must be written in coformant JSON format.

### Schema

filePath: string

outputFilePath: string?

computeImplementation: string?

autoCorrelations: boolean?

useFileSystemCache: bool?

minimumBaselineThreshold: double?

directions: [[number]]

verbosity: int|string?

### JSON Example

```
{
    "filePath": "../..//testdata/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.ms",
    "outputFilePath": "ska_low_cal",
    "readAutoCorrelations": false,
    "computeImplementation": "cpu",
    "directions": [
        [-0.4606549305661674,-0.29719233792392513],
        [-0.4606549305661674,-0.29719233792392513]
    ],
    "verbosity": "info"
}
```
