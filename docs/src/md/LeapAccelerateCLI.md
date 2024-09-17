# Leap Accelerate CLI

leap-accelerate-cli is a command line interface to performing leap calibration that requires at least a measurement set and a set of directions to produce an antenna array calibration.

## Arguments

* `--config <path>` - config file path

* `--filepath <path>` - measurement set file path

* `--directions <array>` - directions for calibration in polar coordinates, e.g. `"[[1.2,0.8],[0.5,0.7]]"`

* `--output <path>` - Calibration output file path

* `--stations` - Overrides number of stations to use in the specified measurement set

* `--solutionInterval <[start,end,interval]>` - Sets the interval to generate solutions using numpy syntax. Additionally supports a single interval integer argument.

* `--referenceAntenna <integer>` - Selects the reference antenna index, default is the last antenna

* `--implementation <type>` - compute implementation type (cpu or cuda)

* `--useFileSystemCache <boolean>` - Whether filesystem caching is used between system calls

* `--autoCorrelations <boolean>` - Set to true if measurement set rows contain autocorrelations

* `--minimumBaselineThreshold <double>` - Minimum antenna baeline length in meters in the range 0.0 -> inf

* `--verbosity <integer>` - Logging verbosity (0=fatal, 1=error, 2=warn, 3=info, 4=debug, 5=trace), defaults to info

### Examples:

`LeapAccelerateCLI --help`

`LeapAccelerateCLI --config testdata/mwa_test.json`

## Logging

Log files are produced in the current working directory at ./log/leap_YYYY_MM_dd_{number}.log Log files rotate per day and store a maximum of 10MiB.

Logging is controlled by the verbosity setting.

## Config File

Config files can be specified via the --config argument to specify runtime arguments as an alternative to command line arguments.

Config files currently must be written in coformant JSON format.

### Schema

```
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Arguments",
    "definitions": {},
    "type": "object",
    "properties": {
        "filePath": { "type": "string" },
        "outputFilePath": { "type": "string" },
        "autoCorrelations": { "type": "boolean" },
        "useFileSystemCache": { "type": "boolean" },
        "minimumBaselineTheshold": { "type": "integer" },
        "solutionInterval": {
            "type": ["integer", "array"],
            "items": { "type": ["number", "null"] },
            "minItems": 3,
            "maxItems": 3
        },
        "referenceAntenna": { "type": "integer" },
        "computeImplementation": {
            "type": "string",
            "enum": ["cpu", "cuda"]
        },
        "verbosity": { "type": "string" },
        "directions": {
            "type": "array",
            "items": {
                "type": "array",
                "items": { "type": "number" }
            }
        }
    },
    "required": [ "filePath", "directions" ]
}
```

Note: Properties are not required when specified in as CLI arguments with a config file.

### Config File Example

```
{
    "filePath": "/testdata/ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.ms",
    "outputFilePath": "ska_low_cal",
    "computeImplementation": "cpu",
    "directions": [
        [-0.4606549305661674,-0.29719233792392513],
        [-0.4606549305661674,-0.29719233792392513]
    ],
    "solutionInterval": [0, None, 1],
    "verbosity": "info"
}
```
