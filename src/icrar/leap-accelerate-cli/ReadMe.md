# leap-accelerate-cli {#leap-accelerate-cli}

leap-accelerate-cli is a command line interface to performing leap calibration.

## Arguments

* --filepath - MeasurementSet filepath

* --stations - Overrides number of stations

* --directions - directions for calibration in polar coordinates, e.g. "[[1.2,0.8],[0.5,0.7]]"

* --autocorrelations (true, false) - whether measurement sets contain autocorrelations

* (unsupported)--mwa-support (true, false) - negates baseline readings 

* --implementation (casa, cpu, cuda) - compute implementation type

* (unsupported)--output - output file for storing calibrations

* --config - config filepath

## Logging

log files are produced in the working directory in log/

## Config File (not supported)

JSON schema:
