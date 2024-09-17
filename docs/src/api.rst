.. _api:

###################
Leap Accelerate API
###################

A generic library containing core functionality for leap calibration.

Getting Started
===============

import the leap-accelerate cmake target and add the following include:

.. code-block:: cpp

    #include <icrar/leap-accelerate/algorithm/LeapCalibratorFactory.h>
    #include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

create a calibrator object using the factory method and an output callback:

.. code-block:: cpp

    ArgumentsValidated args;
    std::vector<cpu::Calibration> calibrations;
    auto outputCallback = [&](const cpu::Calibration& calibration)
    {
        calibrations.push_back(calibration);
    };
    
    LeapCalibratorFactory::Create(args.GetComputeImplementation())->Calibrate(
        outputCallback,
        args.GetMeasurementSet(),
        args.GetDirections(),
        args.GetSolutionInterval(),
        args.GetMinimumBaselineThreshold(),
        args.GetReferenceAntenna(),
        args.IsFileSystemCacheEnabled());


Compute Implementations
=======================

Compute functionality is split into two independant namespaces/implementations:

* cpu
* cuda

cpu
***

Provides computation using Eigen3 math libraries and layout that provides trivial
data layouts compatible with raw buffer copying to acceleration libraries and compatible 
with kernel compilers. See `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ for more details.

cuda
****

Provides Cuda classes and functions for nvidia hardware using cuda 9.x-11.x and cublas.

opencl
******

TBA - Provides OpenCL classes and functions for OpenCL supported hardware.

Modules
=======


* :ref:`core <dir_icrar_leap-accelerate_core>`
   - contains common classes and functions required by other modules
* :ref:`common <dir_icrar_leap-accelerate_common>`
   - contains leap specific files used by other components
* :ref:`model <dir_icrar_leap-accelerate_model>`
   - contains data structures for leap calibration
* :ref:`algorithm <dir_icrar_leap-accelerate_algorithm>`
   - contains utility classes and functions for performing leap calibration
* :ref:`math <dir_icrar_leap-accelerate_math>`
   - contains generic math extensions
* :ref:`ms <dir_icrar_leap-accelerate_ms>`
   - contains abstraction layers for measurement set objects
* :ref:`cuda <dir_icrar_leap-accelerate_cuda>`
   - contains cuda specific classes and helpers
