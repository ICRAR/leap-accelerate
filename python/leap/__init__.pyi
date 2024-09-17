"""
Linear Execision of the Atmosphere in Parallel
"""
from __future__ import annotations
import numpy
import typing
__all__ = ['BeamCalibration', 'BeamCalibrationVector', 'Calibration', 'ComputeImplementation', 'CppAwaitable', 'LeapCalibrator', 'MeasurementSet', 'Tensor3cd', 'Tensor3d', 'Tensor4cd', 'Tensor4d', 'Verbosity', 'cpu', 'cuda', 'debug', 'enable_log', 'error', 'fatal', 'info', 'trace', 'warn']
class BeamCalibration:
    def __init__(self, arg0: numpy.ndarray[numpy.float64[2, 1]], arg1: numpy.ndarray[numpy.float64[m, n]]) -> None:
        ...
    @property
    def antenna_phases(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Calibrated phases of input antennas
        """
    @property
    def direction(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        Beam ra and dec in radians
        """
class BeamCalibrationVector:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> BeamCalibrationVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> ...:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: BeamCalibrationVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: ...) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: BeamCalibrationVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: ...) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: BeamCalibrationVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: ...) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ...:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> ...:
        """
        Remove and return the item at index ``i``
        """
class Calibration:
    def __init__(self, arg0: int, arg1: int) -> None:
        ...
    @property
    def beam_calibrations(self) -> BeamCalibrationVector:
        ...
    @property
    def end_epoch(self) -> float:
        ...
    @property
    def start_epoch(self) -> float:
        ...
class ComputeImplementation:
    """
    Members:
    
      cpu
    
      cuda
    """
    __members__: typing.ClassVar[dict[str, ComputeImplementation]]  # value = {'cpu': <ComputeImplementation.cpu: 0>, 'cuda': <ComputeImplementation.cuda: 1>}
    cpu: typing.ClassVar[ComputeImplementation]  # value = <ComputeImplementation.cpu: 0>
    cuda: typing.ClassVar[ComputeImplementation]  # value = <ComputeImplementation.cuda: 1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CppAwaitable:
    def __await__(self) -> CppAwaitable:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> CppAwaitable:
        ...
    def __next__(self) -> None:
        ...
class LeapCalibrator:
    @typing.overload
    def __init__(self, arg0: ComputeImplementation) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: ComputeImplementation, arg1: Verbosity) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: int) -> None:
        ...
    def calibrate(self, ms_path: str, directions: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous], solution_interval: slice = ..., min_baseline_threshold: float = 0.0, callback: typing.Callable[[leap.Calibration], None]) -> None:
        ...
    def calibrate_to_file(self, ms_path: str, directions: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous], solution_interval: slice = ..., min_baseline_threshold: float = 0.0, output_path: typing.Any) -> None:
        ...
class MeasurementSet:
    def __init__(self, arg0: str) -> None:
        ...
    def read_coords(self, start_timestep: int, num_timesteps: int) -> Tensor3d:
        ...
    def read_vis(self, start_timestep: int, num_timesteps: int) -> Tensor4cd:
        ...
class Tensor3cd:
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    @property
    def numpy_view(self) -> numpy.ndarray[numpy.complex128]:
        ...
class Tensor3d:
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    @property
    def numpy_view(self) -> numpy.ndarray[numpy.float64]:
        ...
class Tensor4cd:
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None:
        ...
    @property
    def numpy_view(self) -> numpy.ndarray[numpy.complex128]:
        ...
class Tensor4d:
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int) -> None:
        ...
    @property
    def numpy_view(self) -> numpy.ndarray[numpy.float64]:
        ...
class Verbosity:
    """
    Members:
    
      fatal
    
      error
    
      warn
    
      info
    
      debug
    
      trace
    """
    __members__: typing.ClassVar[dict[str, Verbosity]]  # value = {'fatal': <Verbosity.fatal: 0>, 'error': <Verbosity.error: 1>, 'warn': <Verbosity.warn: 2>, 'info': <Verbosity.info: 3>, 'debug': <Verbosity.debug: 4>, 'trace': <Verbosity.trace: 5>}
    debug: typing.ClassVar[Verbosity]  # value = <Verbosity.debug: 4>
    error: typing.ClassVar[Verbosity]  # value = <Verbosity.error: 1>
    fatal: typing.ClassVar[Verbosity]  # value = <Verbosity.fatal: 0>
    info: typing.ClassVar[Verbosity]  # value = <Verbosity.info: 3>
    trace: typing.ClassVar[Verbosity]  # value = <Verbosity.trace: 5>
    warn: typing.ClassVar[Verbosity]  # value = <Verbosity.warn: 2>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def enable_log() -> None:
    ...
cpu: ComputeImplementation  # value = <ComputeImplementation.cpu: 0>
cuda: ComputeImplementation  # value = <ComputeImplementation.cuda: 1>
debug: Verbosity  # value = <Verbosity.debug: 4>
error: Verbosity  # value = <Verbosity.error: 1>
fatal: Verbosity  # value = <Verbosity.fatal: 0>
info: Verbosity  # value = <Verbosity.info: 3>
trace: Verbosity  # value = <Verbosity.trace: 5>
warn: Verbosity  # value = <Verbosity.warn: 2>
