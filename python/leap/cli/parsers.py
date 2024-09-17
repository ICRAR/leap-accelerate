import ast
import logging
import numpy as np

from pathlib import Path
import astropy.units as u
from astropy.units import Quantity
import typer


def meters(value: str) -> Quantity:
    return Quantity(value, unit=u.m)


def seconds(value: str) -> Quantity:
    return Quantity(value, unit=u.s)


def az_el(value: str) -> list:
    res = ast.literal_eval(value)
    if not isinstance(res, list) or np.array(res).shape[1] != 2:
        raise typer.BadParameter(
            f"Expected Nx2 matrix e.g. [[az,el]], got {value}"
        )
    return res


def log_level(value: str) -> int:
    return logging._nameToLevel.get(value.upper())


def video_filepath(value: str) -> Path:
    path = Path(value)
    VIDEO_FORMATS = (
        ".mp4",
        ".mpeg4",
        ".gif",
        ".flv",
        ".avi",
        ".mov",
        ".png"
    )
    if path.suffix not in VIDEO_FORMATS:
        raise typer.BadParameter(
            f"Unknown video file extension '{path.suffix}' for '{path}', select one of {VIDEO_FORMATS}"
        )
    return path
