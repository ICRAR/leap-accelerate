import sys
import os
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeExtensionBuilder(build_ext):

    def run(self):
        for ext in self.extensions:
            assert isinstance(ext, CMakeExtension)
            self.build_cmake_extension(ext)
        self.package_extensions()

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        ext_full_path = self.get_ext_fullpath(ext.name)
        dist_version = self.distribution.metadata.get_version()
        extdir = Path(ext_full_path).parent.resolve()
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DBUILD_DOCS=FALSE",
            "-DPYTHON_ENABLED=TRUE",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":  # pragma: no cover
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]
        cmake_args += ["-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_path("include"))]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), dist_version
        )
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name.rpartition(".")[-1]]
            + build_args,
            cwd=self.build_temp,
        )

    def package_extensions(self):
        for output in self.get_outputs():
            relative_extension = os.path.relpath(output, self.build_lib)
            shutil.copyfile(output, relative_extension)
            mode = os.stat(relative_extension).st_mode
            mode |= (mode & 0o444) >> 2
            os.chmod(relative_extension, mode)


if __name__ == "__main__":
    # setuptools here should only be used for building and
    # not installing. Use `poetry build` or `python -m build` from
    # the python project root directory.
    assert len(sys.argv) == 1, f"expected no arguments, got {sys.argv}"
    sys.argv.append("build")
    setup(
        name="leap",
        version='0.13.0',
        packages=['leap'],
        ext_modules=[
            # .suffix must match cmake module
            CMakeExtension(name="leap.LeapAccelerate", sourcedir="..")
        ],
        cmdclass={
            'build_ext': CMakeExtensionBuilder
        }
    )