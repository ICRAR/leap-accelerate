# Pull Submodules
git submodule update --init --recursive

# Add Submodules

git submodule add --name gtest-1.8.1 https://github.com/google/googletest.git external/gtest-1.8.1
git submodule add --name cmake-modules https://gitlab.com/ska-telescope/cmake-modules.git external/cmake-modules
git submodule add --name eigen-3.3.90 https://gitlab.com/libeigen/eigen.git external/eigen-3.3.90
git submodule add --name rapidjson-1.1.0 https://github.com/Tencent/rapidjson.git external/rapidjson-1.1.0