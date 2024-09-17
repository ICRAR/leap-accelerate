# Submodules

## Pull Submodules

Needs to be performed once for any checked out branch:

```sh
git submodule update --init --recursive
```

## Add Submodules

Example commands for how to add a new submodule:

```sh
git submodule add --name cmake-modules https://gitlab.com/ska-telescope/cmake-modules.git external/cmake-modules
```

```sh
git submodule add --name gtest-1.11.0 https://github.com/google/googletest.git external/gtest-1.11.0
```

```sh
git submodule add --name eigen-3.4.90 https://gitlab.com/libeigen/eigen.git external/eigen-3.4.90
```

```sh
git submodule add --name rapidjson-1.1.0 https://github.com/Tencent/rapidjson.git external/rapidjson-1.1.0
```
