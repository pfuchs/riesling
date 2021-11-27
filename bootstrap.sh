#!/bin/bash -eu

# This script will bootstrap an initial configure & build.
# On Linux and Mac a toolchain will be used which includes -march=native and
# ASan/UBSan support. Other platforms will use the default toolchain.

git submodule update --init --recursive

PLATFORM="$( uname -s )"
ARCH="$( uname -m )"
case "$PLATFORM" in
  Linux*)  TOOLCHAIN="../cmake/x64-linux-native.toolchain.cmake";;
  Darwin*)
    case "$ARCH" in
      x86_64*)  TOOLCHAIN="../cmake/x64-osx-native.toolchain.cmake";;
      arm64*)   TOOLCHAIN="../cmake/vcpkg/triplets/community/arm64-osx.cmake";;
    esac
    ;;
  *)       TOOLCHAIN="../cmake/vcpkg/scripts/buildsystems/vcpkg.cmake";;
esac
echo $TOOLCHAIN
# Use Ninja if available, otherwise CMake default
if [ -x "$( command -v ninja )" ]; then
  GEN="-GNinja"
else
  GEN=""
fi

mkdir -p build
cd build
cmake -S ../ $GEN \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
cmake --build .
