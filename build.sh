#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Change into the build directory
cd build

# Default build type to Release
build_type="Release"

# Check for argument "debug" to change build type
if [ "$1" == "debug" ]; then
  echo "Building in debug mode"
  build_type="Debug"
fi

# Run cmake with the specified build type
cmake -DCMAKE_BUILD_TYPE=$build_type ..

# Build the project with dbg if debug was specified
if [ "$1" == "debug" ]; then
  make dbg=1
else
  make
fi