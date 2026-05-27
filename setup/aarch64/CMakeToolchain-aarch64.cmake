# CMake toolchain file for cross-compiling to aarch64-linux-gnu from an x86-64 host.
# Use with build-cross.sh or pass directly:
#   cmake -DCMAKE_TOOLCHAIN_FILE=setup/aarch64/CMakeToolchain-aarch64.cmake ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler — installed via: apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_STRIP        aarch64-linux-gnu-strip)

# Sysroot — use multiarch staging prefix if present, else system root
if(EXISTS "/usr/aarch64-linux-gnu")
    set(CMAKE_SYSROOT "/usr/aarch64-linux-gnu")
    set(CMAKE_FIND_ROOT_PATH "/usr/aarch64-linux-gnu")
endif()

# Only search target paths for libraries/includes; use host for programs
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# pkg-config must target aarch64 packages
set(ENV{PKG_CONFIG_PATH} "/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/local/lib/aarch64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_LIBDIR} "/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/local/lib/aarch64-linux-gnu/pkgconfig")
set(ENV{PKG_CONFIG_SYSROOT_DIR} "/")
