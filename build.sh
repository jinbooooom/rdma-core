#!/bin/bash
set -e

# 解析命令行参数
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -d, --debug     构建调试版本（包含调试信息）"
    echo "  -r, --release   构建发布版本（不包含调试信息，默认）"
    echo "  -h, --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 构建发布版本"
    echo "  $0 --debug      # 构建调试版本"
    echo "  $0 --release    # 构建发布版本"
}

BUILD_TYPE="Release"

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "构建类型: $BUILD_TYPE"

SRCDIR=`dirname $0`
BUILDDIR="$SRCDIR/build"

mkdir -p "$BUILDDIR"

if hash cmake3 2>/dev/null; then
    # CentOS users are encouraged to install cmake3 from EPEL
    CMAKE=cmake3
else
    CMAKE=cmake
fi

if hash ninja-build 2>/dev/null; then
    # Fedora uses this name
    NINJA=ninja-build
elif hash ninja 2>/dev/null; then
    NINJA=ninja
fi

cd "$BUILDDIR"

# 根据构建类型设置不同的编译参数
if [ "$BUILD_TYPE" = "Release" ]; then
    CMAKE_EXTRA_ARGS="-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE:--O2 -s} \
                      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE:--O2 -s} \
                      -DCMAKE_EXE_LINKER_FLAGS_RELEASE=${CMAKE_EXE_LINKER_FLAGS_RELEASE:--Wl,-s} \
                      -DCMAKE_SHARED_LINKER_FLAGS_RELEASE=${CMAKE_SHARED_LINKER_FLAGS_RELEASE:--Wl,-s} \
                      -DCMAKE_INSTALL_DO_STRIP=TRUE"
else
    # Debug 模式，保留调试信息
    CMAKE_EXTRA_ARGS="-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG:--g -O0} \
                      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG:--g -O0}"
fi

if [ "x$NINJA" == "x" ]; then
    $CMAKE -DIN_PLACE=1 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        ${CMAKE_EXTRA_ARGS} \
        ${EXTRA_CMAKE_FLAGS:-} ..
    make
else 
    $CMAKE -DIN_PLACE=1 -GNinja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        ${CMAKE_EXTRA_ARGS} \
        ${EXTRA_CMAKE_FLAGS:-} ..
    $NINJA
fi
