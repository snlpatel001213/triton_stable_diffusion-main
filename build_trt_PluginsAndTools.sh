#!/bin/bash
export TRT_OSSPATH=/opt/tritonserver/TensorRT
export TRT_LIBPATH=/usr/lib/x86_64-linux-gnu
cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)
export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"
cd ../samples/trtexec
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$TRT_OSSPATH/build/out/"
ln -s "$TRT_OSSPATH/build/out/trtexec" /bin/trtexec 