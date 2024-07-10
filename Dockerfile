FROM nvcr.io/nvidia/tritonserver:22.11-py3
RUN pip3 install --upgrade pip && pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install transformers
RUN pip3 install diffusers==0.7.2
RUN pip3 install transformers ftfy scipy accelerate
RUN pip3 install gradio
RUN pip3 install onnx onnxruntime
RUN python3 -m pip install --upgrade tensorrt \
    && python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
    && python3 -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com

RUN pip3 install nvtx && pip3 install cuda-python

RUN apt-get update
RUN apt-get -qq install cmake
RUN git clone https://github.com/NVIDIA/TensorRT.git \
    && cd TensorRT \
    && git submodule update --init --recursive
    
