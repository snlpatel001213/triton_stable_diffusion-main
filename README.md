# Triton_Stable_Diffusion

## Introduction
This demo is created to demonstrate the process of deploying stable diffusion based pipeline with triton inference server. In contrast to the implementation at https://github.com/triton-inference-server/server/tree/main/docs/examples/stable_diffusion, we will utilize the latest inference optimization techniques. Namely, all three essential DL models(CLIP, UNet, VAE) will be optimized via TRT + custom plugins

## Getting Started 


### Build Container Image
It's easy to get started, you can setup the server container with all required dependencies using the provided dockerfile. After clonening the repo, to build the container image

``` bash
cd Triton_Stable_Diffusion
docker build --network=host -t triton_stable_diffusion .
```
### Start Up A Server Container
```bash
docker run --gpus 'device=0' --network=host --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host -v ${PWD}:/workspace/ triton_stable_diffusion
```

### Obtain Model weights 
The V1.5 model weights can be obtained from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main. You should clone the huggingface repo and place the cloned repo under the models directory of this project


### Build Plugins and Trtexec
To build all the required TRT plugins and the trtexec tool
run
```bash
source build_trt_PluginsAndTools.sh
```

### Optimize ONNX Graph
To insert plugin nodes, optimize and export onnx models, run 
```python
python optimize_onnx.py
```
### Build TRT plans
To build the optimized TRT plans, and place them into the respective triton model repository
```bash
bash build_trt_engines.sh
```

### Start the Inference Server
After the above steps, we can start up the triton inference server via
```bash
tritonserver --model-repository=/workspace/model_repository
```

### Start Client and Run Inference
```bash
docker run -it --rm --network=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:22.11-py3-sdk
python client.py
```
## To-Dos
- [ ] Add benchmark data
- [ ] Add support for more pipelines(image2image, inpainting, depth2image)
- [ ] Add V2.0 support
- [ ] Further Optimize the pipeline performance
- [ ] Add a GUI support
