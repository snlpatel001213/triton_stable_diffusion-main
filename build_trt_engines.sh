#!/bin/bash 
trtexec --onnx=optimized_clip.onnx --saveEngine=clip.plan --fp16 --verbose --plugins=$PLUGIN_LIBS --timingCacheFile=clip.cache --buildOnly \
	--minShapes=input_ids:1x77 \
	--optShapes=input_ids:4x77 \
	--maxShapes=input_ids:4x77
mv clip.plan ./model_repository/clip/1/model.plan

trtexec --onnx=optimized_unet.onnx --saveEngine=unet.plan --fp16 --verbose --plugins=$PLUGIN_LIBS --timingCacheFile=unet.cache --buildOnly \
	--memPoolSize=workspace:80896 \
	--minShapes=sample:1x4x64x64,timestep:1x1,encoder_hidden_states:1x77x768 \
	--optShapes=sample:8x4x64x64,timestep:8x1,encoder_hidden_states:8x77x768 \
	--maxShapes=sample:8x4x64x64,timestep:8x1,encoder_hidden_states:8x77x768
mv unet.plan ./model_repository/unet/1/model.plan	

trtexec --onnx=optimized_vae.onnx --saveEngine=vae.plan --fp16 --verbose --plugins=$PLUGIN_LIBS --timingCacheFile=vae.cache --buildOnly \
	--minShapes=latent:1x4x64x64 \
	--optShapes=latent:4x4x64x64 \
	--maxShapes=latent:4x4x64x64
mv vae.plan ./model_repository/vae/1/model.plan