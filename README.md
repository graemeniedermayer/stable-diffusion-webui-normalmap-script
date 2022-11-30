# Normal Maps for Stable Diffusion WebUI
This script is an addon for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that creates `normalmaps` from the generated images. Normal maps are helpful for giving flat textures a sense of depth with lighting.

To generate realistic normal maps from a single image, this script uses code and models from the [MiDaS](https://github.com/isl-org/MiDaS) repository by Intel ISL. See [https://pytorch.org/hub/intelisl_midas_v2/](https://pytorch.org/hub/intelisl_midas_v2/) for more info.

## Heavy work in-progress
Things to add/fix:
* Implement bilateral filtering
* Naming convention doesn't include n_iter
* I have not tested everything yet.

## Examples

Lighting

[![lighting](examples/rocks.gif)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/examples/rocks.gif?raw=true)

Moving Camera

[![moving camera](examples/movcam.gif)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/examples/movcam.gif?raw=true)

example of blurring (with and without)

[![without gaus blur](examples/nogaus.gif)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/examples/nogaus.gif?raw=true)[![with gaus blur](examples/gaus.gif)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/examples/gaus.gif?raw=true)

[![screenshot](examples.jpg)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/examples.jpg?raw=true)

## Updates
This was forked from v0.1.9 depth maps
* v0.1.1 bugfixes
    * sd model moved to system memory while computing depthmap
    * memory leak/fragmentation issue fixed
    * recover from out of memory error


> 💡 To update, only replace the `normalmap.py` script, and restart.

## Install instructions
In the WebUI, in the Extensions tab, in the Install from URL subtab, enter this repository https://github.com/graemeniedermayer/stable-diffusion-webui-normalmap-script and click install.
The midas repository will be cloned to /repositories/midas

Model weights will be downloaded automatically on first use and saved to /models/midas.


## Usage
Select the "NormalMap vX.X.X" script from the script selection box in either txt2img or img2img.
![screenshot](options.jpg)

The model can `Compute on` GPU and CPU, use CPU if low on VRAM. 

There are four models available from the `Model` dropdown : dpt_large, dpt_hybrid, midas_v21_small, and midas_v21. See the [MiDaS](https://github.com/isl-org/MiDaS) repository for more info. The dpt_hybrid model yields good results in our experience, and is much smaller than the dpt_large model, which means shorter loading times when the model is reloaded on every run.

Net size can be set with `net width` and `net height`, or will be the same as the input image when `Match input size` is enabled. There is a trade-off between structural consistency and high-frequency details with respect to net size (see [observations](https://github.com/compphoto/BoostingMonocularDepth#observations)). Large maps will also need lots of VRAM.

When enabled, `Invert NormalMap` will result in a normalmap that's calculated from a flipped depthmap.

Regardless of global settings, `Save NormalMap` will always save the normalmap in the default txt2img or img2img directory with the filename suffix '_depth'. Generation parameters are saved with the image if enabled in settings.

To see the generated output in the webui `Show NormalMap` should be enabled. When using Batch img2img this option should also be enabled.

When `Combine into one image` is enabled, the normalmap will be combined with the original image, the orientation can be selected with `Combine axis`. (TODO: might of broken this while converting).

## FAQ

 * `Can I use this on existing images ?`
    - Yes, in img2img, set denoising strength to 0. This will effectively skip stable diffusion and use the input image. You will still have to set the correct size, and need to select `Crop and resize` instead of `Just resize` when the input image resolution does not match the set size perfectly.
 * `Can I run this on google colab ?`
    - You can run the MiDaS network on their colab linked here https://pytorch.org/hub/intelisl_midas_v2/ . 

## Acknowledgements

This is a modification of an awesome [depthmap repo](https://github.com/thygate/stable-diffusion-webui-depthmap-script) by thygate.

This project uses code and information from following papers, from the repository [github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS) :
```
@ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}
```

Dense Prediction Transformers, DPT-based model :

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}
```
