# Normal Maps for Stable Diffusion WebUI
This script is an addon for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that creates `normalmaps` from the generated images. Normal maps are helpful for giving flat textures a sense of depth with lighting.

To generate realistic normal maps from a single image, this script uses code and models from the [MiDaS](https://github.com/isl-org/MiDaS) repository by Intel ISL. See [https://pytorch.org/hub/intelisl_midas_v2/](https://pytorch.org/hub/intelisl_midas_v2/) for more info.

## Heavy work in-progress

## Examples
todo: update
[![screenshot](examples.png)](https://raw.githubusercontent.com/graemeniedermayer/stable-diffusion-webui-normalmap-script/main/normal_examples.png)

## Updates
This was forked from v0.1.9 depth maps
* v0.1.9 bugfixes
    * sd model moved to system memory while computing depthmap
    * memory leak/fragmentation issue fixed
    * recover from out of memory error


> 💡 To update, only replace the `depthmap.py` script, and restart.

## Install instructions
This installation should be same as depthmap.py
* Save `depthmap.py` into the `stable-diffusion-webui/scripts` folder.
* Clone the [MiDaS](https://github.com/isl-org/MiDaS) repository into `stable-diffusion-webui/repositories/midas` by running this command from the stable-diffusion-webui directory :
    * `git clone https://github.com/isl-org/MiDaS.git  repositories/midas`
* Restart AUTOMATIC1111

>Model `weights` will be downloaded automatically on first use and saved to /models/midas.

## Usage
Select the "DepthMap vX.X.X" script from the script selection box in either txt2img or img2img.
![screenshot](options.png)

The model can `Compute on` GPU and CPU, use CPU if low on VRAM. 

There are four models available from the `Model` dropdown : dpt_large, dpt_hybrid, midas_v21_small, and midas_v21. See the [MiDaS](https://github.com/isl-org/MiDaS) repository for more info. The dpt_hybrid model yields good results in our experience, and is much smaller than the dpt_large model, which means shorter loading times when the model is reloaded on every run.

Net size can be set with `net width` and `net height`, or will be the same as the input image when `Match input size` is enabled. There is a trade-off between structural consistency and high-frequency details with respect to net size (see [observations](https://github.com/compphoto/BoostingMonocularDepth#observations)). Large maps will also need lots of VRAM.

When enabled, `Invert DepthMap` will result in a depthmap with black near and white far.

Regardless of global settings, `Save DepthMap` will always save the depthmap in the default txt2img or img2img directory with the filename suffix '_depth'. Generation parameters are saved with the image if enabled in settings.

To see the generated output in the webui `Show DepthMap` should be enabled. When using Batch img2img this option should also be enabled.

When `Combine into one image` is enabled, the depthmap will be combined with the original image, the orientation can be selected with `Combine axis`. When disabled, the depthmap will be saved as a 16 bit single channel PNG as opposed to a three channel (RGB), 8 bit per channel image when the option is enabled.
> 💡 Saving as any format other than PNG always produces an 8 bit, 3 channel RGB image. A single channel 16 bit image is only supported when saving as PNG.

## FAQ

 * `Can I use this on existing images ?`
    - Yes, in img2img, set denoising strength to 0. This will effectively skip stable diffusion and use the input image. You will still have to set the correct size, and need to select `Crop and resize` instead of `Just resize` when the input image resolution does not match the set size perfectly.
 * `Can I run this on google colab ?`
    - You can run the MiDaS network on their colab linked here https://pytorch.org/hub/intelisl_midas_v2/

## Viewing

### For viewing on 2D displays

* There is the excellent [depthy](https://github.com/panrafal/depthy) by Rafał Lindemann. LIVE link : [https://depthy.stamina.pl/](https://depthy.stamina.pl/)
(Instructions: Drag the rgb image into the window, then select Load depthmap, and drag the depthmap into the dialog inside the window.) Generates GIF and video.

* Simple interactive depthmap viewer using three ([source](https://github.com/thygate/depthmap-viewer-three)). LIVE link : [https://thygate.github.io/depthmap-viewer-three](https://thygate.github.io/depthmap-viewer-three) (Instructions: Drag a combined-rgb-and-depth-horizontally image into the window to view it)

### For viewing on 3D devices

* Simple interactive depthmap viewer for lookingglass using three. LIVE link : [https://thygate.github.io/depthmap-viewer-three-lookingglass](https://thygate.github.io/depthmap-viewer-three-lookingglass) (Instructions: Drag a combined-rgb-and-depth-horizontally image into the window to view it)

* Unity3D project to view the depthmaps on lookingglass in realtime as images are generated. Leave a message in the discussion section if you want me to publish it too.

More updates soon ..
Feel free to comment and share in the discussions. 

## Acknowledgements

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
