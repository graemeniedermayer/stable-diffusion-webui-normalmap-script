# Author: thygate with modifications from graemeniedermayer
# https://github.com/thygate/stable-diffusion-webui-depthmap-script

import modules.scripts as scripts
import gradio as gr

from modules import processing, images, shared, sd_samplers, devices
from modules.processing import create_infotext, process_images, Processed
from modules.shared import opts, cmd_opts, state, Options
from PIL import Image
from pathlib import Path

import sys
import torch, gc
import torch.nn as nn
import cv2
import requests
import os.path
import contextlib
import matplotlib.pyplot as plt
import numpy as np

path_monorepo = Path.joinpath(Path().resolve(), "repositories\BoostingMonocularDepth")
sys.path.append(str(path_monorepo))

# AdelaiDepth imports
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present

from torchvision.transforms import Compose, transforms

#midas imports
from repositories.midas.midas.dpt_depth import DPTDepthModel
from repositories.midas.midas.midas_net import MidasNet
from repositories.midas.midas.midas_net_custom import MidasNet_small
from repositories.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet

import numpy as np
#import matplotlib.pyplot as plt

scriptname = "NormalMap v0.1.2"

class Script(scripts.Script):
	def title(self):
		return scriptname

	def show(self, is_img2img):
		return True

	def ui(self, is_img2img):

		with gr.Row():
			compute_device = gr.Radio(label="Compute on", choices=['GPU','CPU'], value='GPU', type="index")
			model_type = gr.Dropdown(label="Model", choices=['dpt_large','dpt_hybrid','midas_v21','midas_v21_small','res101'], value='dpt_large', type="index", elem_id="model_type")
		with gr.Row():
			net_width = gr.Slider(minimum=64, maximum=2048, step=64, label='Net width', value=384)
			net_height = gr.Slider(minimum=64, maximum=2048, step=64, label='Net height', value=384)
		with gr.Row():	
			match_size = gr.Checkbox(label="Match input size",value=True)
			scale_depth = gr.Slider(minimum=0.1, maximum=3, step=0.1, label='pre-scale depth', value=1)

		with gr.Row():
			pre_gaussian_blur = gr.Checkbox(label="smooth before calculating normals",value=False)
			pre_gaussian_blur_kernel = gr.Slider(minimum=1, maximum=31, step=2, label='pre-smooth kernel size', value=3)

		with gr.Row():
			sobel_gradient = gr.Checkbox(label="sobel gradient",value=True)
			sobel_kernel = gr.Slider(minimum=1, maximum=31, step=2, label='sobel kernel size', value=3)

		with gr.Row():
			post_gaussian_blur = gr.Checkbox(label="smooth after calculating normals",value=False)
			post_gaussian_blur_kernel = gr.Slider(minimum=1, maximum=31, step=2, label='post-smooth kernel size', value=3)

		with gr.Row():
			combine_output = gr.Checkbox(label="Combine into one image.",value=False)
			combine_output_axis = gr.Radio(label="Combine axis", choices=['Vertical','Horizontal'], value='Horizontal', type="index")

		with gr.Row():
			invert_normal = gr.Checkbox(label="invert normal map (pre-flip depth map)",value=False)
			save_normal = gr.Checkbox(label="Save normalMap",value=True)
			show_normal = gr.Checkbox(label="Show normalMap",value=True)
		
		#would passing a dictionary be more readable?
		return [compute_device, model_type, net_width, net_height, match_size, invert_normal, save_normal, show_normal, combine_output, combine_output_axis, sobel_gradient, sobel_kernel, pre_gaussian_blur, pre_gaussian_blur_kernel, post_gaussian_blur, post_gaussian_blur_kernel, scale_depth]

	def run(self, p, compute_device, model_type, net_width, net_height, match_size, invert_normal, save_normal, show_normal, combine_output, combine_output_axis, sobel_gradient, sobel_kernel, pre_gaussian_blur, pre_gaussian_blur_kernel, post_gaussian_blur, post_gaussian_blur_kernel, scale_depth):

		def download_file(filename, url):
			print("Downloading depth model weights to %s" % filename)
			with open(filename, 'wb') as fout:
				response = requests.get(url, stream=True)
				response.raise_for_status()
				# Write response data to file
				for block in response.iter_content(4096):
					fout.write(block)

		def scale_torch(img):
			"""
			Scale the image and output it in torch.tensor.
			:param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
			:param scale: the scale factor. float
			:return: img. [C, H, W]
			"""
			if len(img.shape) == 2:
				img = img[np.newaxis, :, :]
			if img.shape[2] == 3:
				transform = transforms.Compose([transforms.ToTensor(),
												transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
				img = transform(img.astype(np.float32))
			else:
				img = img.astype(np.float32)
				img = torch.from_numpy(img)
			return img

		# sd process 
		processed = processing.process_images(p)

		# unload sd model
		shared.sd_model.cond_stage_model.to(devices.cpu)
		shared.sd_model.first_stage_model.to(devices.cpu)

		print('\n%s' % scriptname)
		
		# init torch device
		if compute_device == 0 or model_type == 4:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			device = torch.device("cpu")
		print("device: %s" % device)

		# model path and name
		model_dir = "./models/midas"
		if model_type == 4:
			model_dir = "./models/leres"
		# create path to model if not present
		os.makedirs(model_dir, exist_ok=True)

		print("Loading depth model weights from ", end=" ")

		try:
			#"dpt_large"
			if model_type == 0: 
				model_path = f"{model_dir}/dpt_large-midas-2f21e586.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="vitl16_384",
					non_negative=True,
				)
				net_w, net_h = 384, 384
				resize_mode = "minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"dpt_hybrid"
			elif model_type == 1: 
				model_path = f"{model_dir}/dpt_hybrid-midas-501f0c75.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt")
				model = DPTDepthModel(
					path=model_path,
					backbone="vitb_rn50_384",
					non_negative=True,
				)
				net_w, net_h = 384, 384
				resize_mode="minimal"
				normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			#"midas_v21"
			elif model_type == 2: 
				model_path = f"{model_dir}/midas_v21-f6b98070.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt")
				model = MidasNet(model_path, non_negative=True)
				net_w, net_h = 384, 384
				resize_mode="upper_bound"
				normalization = NormalizeImage(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				)

			#"midas_v21_small"
			elif model_type == 3: 
				model_path = f"{model_dir}/midas_v21_small-70d6b9c8.pt"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt")
				model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
				net_w, net_h = 256, 256
				resize_mode="upper_bound"
				normalization = NormalizeImage(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				)
			#"res101"
			elif model_type == 4: 
				model_path = f"{model_dir}/res101.pth"
				print(model_path)
				if not os.path.exists(model_path):
					download_file(model_path,"https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download")
				checkpoint = torch.load(model_path)
				model = RelDepthModel(backbone='resnext101')
				model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
				del checkpoint

			# override net size
			if (match_size):
				net_width, net_height = processed.width, processed.height

			# init transform
			if model_type != 4:
				transform = Compose(
					[
						Resize(
							net_width,
							net_height,
							resize_target=None,
							keep_aspect_ratio=True,
							ensure_multiple_of=32,
							resize_method=resize_mode,
							image_interpolation_method=cv2.INTER_CUBIC,
						),
						normalization,
						PrepareForNet(),
					]
				)

			model.eval()
			
			# optimize
			if device == torch.device("cuda"):
				model = model.to(memory_format=torch.channels_last)  
				if not cmd_opts.no_half and model_type != 4:
					model = model.half()

			model.to(device)

			print("Computing normalmap(s) ..")
			# iterate over input (generated) images
			for count in range(0,len(processed.images)):
				# skip first (grid) image if count > 1
				if count == 0 and len(processed.images) > 1:
					continue

				# input image
				img = cv2.cvtColor(np.asarray(processed.images[count]), cv2.COLOR_BGR2RGB) / 255.0
				
				if model_type == 4:

					# leres transform input
					rgb_c = img[:, :, ::-1].copy()
					A_resize = cv2.resize(rgb_c, (net_width, net_height))
					img_torch = scale_torch(A_resize)[None, :, :, :] 
					# Forward pass
					with torch.no_grad():
						prediction = model.inference(img_torch)
					prediction = prediction.squeeze().cpu().numpy()
					prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

				else:

					# midas transform input
					img_input = transform({"image": img})["image"]

					# compute
					precision_scope = torch.autocast if shared.cmd_opts.precision == "autocast" and device == torch.device("cuda") else contextlib.nullcontext
					with torch.no_grad(), precision_scope("cuda"):
						sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
						if device == torch.device("cuda"):
							sample = sample.to(memory_format=torch.channels_last) 
							if not cmd_opts.no_half:
								sample = sample.half()
						prediction = model.forward(sample)
						prediction = (
							torch.nn.functional.interpolate(
								prediction.unsqueeze(1),
								size=img.shape[:2],
								mode="bicubic",
								align_corners=False,
							)
							.squeeze()
							.cpu()
							.numpy()
						)

				# output
				normal = prediction
				numbytes=2
				normal_min = normal.min()
				normal_max = normal.max()
				max_val = (2**(8*numbytes))-1

				# check output before normalizing and mapping to 16 bit
				if normal_max - normal_min > np.finfo("float").eps:
					out = max_val * (normal - normal_min) / (normal_max - normal_min)
				else:
					out = np.zeros(normal.shape)
				
				# single channel, 16 bit image
				img_output = out.astype("uint16")

				# invert normal map
				if invert_normal ^ model_type == 4:
					img_output = cv2.bitwise_not(img_output)

				img_input *= depth_scale

				# three channel, 8 bits per channel image
				img_output2 = np.zeros_like(processed.images[count])
				img_output2[:,:,0] = img_output / 256.0
				img_output2[:,:,1] = img_output / 256.0
				img_output2[:,:,2] = img_output / 256.0

				#pre blur (only blurs z-axis)
				if pre_gaussian_blur:
					img_output = cv2.GaussianBlur(img_output, (pre_gaussian_blur_kernel, pre_gaussian_blur_kernel), pre_gaussian_blur_kernel)

				# take gradients 
				if sobel_gradient:
					zx = cv2.Sobel(np.float64(img_output), cv2.CV_64F, 1, 0, ksize=sobel_kernel)     
					zy = cv2.Sobel(np.float64(img_output), cv2.CV_64F, 0, 1, ksize=sobel_kernel) 
				else:
					zy, zx = np.gradient(img_output)

				# combine and normalize gradients.
				normal = np.dstack((zx, -zy, np.ones_like(img_output)))
				n = np.linalg.norm(normal, axis=2)
				normal[:, :, 0] /= n
				normal[:, :, 1] /= n
				normal[:, :, 2] /= n

				# post blur (will break normal maps unitary values)
				if post_gaussian_blur:
					normal = cv2.GaussianBlur(normal, (post_gaussian_blur_kernel, post_gaussian_blur_kernel), post_gaussian_blur_kernel)

				# offset and rescale values to be in 0-255
				normal += 1
				normal /= 2
				normal *= 255	
				normal = normal.astype(np.uint8)

				# get generation parameters
				if hasattr(p, 'all_prompts') and opts.enable_pnginfo:
					info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, "", 0, count-1)
				else:
					info = None

				if not combine_output:
					if show_normal:
						processed.images.append(Image.fromarray(normal))
					if save_normal:
						images.save_image(Image.fromarray(normal), p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_normal")
				else:
					img_concat = np.concatenate((processed.images[count], normal), axis=combine_output_axis)
					if show_normal:
						processed.images.append(Image.fromarray(img_concat))
					if save_normal:
						images.save_image(Image.fromarray(img_concat), p.outpath_samples, "", processed.all_prompts[count-1], p.prompt, opts.samples_format, info=info, p=p, suffix="_normal")


		except RuntimeError as e:
			if 'out of memory' in str(e):
				print("ERROR: out of memory, could not generate normalmap !")
			else:
				print(e)

		finally:
			del model
			gc.collect()
			devices.torch_gc()

			# reload sd model
			shared.sd_model.cond_stage_model.to(devices.device)
			shared.sd_model.first_stage_model.to(devices.device)

		return processed

