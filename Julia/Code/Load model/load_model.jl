### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ 4a04f147-0b54-4d63-abe7-9c7e439189aa
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	# Pkg.add("CSV")
	# Pkg.add("Glob")
	# Pkg.add("Flux")
	# Pkg.add("NIfTI")
	# Pkg.add("Images")
	# Pkg.add("FastAI")
	# Pkg.add("FastVision")
	# Pkg.add("CairoMakie")
	# Pkg.add("DataFrames")
	# Pkg.add("StaticArrays")
	# Pkg.add("MLDataPattern")
	using CSV
	using Glob
	using Flux
	using NIfTI
	using Images
	using FastAI
	using FastVision
	using CairoMakie
	using DataFrames
	using StaticArrays
	using MLDataPattern
end

# ╔═╡ 024b01cd-e065-4657-8e28-2ab0b4569b6a
begin
	function load_data_and_model(data_dir, dice_path, dice_hd_path; batch_size = 4)
		function loadfn_label(p)
			a = NIfTI.niread(string(p)).raw
			convert_a = convert(Array{UInt8}, a)
			convert_a = convert_a .+ 1
			return convert_a
		end
		function loadfn_image(p)
			a = NIfTI.niread(string(p)).raw
			convert_a = convert(Array{Float32}, a)
			convert_a = convert_a / max(convert_a...)
			return convert_a
		end
		function presize(files, image_size)
			container_images = Array{Float32,4}(undef, image_size..., numobs(files))
			container_masks = Array{Int64,4}(undef, image_size..., numobs(files))
			for i in 1:numobs(files)
				image, mask = FastAI.getobs(files, i)
				img = imresize(image, image_size)
				msk = round.(imresize(mask, image_size))
				container_images[:, :, :, i] = img
				container_masks[:, :, :, i] = msk
			end
			return container_images, container_masks
		end
		print("Starting --> ")
		task, model_dice = loadtaskmodel(dice_path)
		_, model_dice_hd = loadtaskmodel(dice_hd_path)
		print("Model loaded --> ")
		images(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
		masks(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
		pre_data = (
			images(joinpath(data_dir, "imagesTr")),
			masks(joinpath(data_dir, "labelsTr")),
		)
		print("Data loaded --> ")
		image_size = (96, 96, 96)
		img_container, mask_container = presize(pre_data, image_size)
		data_resized = (img_container,mask_container)
		println("Data Prepared.")
		a, b = FastVision.imagedatasetstats(img_container, Gray{N0f8}) 
		means, stds = SVector{1, Float32}(a[1]), SVector{1, Float32}(b[1])
		train_files, val_files = MLDataPattern.splitobs(data_resized, 0.8)
		tdl, vdl = FastAI.taskdataloaders(train_files, val_files, task, batch_size);
		println("Dataloader Created --> Finished!")
		return tdl, vdl, model_dice,model_dice_hd
	end
	function argmax_2ch(pred)
		img1, img2 = pred[:,:,:,1,:], pred[:,:,:,2,:]
		rslt = similar(img1)
		for i in CartesianIndices(rslt)
			rslt[i] = img1[i] > img2[i] ? 0 : 1
		end
		return rslt
	end
end;

# ╔═╡ 353a2064-1f14-4fd2-89ea-dbc4d03ba637
md"""
# Setting up
This notebook loads models from local disk. It is designed to run without CUDA. i.e. cpu only. This notebook was tested on a macbook with Julia = 1.7.4. But Julia and pkg versions do not matter.  

Before running this notebook, you need to:
- Have 'Task02_Heart' dataset at local disk.
- Have model files end with '.jld2 at local disk.

To load data and models:
- Set correct dataset path(**TODO 1**).
- Set correct model path(**TODO 2, 3**).
- Call `load_data_and_model(..)`(**TODO 4**).
`load_data_and_model(..)` takes in :
1. Path to the dataset,
2. Path to the dice model,
3. Path to the dice hd model,
4. (optional) batch size, default = `4`.
And returns:
1. Train dataloader,
2. Validation dataloader,
3. The dice model,
4. The dice hd model.
"""

# ╔═╡ fdb52331-f035-482f-b6b1-e206a6394b42
md"""
# Load model
"""

# ╔═╡ c828a11d-bd2e-4b46-8de9-bb7d966f751d
begin
	# TODO 1
	path_to_dataset = "/Users/wenboli/Desktop/ssd/Task02_Heart"
	# TODO 2
	path_to_DICE_model = "/Users/wenboli/Desktop/Load Model/Saved models/1/Dice_250.jld2"
	# TODO 3
	path_to_DICE_HD_model = "/Users/wenboli/Desktop/Load Model/Saved models/1/Dice_HD_175.jld2"
	# TODO 4
	tdl, vdl, model_dice, model_dice_hd = 
		load_data_and_model(path_to_dataset, path_to_DICE_model, path_to_DICE_HD_model)
end;

# ╔═╡ ab5ca648-9272-47a7-8520-df587e0d46ff
md"""
# Apply image from dataloader to model
"""

# ╔═╡ 4f3632a9-1998-44fa-92e7-0eb5c5918174
begin
	model_trained = model_dice_hd
	# Pick example from dataloader
	(example1, ) = vdl
	x1, y1 = example1
	# Model forward
	y_pred1 = model_trained(x1)
	# argmax of backgound and foreground
	y_pred1_argmaxed = argmax_2ch(y_pred1)
end;

# ╔═╡ 8b70a85f-12f5-4eee-aba1-1ff47a377d6e
CairoMakie.heatmap(y1[42,:,:,2,1])

# ╔═╡ c5558436-9fce-4382-98c6-e3936aa9c75a
CairoMakie.heatmap(y_pred1_argmaxed[42,:,:,1,1])

# ╔═╡ 554b84ac-9809-4c13-b305-9c201ab895d0
# Do something else...

# ╔═╡ Cell order:
# ╠═4a04f147-0b54-4d63-abe7-9c7e439189aa
# ╟─024b01cd-e065-4657-8e28-2ab0b4569b6a
# ╟─353a2064-1f14-4fd2-89ea-dbc4d03ba637
# ╟─fdb52331-f035-482f-b6b1-e206a6394b42
# ╠═c828a11d-bd2e-4b46-8de9-bb7d966f751d
# ╟─ab5ca648-9272-47a7-8520-df587e0d46ff
# ╠═4f3632a9-1998-44fa-92e7-0eb5c5918174
# ╠═8b70a85f-12f5-4eee-aba1-1ff47a377d6e
# ╠═c5558436-9fce-4382-98c6-e3936aa9c75a
# ╠═554b84ac-9809-4c13-b305-9c201ab895d0
