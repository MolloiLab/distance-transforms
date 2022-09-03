### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 7ef078de-23f9-11ed-104e-5f232d5f92b1
# ╠═╡ show_logs = false
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("Revise")
		Pkg.add("BenchmarkTools")
		Pkg.add("CairoMakie")
		Pkg.add("PlutoUI")
		Pkg.add("DataFrames")
		Pkg.add("CSV")
		Pkg.add("CUDA")
		Pkg.add("FoldsThreads")
		Pkg.develop(path="/Users/daleblack/Google Drive/dev/julia/DistanceTransforms")
		Pkg.develop(path="/Users/daleblack/Google Drive/dev/julia/Losers")
	end
	using Revise
	using PlutoUI
	using BenchmarkTools
	using CairoMakie
	using DataFrames
	using CSV
	using CUDA
	using FoldsThreads
	using DistanceTransforms
	using Losers
end

# ╔═╡ bb360c60-7eb3-4fea-91c9-fc53bc643c45
TableOfContents()

# ╔═╡ 708245f0-ced3-4f1d-9062-140e39fed6f7
md"""
# Distance Transforms
"""

# ╔═╡ e1bcf8e2-7824-4443-8e0c-4800d20a4cbb
md"""
## 2D
"""

# ╔═╡ 578eda3b-8f90-4992-87da-d44950f4b891
num_range = range(1, 200; length=4)

# ╔═╡ 064702c2-cb91-4c19-baab-4586b710f9cf
threads = Threads.nthreads()

# ╔═╡ e313c086-ccf4-44ed-947d-5582c72f9b39
begin
	edt_mean_2D = []
	edt_std_2D = []
	
	sedt_mean_2D = []
	sedt_std_2D = []
	
	sedt_inplace_mean_2D = []
	sedt_inplace_std_2D = []

	sedt_threaded_mean_2D = []
	sedt_threaded_std_2D = []

	sedt_threaded_mean_2D_depth = []
	sedt_threaded_std_2D_depth = []

	sedt_threaded_mean_2D_nonthread = []
	sedt_threaded_std_2D_nonthread = []

	sedt_threaded_mean_2D_worksteal = []
	sedt_threaded_std_2D_worksteal = []

	sedt_gpu_mean_2D = []
	sedt_gpu_std_2D = []

	sizes_2D = []
	
	for n in num_range
		n = Int(round(n))
		@info n
		_size = n^2
		append!(sizes_2D, _size)
		
		# EDT
		f = Bool.(rand([0, 1], n, n))
		edt = @benchmark euclidean($f)
		
		append!(edt_mean_2D, BenchmarkTools.mean(edt).time)
		append!(edt_std_2D, BenchmarkTools.std(edt).time)
		
		# SEDT
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		tfm = SquaredEuclidean()
		sedt = @benchmark DistanceTransforms.transform($b_f, $tfm)
		
		append!(sedt_mean_2D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_2D, BenchmarkTools.std(sedt).time)
		
		# SEDT In-Place
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		sedt_inplace = @benchmark DistanceTransforms.transform!($b_f, $tfm)
		
		append!(sedt_inplace_mean_2D, BenchmarkTools.mean(sedt_inplace).time)
		append!(sedt_inplace_std_2D, BenchmarkTools.std(sedt_inplace).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		sedt_threaded = @benchmark DistanceTransforms.transform!($b_f, $tfm, $threads)
		
		append!(sedt_threaded_mean_2D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_2D, BenchmarkTools.std(sedt_threaded).time)

		# SEDT DepthFirst()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = DepthFirstEx()
		sedt_threaded_depth = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_depth, BenchmarkTools.mean(sedt_threaded_depth).time)
		append!(sedt_threaded_std_2D_depth, BenchmarkTools.std(sedt_threaded_depth).time)

		# SEDT NonThreadedEx()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = NonThreadedEx()
		sedt_threaded_nonthread = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_nonthread, BenchmarkTools.mean(sedt_threaded_nonthread).time)
		append!(sedt_threaded_std_2D_nonthread, BenchmarkTools.std(sedt_threaded_nonthread).time)

		# SEDT WorkStealingEx()
		f = Bool.(rand([0, 1], n, n))
		b_f = boolean_indicator(f)
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = WorkStealingEx()
		sedt_threaded_worksteal = @benchmark DistanceTransforms.transform!($b_f, $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_worksteal, BenchmarkTools.mean(sedt_threaded_worksteal).time)
		append!(sedt_threaded_std_2D_worksteal, BenchmarkTools.std(sedt_threaded_worksteal).time)

		if has_cuda_gpu()
			# SEDT GPU
			f = Bool.(rand([0, 1], n, n))
			b_f = CuArray(boolean_indicator(f))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			tfm = SquaredEuclidean()
			sedt_gpu_2D = @benchmark DistanceTransforms.transform!($b_f, $tfm; output=$output, v=$v, z=$z)
			
			append!(sedt_gpu_mean_2D, BenchmarkTools.mean(sedt_gpu_2D).time)
			append!(sedt_gpu_std_2D, BenchmarkTools.std(sedt_gpu_2D).time)
		end
	end
end

# ╔═╡ 47c7ce53-b4f4-4930-a4ce-b681b331f1d8
md"""
### Save CSV
"""

# ╔═╡ b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
let
	path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings_dt_2D.csv"
	if has_cuda_gpu()
		df = DataFrame(
			sizes_2D = Float64.(sizes_2D),
			edt_mean_2D = Float64.(edt_mean_2D),
			edt_std_2D = Float64.(edt_std_2D),
			sedt_mean_2D = Float64.(sedt_mean_2D),
			sedt_std_2D = Float64.(sedt_std_2D),
			sedt_inplace_mean_2D = Float64.(sedt_inplace_mean_2D),
			sedt_inplace_std_2D = Float64.(sedt_inplace_std_2D),
			sedt_threaded_mean_2D = Float64.(sedt_threaded_mean_2D),
			sedt_threaded_std_2D = Float64.(sedt_threaded_std_2D),
			sedt_threaded_mean_2D_depth = sedt_threaded_mean_2D_depth,
			sedt_threaded_std_2D_depth = sedt_threaded_std_2D_depth,
			sedt_threaded_mean_2D_nonthread = sedt_threaded_mean_2D_nonthread,
			sedt_threaded_std_2D_nonthread = sedt_threaded_std_2D_nonthread,
			sedt_threaded_mean_2D_worksteal = sedt_threaded_mean_2D_worksteal,
			sedt_threaded_std_2D_worksteal = sedt_threaded_std_2D_worksteal,
			sedt_gpu_mean_2D = sedt_gpu_mean_2D,
			sedt_gpu_std_2D = sedt_gpu_std_2D,
		)
	else
		df = DataFrame(
			sizes_2D = Float64.(sizes_2D),
			edt_mean_2D = Float64.(edt_mean_2D),
			edt_std_2D = Float64.(edt_std_2D),
			sedt_mean_2D = Float64.(sedt_mean_2D),
			sedt_std_2D = Float64.(sedt_std_2D),
			sedt_inplace_mean_2D = Float64.(sedt_inplace_mean_2D),
			sedt_inplace_std_2D = Float64.(sedt_inplace_std_2D),
			sedt_threaded_mean_2D = Float64.(sedt_threaded_mean_2D),
			sedt_threaded_std_2D = Float64.(sedt_threaded_std_2D),
			sedt_threaded_mean_2D_depth = sedt_threaded_mean_2D_depth,
			sedt_threaded_std_2D_depth = sedt_threaded_std_2D_depth,
			sedt_threaded_mean_2D_nonthread = sedt_threaded_mean_2D_nonthread,
			sedt_threaded_std_2D_nonthread = sedt_threaded_std_2D_nonthread,
			sedt_threaded_mean_2D_worksteal = sedt_threaded_mean_2D_worksteal,
			sedt_threaded_std_2D_worksteal = sedt_threaded_std_2D_worksteal,
		)
	end
	CSV.write(path, df)
end

# ╔═╡ 788654ed-4bd0-4f1a-8a4d-007448de19f9
md"""
## 3D
"""

# ╔═╡ bfdc1c54-546b-4ab0-8312-b48ce1ed557e
# begin
# 	edt_mean_3D = []
# 	edt_std_3D = []
	
# 	sedt_mean_3D = []
# 	sedt_std_3D = []
	
# 	sedt_inplace_mean_3D = []
# 	sedt_inplace_std_3D = []

# 	sedt_threaded_mean_3D = []
# 	sedt_threaded_std_3D = []

# 	sizes_3D = []
	
# 	for n in num_range
# 		_size = n^3
# 		push!(sizes_3D, _size)
		
# 		# EDT
# 		f = Bool.(rand([0, 1], n, n, n))
# 		edt = @benchmark euclidean($f)
		
# 		append!(edt_mean_3D, BenchmarkTools.mean(edt).time)
# 		append!(edt_std_3D, BenchmarkTools.std(edt).time)
		
# 		# SEDT
# 		f = boolean_indicator(rand([0, 1], n, n, n))
# 		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
# 		tfm = SquaredEuclidean()
# 		sedt = @benchmark DistanceTransforms.transform($f, $tfm; output=$output, v=$v, z=$z)
		
# 		append!(sedt_mean_3D, BenchmarkTools.mean(sedt).time)
# 		append!(sedt_std_3D, BenchmarkTools.std(sedt).time)
		
# 		# SEDT In-Place
# 		f = boolean_indicator(rand([0, 1], n, n, n))
# 		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
# 		tfm = SquaredEuclidean()
# 		sedt_inplace = @benchmark DistanceTransforms.transform!($f, $tfm; output=$output, v=$v, z=$z)
		
# 		append!(sedt_inplace_mean_3D, BenchmarkTools.mean(sedt_inplace).time)
# 		append!(sedt_inplace_std_3D, BenchmarkTools.std(sedt_inplace).time)
		
# 		# SEDT Threaded
# 		f = boolean_indicator(rand([0, 1], n, n, n))
# 		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
# 		tfm = SquaredEuclidean()
# 		sedt_threaded = @benchmark DistanceTransforms.transform!($f, $tfm, $nthreads; output=$output, v=$v, z=$z)
		
# 		append!(sedt_threaded_mean_3D, BenchmarkTools.mean(sedt_threaded).time)
# 		append!(sedt_threaded_std_3D, BenchmarkTools.std(sedt_threaded).time)
# 	end
# end

# ╔═╡ 846260d6-b64a-4ab1-953b-a296d59db2dd
md"""
### Save CSV
"""

# ╔═╡ f5189fc3-fea9-4e57-a4b3-b9ae1a8c0c10
# let
# 	path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings_dt_3D.csv"
# 	df = DataFrame(
# 		sizes_3D = Float64.(sizes_3D),
# 		edt_mean_3D = Float64.(edt_mean_3D),
# 		edt_std_3D = Float64.(edt_std_3D),
# 		sedt_mean_3D = Float64.(sedt_mean_3D),
# 		sedt_std_3D = Float64.(sedt_std_3D),
# 		sedt_inplace_mean_3D = Float64.(sedt_inplace_mean_3D),
# 		sedt_inplace_std_3D = Float64.(sedt_inplace_std_3D),
# 		sedt_threaded_mean_3D = Float64.(sedt_threaded_mean_3D),
# 		sedt_threaded_std_3D = Float64.(sedt_threaded_std_3D),
# 	)
# 	CSV.write(path, df)
# end

# ╔═╡ 2d77bfc2-0457-435f-af13-6b70ef07c17d
md"""
# Loss Functions
"""

# ╔═╡ 1d7dc8c9-650c-4af1-a570-04fdd538df67
md"""
## 2D
"""

# ╔═╡ 9cc8791a-f577-475f-a96d-93d01edcb35b
# begin
# 	dice_mean_2D = []
# 	dice_std_2D = []

# 	hausdorff_mean_2D = []
# 	hausdorff_std_2D = []

# 	sizes_loss_2D = []
	
# 	for n in num_range
# 		_size = n*n
# 		push!(sizes_loss_2D, _size)
		
# 		# DICE
# 		f = Bool.(rand([0, 1], n, n))
# 		dice_loss = @benchmark dice($f, $f)
		
# 		append!(dice_mean_2D, BenchmarkTools.mean(dice_loss).time)
# 		append!(dice_std_2D, BenchmarkTools.std(dice_loss).time)
		
# 		# Hausdorff
# 		f = rand([0, 1], n, n)
# 		tfm = SquaredEuclidean()
# 		f_dtm = DistanceTransforms.transform(boolean_indicator(f), tfm)
# 		hausdorff_loss = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
		
# 		append!(hausdorff_mean_2D, BenchmarkTools.mean(hausdorff_loss).time)
# 		append!(hausdorff_std_2D, BenchmarkTools.std(hausdorff_loss).time)
		
# 	end
# end

# ╔═╡ 6c4632e3-ca49-431b-9ced-4c6269dd361e
# let
# 	path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings_loss_2D.csv"
# 	df = DataFrame(
# 		sizes_loss_2D = Float64.(sizes_loss_2D),
# 		dice_mean_2D = dice_mean_2D,
# 		dice_std_2D = dice_std_2D,
# 		hausdorff_mean_2D = hausdorff_mean_2D,
# 		hausdorff_std_2D = hausdorff_std_2D
# 	)
# 	CSV.write(path, df)
# end

# ╔═╡ Cell order:
# ╠═7ef078de-23f9-11ed-104e-5f232d5f92b1
# ╠═bb360c60-7eb3-4fea-91c9-fc53bc643c45
# ╟─708245f0-ced3-4f1d-9062-140e39fed6f7
# ╟─e1bcf8e2-7824-4443-8e0c-4800d20a4cbb
# ╠═578eda3b-8f90-4992-87da-d44950f4b891
# ╠═064702c2-cb91-4c19-baab-4586b710f9cf
# ╠═e313c086-ccf4-44ed-947d-5582c72f9b39
# ╟─47c7ce53-b4f4-4930-a4ce-b681b331f1d8
# ╠═b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
# ╟─788654ed-4bd0-4f1a-8a4d-007448de19f9
# ╠═bfdc1c54-546b-4ab0-8312-b48ce1ed557e
# ╟─846260d6-b64a-4ab1-953b-a296d59db2dd
# ╠═f5189fc3-fea9-4e57-a4b3-b9ae1a8c0c10
# ╟─2d77bfc2-0457-435f-af13-6b70ef07c17d
# ╟─1d7dc8c9-650c-4af1-a570-04fdd538df67
# ╠═9cc8791a-f577-475f-a96d-93d01edcb35b
# ╠═6c4632e3-ca49-431b-9ced-4c6269dd361e
