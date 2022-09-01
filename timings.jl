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
		Pkg.develop(path="/Users/daleblack/Google Drive/dev/julia/DistanceTransforms")
	end
	using Revise
	using PlutoUI
	using BenchmarkTools
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

TODO:
- Add SquaredEuclidean 2D GPU
- Add Wenbo 2D
- Add Wenbo 2D In-Place
- Add Wenbo 2D Threaded
- Add Wenbo 2D GPU


- Add SquaredEuclidean 3D GPU
- Add Wenbo 3D
- Add Wenbo 3D In-Place
- Add Wenbo 3D Threaded
- Add Wenbo 3D GPU
"""

# ╔═╡ e1bcf8e2-7824-4443-8e0c-4800d20a4cbb
md"""
## 2D
"""

# ╔═╡ 064702c2-cb91-4c19-baab-4586b710f9cf
nthreads = Threads.nthreads()

# ╔═╡ 578eda3b-8f90-4992-87da-d44950f4b891
num_range = 1:2:20

# ╔═╡ b5b6608f-0b75-495e-a37c-ed45d47fe407
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

	sizes = []
	
	for n in 1:10:100
		_size = n*n
		append!(sizes, _size)
		
		# EDT
		f = Bool.(rand([0, 1], n, n))
		edt = @benchmark euclidean($f)
		
		append!(edt_mean_2D, BenchmarkTools.mean(edt).time)
		append!(edt_std_2D, BenchmarkTools.std(edt).time)
		
		# SEDT
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt = @benchmark DistanceTransforms.transform($boolean_indicator($f), $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_mean_2D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_2D, BenchmarkTools.std(sedt).time)
		
		# SEDT In-Place
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_inplace = @benchmark DistanceTransforms.transform!($boolean_indicator($f), $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_inplace_mean_2D, BenchmarkTools.mean(sedt_inplace).time)
		append!(sedt_inplace_std_2D, BenchmarkTools.std(sedt_inplace).time)
		
		# SEDT Threaded
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_threaded = @benchmark DistanceTransforms.transform!($boolean_indicator($f), $tfm, $nthreads; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_2D, BenchmarkTools.std(sedt_threaded).time)

		# SEDT DepthFirst()
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = DepthFirstEx()
		sedt_threaded_depth = @benchmark DistanceTransforms.transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_depth, BenchmarkTools.mean(sedt_threaded_depth).time)
		append!(sedt_threaded_std_2D_depth, BenchmarkTools.std(sedt_threaded_depth).time)

		# SEDT NonThreadedEx()
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = NonThreadedEx()
		sedt_threaded_nonthread = @benchmark DistanceTransforms.transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_nonthread, BenchmarkTools.mean(sedt_threaded_nonthread).time)
		append!(sedt_threaded_std_2D_nonthread, BenchmarkTools.std(sedt_threaded_nonthread).time)

		# SEDT WorkStealingEx()
		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		ex = WorkStealingEx()
		sedt_threaded_worksteal = @benchmark DistanceTransforms.transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_2D_worksteal, BenchmarkTools.mean(sedt_threaded_worksteal).time)
		append!(sedt_threaded_std_2D_worksteal, BenchmarkTools.std(sedt_threaded_worksteal).time)
	end
end

# ╔═╡ ccf9ac88-7051-4b13-aaf7-9b38cf209a5e
sizes_2D = sizes

# ╔═╡ 788654ed-4bd0-4f1a-8a4d-007448de19f9
md"""
### Save CSV
"""

<<<<<<< HEAD
# ╔═╡ b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
let
	path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings_dt_2D.csv"
=======
# ╔═╡ bfdc1c54-546b-4ab0-8312-b48ce1ed557e
begin
	edt_mean_3D = []
	edt_std_3D = []
	
	sedt_mean_3D = []
	sedt_std_3D = []
	
	sedt_inplace_mean_3D = []
	sedt_inplace_std_3D = []

	sedt_threaded_mean_3D = []
	sedt_threaded_std_3D = []

	sizes_3D = []
	
	for n in 1:2:10
		_size = n^3
		push!(sizes_3D, _size)
		
		# EDT
		f = Bool.(rand([0, 1], n, n, n))
		edt = @benchmark euclidean($f)
		
		append!(edt_mean_3D, BenchmarkTools.mean(edt).time)
		append!(edt_std_3D, BenchmarkTools.std(edt).time)
		
		# SEDT
		f = boolean_indicator(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt = @benchmark DistanceTransforms.transform($f, $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_mean_3D, BenchmarkTools.mean(sedt).time)
		append!(sedt_std_3D, BenchmarkTools.std(sedt).time)
		
		# SEDT In-Place
		f = boolean_indicator(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_inplace = @benchmark DistanceTransforms.transform!($f, $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_inplace_mean_3D, BenchmarkTools.mean(sedt_inplace).time)
		append!(sedt_inplace_std_3D, BenchmarkTools.std(sedt_inplace).time)
		
		# SEDT Threaded
		f = boolean_indicator(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_threaded = @benchmark DistanceTransforms.transform!($f, $tfm, $nthreads; output=$output, v=$v, z=$z)
		
		append!(sedt_threaded_mean_3D, BenchmarkTools.mean(sedt_threaded).time)
		append!(sedt_threaded_std_3D, BenchmarkTools.std(sedt_threaded).time)
	end
end

# ╔═╡ 99042c20-2627-4c91-9690-95f3f9508588
md"""
## GPU 2D
"""

# ╔═╡ 01f15cd7-a57a-4963-b96f-08404be84e6f
if has_cuda_gpu()
	sedt_gpu_mean_2D = []
	sedt_gpu_std_2D = []
	
	for n in 1:2:10
		
		# SEDT GPU
		f = CuArray(boolean_indicator(rand([0, 1], n, n)))
		output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_gpu = @benchmark DistanceTransforms.transform!($f, $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_gpu_mean_2D, BenchmarkTools.mean(sedt_gpu).time)
		append!(sedt_gpu_std_2D, BenchmarkTools.std(sedt_gpu).time)
	end
else
	@show "No GPU available"
end	

# ╔═╡ 930ff718-bd8a-4b48-a14e-2aa4b61f0f39
md"""
## GPU 3D
"""

# ╔═╡ 39035604-5bdd-4267-a1ce-1610d4b815c0
if has_cuda_gpu()
	sedt_gpu_mean_3D = []
	sedt_gpu_std_3D = []
	
	for n in 1:2:10
		
		# SEDT GPU
		f = CuArray(boolean_indicator(rand([0, 1], n, n, n)))
		output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
		tfm = SquaredEuclidean()
		sedt_gpu = @benchmark DistanceTransforms.transform!($f, $tfm; output=$output, v=$v, z=$z)
		
		append!(sedt_gpu_mean_3D, BenchmarkTools.mean(sedt_gpu).time)
		append!(sedt_gpu_std_3D, BenchmarkTools.std(sedt_gpu).time)
	end
else
	@show "No GPU available"
end

# ╔═╡ d4c0b24a-bad3-4814-9975-84365d31e37f
md"""
## Save to CSV file
"""

# ╔═╡ 75fba553-8d26-4ecb-8807-1621e2fefed3
# if CUDA.functional()
# 	path = ""
	# df = DataFrame(
	# 	edt_mean_2D = Float64.(edt_mean_2D),
	# 	edt_std_2D = Float64.(edt_std_2D),
	# 	sedt_mean_2D = Float64.(sedt_mean_2D),
	# 	sedt_std_2D = Float64.(sedt_std_2D),
	# 	sedt_inplace_mean_2D = Float64.(sedt_inplace_mean_2D),
	# 	sedt_inplace_std_2D = Float64.(sedt_inplace_std_2D),
	# 	sedt_threaded_mean_2D = Float64.(sedt_threaded_mean_2D),
	# 	sedt_threaded_std_2D = Float64.(sedt_threaded_std_2D),
	# 	sedt_gpu_mean_2D = Float64.(sedt_gpu_mean_2D),
	# 	sedt_gpu_std_2D = Float64.(sedt_gpu_std_2D),
	# 	edt_mean_3D = Float64.(edt_mean_3D),
	# 	edt_std_3D = Float64.(edt_std_3D),
	# 	sedt_mean_3D = Float64.(sedt_mean_3D),
	# 	sedt_std_3D = Float64.(sedt_std_3D),
	# 	sedt_inplace_mean_3D = Float64.(sedt_inplace_mean_3D),
	# 	sedt_inplace_std_3D = Float64.(sedt_inplace_std_3D),
	# 	sedt_threaded_mean_3D = Float64.(sedt_threaded_mean_3D),
	# 	sedt_threaded_std_3D = Float64.(sedt_threaded_std_3D)
	# 	sedt_gpu_mean_3D = Float64.(sedt_gpu_mean_3D),
	# 	sedt_gpu_std_3D = Float64.(sedt_gpu_std_3D)
	# )
# 	CSV.write(df, path)
# else
# 	@show "No GPU available"
# end

# ╔═╡ b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
begin
	path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings.csv"
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
		sizes_3D = Float64.(sizes_3D),
		edt_mean_3D = Float64.(edt_mean_3D),
		edt_std_3D = Float64.(edt_std_3D),
		sedt_mean_3D = Float64.(sedt_mean_3D),
		sedt_std_3D = Float64.(sedt_std_3D),
		sedt_inplace_mean_3D = Float64.(sedt_inplace_mean_3D),
		sedt_inplace_std_3D = Float64.(sedt_inplace_std_3D),
		sedt_threaded_mean_3D = Float64.(sedt_threaded_mean_3D),
		sedt_threaded_std_3D = Float64.(sedt_threaded_std_3D),
	)
	CSV.write(path, df)
else
	@show "No GPU available"
end

# ╔═╡ 2d77bfc2-0457-435f-af13-6b70ef07c17d
md"""
# Loss Functions
"""

# ╔═╡ Cell order:
# ╠═7ef078de-23f9-11ed-104e-5f232d5f92b1
# ╠═bb360c60-7eb3-4fea-91c9-fc53bc643c45
# ╟─708245f0-ced3-4f1d-9062-140e39fed6f7
# ╟─e1bcf8e2-7824-4443-8e0c-4800d20a4cbb
# ╠═064702c2-cb91-4c19-baab-4586b710f9cf
# ╠═ccf9ac88-7051-4b13-aaf7-9b38cf209a5e
# ╠═578eda3b-8f90-4992-87da-d44950f4b891
# ╠═b5b6608f-0b75-495e-a37c-ed45d47fe407
# ╟─47c7ce53-b4f4-4930-a4ce-b681b331f1d8
# ╠═b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
# ╟─788654ed-4bd0-4f1a-8a4d-007448de19f9
# ╠═bfdc1c54-546b-4ab0-8312-b48ce1ed557e
# ╟─846260d6-b64a-4ab1-953b-a296d59db2dd
# ╠═f5189fc3-fea9-4e57-a4b3-b9ae1a8c0c10
# ╟─99042c20-2627-4c91-9690-95f3f9508588
# ╠═01f15cd7-a57a-4963-b96f-08404be84e6f
# ╟─43f7c561-27ce-4fa1-90cd-a7591470da20
# ╠═75fba553-8d26-4ecb-8807-1621e2fefed3
# ╟─930ff718-bd8a-4b48-a14e-2aa4b61f0f39
# ╠═39035604-5bdd-4267-a1ce-1610d4b815c0
# ╟─73fcf580-75a5-40b8-976a-ff9d5865b533
# ╠═d18fb298-f0e5-4417-bd01-04c274016a6c
# ╟─2d77bfc2-0457-435f-af13-6b70ef07c17d
