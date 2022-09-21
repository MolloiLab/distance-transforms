### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 7ef078de-23f9-11ed-104e-5f232d5f92b1
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
		
	using Revise
	using PlutoUI
	using BenchmarkTools
	using CairoMakie
	using DataFrames
	using CSV
	using CUDA
	using FoldsThreads
	using DistanceTransforms
	using DistanceTransforms: transform, transform!
	using Losers
end

# ╔═╡ bb360c60-7eb3-4fea-91c9-fc53bc643c45
TableOfContents()

# ╔═╡ bcdcca5c-4eba-4cb5-a955-897f0ef99120
current_dir = pwd()

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
    maurer_mean_2D = []
	maurer_std_2D = []
	
	felzenszwalb_mean_2D = []
	felzenszwalb_std_2D = []
	
	felzenszwalb_inplace_mean_2D = []
	felzenszwalb_inplace_std_2D = []

	felzenszwalb_threaded_mean_2D = []
	felzenszwalb_threaded_std_2D = []

	felzenszwalb_threaded_mean_2D_depth = []
	felzenszwalb_threaded_std_2D_depth = []

	felzenszwalb_threaded_mean_2D_nonthread = []
	felzenszwalb_threaded_std_2D_nonthread = []

	felzenszwalb_threaded_mean_2D_worksteal = []
	felzenszwalb_threaded_std_2D_worksteal = []

	felzenszwalb_gpu_mean_2D = []
	felzenszwalb_gpu_std_2D = []

	wenbo_inplace_mean_2D = []
	wenbo_inplace_std_2D = []

	wenbo_threaded_mean_2D = []
	wenbo_threaded_std_2D = []

	wenbo_threaded_mean_2D_depth = []
	wenbo_threaded_std_2D_depth = []

	wenbo_threaded_mean_2D_nonthread = []
	wenbo_threaded_std_2D_nonthread = []

	wenbo_threaded_mean_2D_worksteal = []
	wenbo_threaded_std_2D_worksteal = []

	wenbo_gpu_mean_2D = []
	wenbo_gpu_std_2D = []
	

	sizes_2D = []
	
	for n in num_range
		n = Int(round(n))
		@info "n = $(n)"
		_size = n^2
		append!(sizes_2D, _size)
		
		####--- Maurer (ImageMorphology) ---####
		f = Bool.(rand([0, 1], n, n))
		maurer = @benchmark transform($f, $Maurer())
		append!(maurer_mean_2D, BenchmarkTools.mean(maurer).time)
		append!(maurer_std_2D, BenchmarkTools.std(maurer).time)
		
		####--- Felzenszwalb ---####
		f = Bool.(rand([0, 1], n, n))
		tfm = Felzenszwalb()
		felzenszwalb = @benchmark transform($boolean_indicator($f), $tfm)
		append!(felzenszwalb_mean_2D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_std_2D, BenchmarkTools.std(felzenszwalb).time)
		
		f = Bool.(rand([0, 1], n, n))
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm)
		append!(felzenszwalb_inplace_mean_2D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_inplace_std_2D, BenchmarkTools.std(felzenszwalb).time)
		
		f = Bool.(rand([0, 1], n, n))
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $threads)
		append!(felzenszwalb_threaded_mean_2D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_2D, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = DepthFirstEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_2D_depth, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_2D_depth, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = NonThreadedEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_2D_nonthread, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_2D_nonthread, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = WorkStealingEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_2D_worksteal, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_2D_worksteal, BenchmarkTools.std(felzenszwalb).time)

		if has_cuda_gpu()
			f = Bool.(rand([0, 1], n, n))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			felzenszwalb = @benchmark transform!($CuArray($boolean_indicator($f)), $tfm; output=$output, v=$v, z=$z)
			append!(felzenszwalb_gpu_mean_2D, BenchmarkTools.mean(felzenszwalb).time)
			append!(felzenszwalb_gpu_std_2D, BenchmarkTools.std(felzenszwalb).time)
		end

		####--- Wenbo ---####
		f = Bool.(rand([0, 1], n, n))
		tfm2 = Wenbo()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2)
		append!(wenbo_inplace_mean_2D, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_inplace_std_2D, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n))
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $threads)
		append!(wenbo_threaded_mean_2D, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_2D, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n))
		ex = DepthFirstEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_2D_depth, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_2D_depth, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n))
		ex = NonThreadedEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_2D_nonthread, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_2D_nonthread, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n))
		ex = WorkStealingEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_2D_worksteal, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_2D_worksteal, BenchmarkTools.std(wenbo).time)

		if has_cuda_gpu()
			f = Bool.(rand([0, 1], n, n))
			wenbo = @benchmark transform!($CuArray($boolean_indicator($f)), $tfm2)
			append!(wenbo_gpu_mean_2D, BenchmarkTools.mean(wenbo).time)
			append!(wenbo_gpu_std_2D, BenchmarkTools.std(wenbo).time)
		end
	end
end

# ╔═╡ b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
let
	path = current_dir * "/dt_2D.csv"
	if has_cuda_gpu()
		df = DataFrame(
			sizes_2D = Float64.(sizes_2D),
		    maurer_mean_2D = Float64.(maurer_mean_2D),
			maurer_std_2D = Float64.(maurer_std_2D),
			felzenszwalb_mean_2D = Float64.(felzenszwalb_mean_2D),
			felzenszwalb_std_2D = Float64.(felzenszwalb_std_2D),
			felzenszwalb_inplace_mean_2D = Float64.(felzenszwalb_inplace_mean_2D),
			felzenszwalb_inplace_std_2D = Float64.(felzenszwalb_inplace_std_2D),
			felzenszwalb_threaded_mean_2D = Float64.(felzenszwalb_threaded_mean_2D),
			felzenszwalb_threaded_std_2D = Float64.(felzenszwalb_threaded_std_2D),
			felzenszwalb_threaded_mean_2D_depth = Float64.(felzenszwalb_threaded_mean_2D_depth),
			felzenszwalb_threaded_std_2D_depth = Float64.(felzenszwalb_threaded_std_2D_depth),
			felzenszwalb_threaded_mean_2D_nonthread = Float64.(felzenszwalb_threaded_mean_2D_nonthread),
			felzenszwalb_threaded_std_2D_nonthread = Float64.(felzenszwalb_threaded_std_2D_nonthread),
			felzenszwalb_threaded_mean_2D_worksteal = Float64.(felzenszwalb_threaded_mean_2D_worksteal),
			felzenszwalb_threaded_std_2D_worksteal = Float64.(felzenszwalb_threaded_std_2D_worksteal),
			felzenszwalb_gpu_mean_2D = Float64.(felzenszwalb_gpu_mean_2D),
			felzenszwalb_gpu_std_2D = Float64.(felzenszwalb_gpu_std_2D),
			wenbo_inplace_mean_2D = Float64.(wenbo_inplace_mean_2D),
			wenbo_inplace_std_2D = Float64.(wenbo_inplace_std_2D),
			wenbo_threaded_mean_2D = Float64.(wenbo_threaded_mean_2D),
			wenbo_threaded_std_2D = Float64.(wenbo_threaded_std_2D),
			wenbo_threaded_mean_2D_depth = Float64.(wenbo_threaded_mean_2D_depth),
			wenbo_threaded_std_2D_depth = Float64.(wenbo_threaded_std_2D_depth),
			wenbo_threaded_mean_2D_nonthread = Float64.(wenbo_threaded_mean_2D_nonthread),
			wenbo_threaded_std_2D_nonthread = Float64.(wenbo_threaded_std_2D_nonthread),
			wenbo_threaded_mean_2D_worksteal = Float64.(wenbo_threaded_mean_2D_worksteal),
			wenbo_threaded_std_2D_worksteal = Float64.(wenbo_threaded_std_2D_worksteal),
			wenbo_gpu_mean_2D = Float64.(wenbo_gpu_mean_2D),
			wenbo_gpu_std_2D = Float64.(wenbo_gpu_std_2D)
		)
	else
		df = DataFrame(
			sizes_2D = Float64.(sizes_2D),
		    maurer_mean_2D = Float64.(maurer_mean_2D),
			maurer_std_2D = Float64.(maurer_std_2D),
			felzenszwalb_mean_2D = Float64.(felzenszwalb_mean_2D),
			felzenszwalb_std_2D = Float64.(felzenszwalb_std_2D),
			felzenszwalb_inplace_mean_2D = Float64.(felzenszwalb_inplace_mean_2D),
			felzenszwalb_inplace_std_2D = Float64.(felzenszwalb_inplace_std_2D),
			felzenszwalb_threaded_mean_2D = Float64.(felzenszwalb_threaded_mean_2D),
			felzenszwalb_threaded_std_2D = Float64.(felzenszwalb_threaded_std_2D),
			felzenszwalb_threaded_mean_2D_depth = Float64.(felzenszwalb_threaded_mean_2D_depth),
			felzenszwalb_threaded_std_2D_depth = Float64.(felzenszwalb_threaded_std_2D_depth),
			felzenszwalb_threaded_mean_2D_nonthread = Float64.(felzenszwalb_threaded_mean_2D_nonthread),
			felzenszwalb_threaded_std_2D_nonthread = Float64.(felzenszwalb_threaded_std_2D_nonthread),
			felzenszwalb_threaded_mean_2D_worksteal = Float64.(felzenszwalb_threaded_mean_2D_worksteal),
			felzenszwalb_threaded_std_2D_worksteal = Float64.(felzenszwalb_threaded_std_2D_worksteal),
			wenbo_inplace_mean_2D = Float64.(wenbo_inplace_mean_2D),
			wenbo_inplace_std_2D = Float64.(wenbo_inplace_std_2D),
			wenbo_threaded_mean_2D = Float64.(wenbo_threaded_mean_2D),
			wenbo_threaded_std_2D = Float64.(wenbo_threaded_std_2D),
			wenbo_threaded_mean_2D_depth = Float64.(wenbo_threaded_mean_2D_depth),
			wenbo_threaded_std_2D_depth = Float64.(wenbo_threaded_std_2D_depth),
			wenbo_threaded_mean_2D_nonthread = Float64.(wenbo_threaded_mean_2D_nonthread),
			wenbo_threaded_std_2D_nonthread = Float64.(wenbo_threaded_std_2D_nonthread),
			wenbo_threaded_mean_2D_worksteal = Float64.(wenbo_threaded_mean_2D_worksteal),
			wenbo_threaded_std_2D_worksteal = Float64.(wenbo_threaded_std_2D_worksteal)
		)
	end
	CSV.write(path, df)
end

# ╔═╡ 788654ed-4bd0-4f1a-8a4d-007448de19f9
md"""
## 3D
"""

# ╔═╡ bfdc1c54-546b-4ab0-8312-b48ce1ed557e
begin
    maurer_mean_3D = []
	maurer_std_3D = []
	
	felzenszwalb_mean_3D = []
	felzenszwalb_std_3D = []
	
	felzenszwalb_inplace_mean_3D = []
	felzenszwalb_inplace_std_3D = []

	felzenszwalb_threaded_mean_3D = []
	felzenszwalb_threaded_std_3D = []

	felzenszwalb_threaded_mean_3D_depth = []
	felzenszwalb_threaded_std_3D_depth = []

	felzenszwalb_threaded_mean_3D_nonthread = []
	felzenszwalb_threaded_std_3D_nonthread = []

	felzenszwalb_threaded_mean_3D_worksteal = []
	felzenszwalb_threaded_std_3D_worksteal = []

	felzenszwalb_gpu_mean_3D = []
	felzenszwalb_gpu_std_3D = []

	wenbo_inplace_mean_3D = []
	wenbo_inplace_std_3D = []

	wenbo_threaded_mean_3D = []
	wenbo_threaded_std_3D = []

	wenbo_threaded_mean_3D_depth = []
	wenbo_threaded_std_3D_depth = []

	wenbo_threaded_mean_3D_nonthread = []
	wenbo_threaded_std_3D_nonthread = []

	wenbo_threaded_mean_3D_worksteal = []
	wenbo_threaded_std_3D_worksteal = []

	wenbo_gpu_mean_3D = []
	wenbo_gpu_std_3D = []
	

	sizes_3D = []
	
	for n in num_range
		n = Int(round(n))
		@info "n = $(n)"
		_size = n^3
		append!(sizes_3D, _size)
		
		####--- Maurer (ImageMorphology) ---####
		f = Bool.(rand([0, 1], n, n, n))
		maurer = @benchmark transform($f, $Maurer())
		append!(maurer_mean_3D, BenchmarkTools.mean(maurer).time)
		append!(maurer_std_3D, BenchmarkTools.std(maurer).time)
		
		####--- Felzenszwalb ---####
		f = Bool.(rand([0, 1], n, n, n))
		tfm = Felzenszwalb()
		felzenszwalb = @benchmark transform($boolean_indicator($f), $tfm)
		append!(felzenszwalb_mean_3D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_std_3D, BenchmarkTools.std(felzenszwalb).time)
		
		f = Bool.(rand([0, 1], n, n, n))
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm)
		append!(felzenszwalb_inplace_mean_3D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_inplace_std_3D, BenchmarkTools.std(felzenszwalb).time)
		
		f = Bool.(rand([0, 1], n, n, n))
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $threads)
		append!(felzenszwalb_threaded_mean_3D, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_3D, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = DepthFirstEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_3D_depth, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_3D_depth, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = NonThreadedEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_3D_nonthread, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_3D_nonthread, BenchmarkTools.std(felzenszwalb).time)

		f = Bool.(rand([0, 1], n, n, n))
		output, v, z = zeros(size(f)), ones(Int32, size(f)), ones(size(f) .+ 1)
		ex = WorkStealingEx()
		felzenszwalb = @benchmark transform!($boolean_indicator($f), $tfm, $ex; output=$output, v=$v, z=$z)
		append!(felzenszwalb_threaded_mean_3D_worksteal, BenchmarkTools.mean(felzenszwalb).time)
		append!(felzenszwalb_threaded_std_3D_worksteal, BenchmarkTools.std(felzenszwalb).time)

		if has_cuda_gpu()
			f = Bool.(rand([0, 1], n, n, n))
			output, v, z = CUDA.zeros(size(f)), CUDA.ones(Int32, size(f)), CUDA.ones(size(f) .+ 1)
			felzenszwalb = @benchmark transform!($CuArray($boolean_indicator($f)), $tfm; output=$output, v=$v, z=$z)
			append!(felzenszwalb_gpu_mean_3D, BenchmarkTools.mean(felzenszwalb).time)
			append!(felzenszwalb_gpu_std_3D, BenchmarkTools.std(felzenszwalb).time)
		end

		####--- Wenbo ---####
		f = Bool.(rand([0, 1], n, n, n))
		tfm2 = Wenbo()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2)
		append!(wenbo_inplace_mean_3D, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_inplace_std_3D, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n, n))
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $threads)
		append!(wenbo_threaded_mean_3D, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_3D, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n, n))
		ex = DepthFirstEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_3D_depth, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_3D_depth, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n, n))
		ex = NonThreadedEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_3D_nonthread, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_3D_nonthread, BenchmarkTools.std(wenbo).time)

		f = Bool.(rand([0, 1], n, n, n))
		ex = WorkStealingEx()
		wenbo = @benchmark transform!($boolean_indicator($f), $tfm2, $ex)
		append!(wenbo_threaded_mean_3D_worksteal, BenchmarkTools.mean(wenbo).time)
		append!(wenbo_threaded_std_3D_worksteal, BenchmarkTools.std(wenbo).time)

		if has_cuda_gpu()
			f = Bool.(rand([0, 1], n, n, n))
			wenbo = @benchmark transform!($CuArray($boolean_indicator($f)), $tfm2)
			append!(wenbo_gpu_mean_3D, BenchmarkTools.mean(wenbo).time)
			append!(wenbo_gpu_std_3D, BenchmarkTools.std(wenbo).time)
		end
	end
end

# ╔═╡ f5189fc3-fea9-4e57-a4b3-b9ae1a8c0c10
let
	path = current_dir * "/dt_3D.csv"
	if has_cuda_gpu()
		df = DataFrame(
			sizes_3D = Float64.(sizes_3D),
		    maurer_mean_3D = Float64.(maurer_mean_3D),
			maurer_std_3D = Float64.(maurer_std_3D),
			felzenszwalb_mean_3D = Float64.(felzenszwalb_mean_3D),
			felzenszwalb_std_3D = Float64.(felzenszwalb_std_3D),
			felzenszwalb_inplace_mean_3D = Float64.(felzenszwalb_inplace_mean_3D),
			felzenszwalb_inplace_std_3D = Float64.(felzenszwalb_inplace_std_3D),
			felzenszwalb_threaded_mean_3D = Float64.(felzenszwalb_threaded_mean_3D),
			felzenszwalb_threaded_std_3D = Float64.(felzenszwalb_threaded_std_3D),
			felzenszwalb_threaded_mean_3D_depth = Float64.(felzenszwalb_threaded_mean_3D_depth),
			felzenszwalb_threaded_std_3D_depth = Float64.(felzenszwalb_threaded_std_3D_depth),
			felzenszwalb_threaded_mean_3D_nonthread = Float64.(felzenszwalb_threaded_mean_3D_nonthread),
			felzenszwalb_threaded_std_3D_nonthread = Float64.(felzenszwalb_threaded_std_3D_nonthread),
			felzenszwalb_threaded_mean_3D_worksteal = Float64.(felzenszwalb_threaded_mean_3D_worksteal),
			felzenszwalb_threaded_std_3D_worksteal = Float64.(felzenszwalb_threaded_std_3D_worksteal),
			felzenszwalb_gpu_mean_3D = Float64.(felzenszwalb_gpu_mean_3D),
			felzenszwalb_gpu_std_3D = Float64.(felzenszwalb_gpu_std_3D),
			wenbo_inplace_mean_3D = Float64.(wenbo_inplace_mean_3D),
			wenbo_inplace_std_3D = Float64.(wenbo_inplace_std_3D),
			wenbo_threaded_mean_3D = Float64.(wenbo_threaded_mean_3D),
			wenbo_threaded_std_3D = Float64.(wenbo_threaded_std_3D),
			wenbo_threaded_mean_3D_depth = Float64.(wenbo_threaded_mean_3D_depth),
			wenbo_threaded_std_3D_depth = Float64.(wenbo_threaded_std_3D_depth),
			wenbo_threaded_mean_3D_nonthread = Float64.(wenbo_threaded_mean_3D_nonthread),
			wenbo_threaded_std_3D_nonthread = Float64.(wenbo_threaded_std_3D_nonthread),
			wenbo_threaded_mean_3D_worksteal = Float64.(wenbo_threaded_mean_3D_worksteal),
			wenbo_threaded_std_3D_worksteal = Float64.(wenbo_threaded_std_3D_worksteal),
			wenbo_gpu_mean_3D = Float64.(wenbo_gpu_mean_3D),
			wenbo_gpu_std_3D = Float64.(wenbo_gpu_std_3D)
		)
	else
		df = DataFrame(
			sizes_3D = Float64.(sizes_3D),
		    maurer_mean_3D = Float64.(maurer_mean_3D),
			maurer_std_3D = Float64.(maurer_std_3D),
			felzenszwalb_mean_3D = Float64.(felzenszwalb_mean_3D),
			felzenszwalb_std_3D = Float64.(felzenszwalb_std_3D),
			felzenszwalb_inplace_mean_3D = Float64.(felzenszwalb_inplace_mean_3D),
			felzenszwalb_inplace_std_3D = Float64.(felzenszwalb_inplace_std_3D),
			felzenszwalb_threaded_mean_3D = Float64.(felzenszwalb_threaded_mean_3D),
			felzenszwalb_threaded_std_3D = Float64.(felzenszwalb_threaded_std_3D),
			felzenszwalb_threaded_mean_3D_depth = Float64.(felzenszwalb_threaded_mean_3D_depth),
			felzenszwalb_threaded_std_3D_depth = Float64.(felzenszwalb_threaded_std_3D_depth),
			felzenszwalb_threaded_mean_3D_nonthread = Float64.(felzenszwalb_threaded_mean_3D_nonthread),
			felzenszwalb_threaded_std_3D_nonthread = Float64.(felzenszwalb_threaded_std_3D_nonthread),
			felzenszwalb_threaded_mean_3D_worksteal = Float64.(felzenszwalb_threaded_mean_3D_worksteal),
			felzenszwalb_threaded_std_3D_worksteal = Float64.(felzenszwalb_threaded_std_3D_worksteal),
			wenbo_inplace_mean_3D = Float64.(wenbo_inplace_mean_3D),
			wenbo_inplace_std_3D = Float64.(wenbo_inplace_std_3D),
			wenbo_threaded_mean_3D = Float64.(wenbo_threaded_mean_3D),
			wenbo_threaded_std_3D = Float64.(wenbo_threaded_std_3D),
			wenbo_threaded_mean_3D_depth = Float64.(wenbo_threaded_mean_3D_depth),
			wenbo_threaded_std_3D_depth = Float64.(wenbo_threaded_std_3D_depth),
			wenbo_threaded_mean_3D_nonthread = Float64.(wenbo_threaded_mean_3D_nonthread),
			wenbo_threaded_std_3D_nonthread = Float64.(wenbo_threaded_std_3D_nonthread),
			wenbo_threaded_mean_3D_worksteal = Float64.(wenbo_threaded_mean_3D_worksteal),
			wenbo_threaded_std_3D_worksteal = Float64.(wenbo_threaded_std_3D_worksteal)
		)
	end
	CSV.write(path, df)
end

# ╔═╡ 2d77bfc2-0457-435f-af13-6b70ef07c17d
md"""
# Loss Functions
"""

# ╔═╡ 1d7dc8c9-650c-4af1-a570-04fdd538df67
md"""
## 2D
"""

# ╔═╡ 9cc8791a-f577-475f-a96d-93d01edcb35b
begin
	dice_mean_2D = []
	dice_std_2D = []

	dice_mean_gpu_2D = []
	dice_std_gpu_2D = []

	hausdorff_mean_2D = []
	hausdorff_std_2D = []

	hausdorff_mean_gpu_2D = []
	hausdorff_std_gpu_2D = []

	sizes_loss_2D = []
	
	for n in num_range
		n = Int(round(n))
		_size = n^2
		push!(sizes_loss_2D, _size)
		@info "n = $(n)"
		
		# DICE
		f = Bool.(rand([0, 1], n, n))
		dice_loss = @benchmark dice($f, $f)
		append!(dice_mean_2D, BenchmarkTools.mean(dice_loss).time)
		append!(dice_std_2D, BenchmarkTools.std(dice_loss).time)

		if has_cuda_gpu()
			f = CuArray(Bool.(rand([0, 1], n, n)))
			dice_loss = @benchmark dice($f, $f)
			append!(dice_mean_gpu_2D, BenchmarkTools.mean(dice_loss).time)
			append!(dice_std_gpu_2D, BenchmarkTools.std(dice_loss).time)
		end
		
		# Hausdorff
		f = rand([0, 1], n, n)
		tfm = Wenbo()
		f_dtm = transform!(boolean_indicator(f), tfm, threads)
		hausdorff_loss = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
		append!(hausdorff_mean_2D, BenchmarkTools.mean(hausdorff_loss).time)
		append!(hausdorff_std_2D, BenchmarkTools.std(hausdorff_loss).time)

		if has_cuda_gpu()
			f = CuArray(rand([0, 1], n, n))
			tfm = Wenbo()
			f_dtm = CuArray(transform!(boolean_indicator(f), tfm, threads))
			hausdorff_loss = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
			append!(hausdorff_mean_gpu_2D, BenchmarkTools.mean(hausdorff_loss).time)
			append!(hausdorff_std_gpu_2D, BenchmarkTools.std(hausdorff_loss).time)
		end
		
	end
end

# ╔═╡ 6c4632e3-ca49-431b-9ced-4c6269dd361e
let
	path = current_dir * "/loss_2D.csv"
	if has_cuda_gpu()
		df = DataFrame(
			sizes_loss_2D = Float64.(sizes_loss_2D),
			dice_mean_2D = Float64.(dice_mean_2D),
			dice_std_2D = Float64.(dice_std_2D),
			dice_mean_gpu_2D = Float64.(dice_mean_gpu_2D),
			dice_std_gpu_2D = Float64.(dice_std_gpu_2D),
			hausdorff_mean_2D = Float64.(hausdorff_mean_2D),
			hausdorff_std_2D = Float64.(hausdorff_std_2D),
			hausdorff_mean_gpu_2D = Float64.(hausdorff_mean_gpu_2D),
			hausdorff_std_gpu_2D = Float64.(hausdorff_std_gpu_2D)
		)
	else
		df = DataFrame(
			sizes_loss_2D = Float64.(sizes_loss_2D),
			dice_mean_2D = Float64.(dice_mean_2D),
			dice_std_2D = Float64.(dice_std_2D),
			hausdorff_mean_2D = Float64.(hausdorff_mean_2D),
			hausdorff_std_2D = Float64.(hausdorff_std_2D),
		)
	end
	CSV.write(path, df)
end

# ╔═╡ a1662c9c-1a78-404d-993a-870710eb6090
md"""
## 3D
"""

# ╔═╡ e6bc01a6-5f59-43e7-8ba2-32f8127ce6f3
begin
	dice_mean_3D = []
	dice_std_3D = []

	dice_mean_gpu_3D = []
	dice_std_gpu_3D = []

	hausdorff_mean_3D = []
	hausdorff_std_3D = []

	hausdorff_mean_gpu_3D = []
	hausdorff_std_gpu_3D = []

	sizes_loss_3D = []
	
	for n in num_range
		n = Int(round(n))
		_size = n^3
		push!(sizes_loss_3D, _size)
		@info "n = $(n)"
		
		# DICE
		f = Bool.(rand([0, 1], n, n, n))
		dice_loss = @benchmark dice($f, $f)
		append!(dice_mean_3D, BenchmarkTools.mean(dice_loss).time)
		append!(dice_std_3D, BenchmarkTools.std(dice_loss).time)

		if has_cuda_gpu()
			f = CuArray(Bool.(rand([0, 1], n, n, n)))
			dice_loss = @benchmark dice($f, $f)
			append!(dice_mean_gpu_3D, BenchmarkTools.mean(dice_loss).time)
			append!(dice_std_gpu_3D, BenchmarkTools.std(dice_loss).time)
		end
		
		# Hausdorff
		f = rand([0, 1], n, n, n)
		tfm = Wenbo()
		f_dtm = transform!(boolean_indicator(f), tfm, threads)
		hausdorff_loss = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
		append!(hausdorff_mean_3D, BenchmarkTools.mean(hausdorff_loss).time)
		append!(hausdorff_std_3D, BenchmarkTools.std(hausdorff_loss).time)

		if has_cuda_gpu()
			f = CuArray(rand([0, 1], n, n, n))
			tfm = Wenbo()
			f_dtm = CuArray(transform!(boolean_indicator(f), tfm, threads))
			hausdorff_loss = @benchmark hausdorff($f, $f, $f_dtm, $f_dtm)
			append!(hausdorff_mean_gpu_3D, BenchmarkTools.mean(hausdorff_loss).time)
			append!(hausdorff_std_gpu_3D, BenchmarkTools.std(hausdorff_loss).time)
		end
		
	end
end

# ╔═╡ c00689e0-fecb-4769-a53f-f18da7509982
let
	path = current_dir * "/loss_3D.csv"
	if has_cuda_gpu()
		df = DataFrame(
			sizes_loss_3D = Float64.(sizes_loss_3D),
			dice_mean_3D = Float64.(dice_mean_3D),
			dice_std_3D = Float64.(dice_std_3D),
			dice_mean_gpu_3D = Float64.(dice_mean_gpu_3D),
			dice_std_gpu_3D = Float64.(dice_std_gpu_3D),
			hausdorff_mean_3D = Float64.(hausdorff_mean_3D),
			hausdorff_std_3D = Float64.(hausdorff_std_3D),
			hausdorff_mean_gpu_3D = Float64.(hausdorff_mean_gpu_3D),
			hausdorff_std_gpu_3D = Float64.(hausdorff_std_gpu_3D)
		)
	else
		df = DataFrame(
			sizes_loss_3D = Float64.(sizes_loss_3D),
			dice_mean_3D = Float64.(dice_mean_3D),
			dice_std_3D = Float64.(dice_std_3D),
			hausdorff_mean_3D = Float64.(hausdorff_mean_3D),
			hausdorff_std_3D = Float64.(hausdorff_std_3D),
		)
	end
	CSV.write(path, df)
end

# ╔═╡ Cell order:
# ╠═7ef078de-23f9-11ed-104e-5f232d5f92b1
# ╠═bb360c60-7eb3-4fea-91c9-fc53bc643c45
# ╠═bcdcca5c-4eba-4cb5-a955-897f0ef99120
# ╟─708245f0-ced3-4f1d-9062-140e39fed6f7
# ╟─e1bcf8e2-7824-4443-8e0c-4800d20a4cbb
# ╠═578eda3b-8f90-4992-87da-d44950f4b891
# ╠═064702c2-cb91-4c19-baab-4586b710f9cf
# ╠═e313c086-ccf4-44ed-947d-5582c72f9b39
# ╠═b23d7941-8cd0-49f6-bdac-f5bb05de2cdd
# ╟─788654ed-4bd0-4f1a-8a4d-007448de19f9
# ╠═bfdc1c54-546b-4ab0-8312-b48ce1ed557e
# ╠═f5189fc3-fea9-4e57-a4b3-b9ae1a8c0c10
# ╟─2d77bfc2-0457-435f-af13-6b70ef07c17d
# ╟─1d7dc8c9-650c-4af1-a570-04fdd538df67
# ╠═9cc8791a-f577-475f-a96d-93d01edcb35b
# ╠═6c4632e3-ca49-431b-9ced-4c6269dd361e
# ╟─a1662c9c-1a78-404d-993a-870710eb6090
# ╠═e6bc01a6-5f59-43e7-8ba2-32f8127ce6f3
# ╠═c00689e0-fecb-4769-a53f-f18da7509982
