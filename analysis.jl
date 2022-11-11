### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 9b9938f3-a49e-43e4-abf8-478216eefc58
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	
	using Revise, PlutoUI, BenchmarkTools, CairoMakie, DataFrames, CSV, CUDA, DistanceTransforms
end

# ╔═╡ 912daec8-3954-4b64-908d-75175ce2dfd2
TableOfContents()

# ╔═╡ f1da69f4-72c4-40cc-b51e-d38e0cc8ecd1
medphys_theme = Theme(;
    Axis=(
        backgroundcolor=:white,
        xgridcolor=:gray,
        xgridwidth=0.5,
        xlabelfont=:Helvetica,
        xticklabelfont=:Helvetica,
        xlabelsize=20,
        xticklabelsize=20,
        ygridcolor=:gray,
        ygridwidth=0.5,
        ylabelfont=:Helvetica,
        yticklabelfont=:Helvetica,
        ylabelsize=20,
        yticklabelsize=20,
        bottomsplinecolor=:black,
        leftspinecolor=:black,
        titlefont=:Helvetica,
        titlesize=30,
    ),
)

# ╔═╡ aafe75ac-98d1-4050-a2cf-45574c2613c5
current_dir = pwd()

# ╔═╡ 86a7ccf5-5de0-4da4-a6e8-6afc71b515a7
folders = readdir(current_dir)

# ╔═╡ 25a6f92a-f171-49b1-b428-95f76dd852a8
md"""
# Distance Transforms
"""

# ╔═╡ f2727c97-e288-4a49-a345-ade5058de27b
julia_dir = joinpath(current_dir, "Julia", "Results")

# ╔═╡ 3196c8b0-435e-41c7-83a9-c350dade38aa
python_dir = joinpath(current_dir, "Python", "Results")

# ╔═╡ 6382c6cd-9d68-4e65-862e-79af21b86ff6
df_dt_2D = CSV.read(joinpath(julia_dir, "DT and Loss", "dt_2D.csv"), DataFrame);

# ╔═╡ d3805664-6315-4129-831e-3d2738baeeb6
python_df_dt_2D = CSV.read(joinpath(python_dir, "DT and Loss", "purePython_DT_2D_Nov06.csv"), DataFrame);

# ╔═╡ da760574-b39f-4766-91fe-96ef9638d2fa
df_dt_2D

# ╔═╡ bfae16ab-a6f9-4659-8300-ef3a4d8db9d8
python_df_dt_2D

# ╔═╡ 8284bfba-bf32-482e-8310-95578c75f64d
python_sizes_2D = python_df_dt_2D[!, :sizes_2D] .^ 2

# ╔═╡ a6bc73aa-573b-466c-bb8a-7bb54c376a0e
md"""
## 2D
"""

# ╔═╡ fba53777-11fc-4e40-b780-6a83c44c8f93
md"""
### Felzenszwalb
"""

# ╔═╡ e1c30865-7110-4e67-8338-58596206c9c6
function dt_2D_felzenszwalb()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Felzenszwalb 2D)"

	df = df_dt_2D

	sc2 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_mean_2D], df[!, :felzenszwalb_std_2D])

	sc3 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_inplace_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_inplace_mean_2D], df[!, :felzenszwalb_inplace_std_2D])

	sc4 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D], df[!, :felzenszwalb_threaded_std_2D])

	sc5 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth], df[!, :felzenszwalb_threaded_std_2D_depth])

	sc6 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_nonthread])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_nonthread], df[!, :felzenszwalb_threaded_std_2D_nonthread])

	sc7 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_worksteal])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_worksteal], df[!, :felzenszwalb_threaded_std_2D_worksteal])

	# if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D], df[!, :felzenszwalb_gpu_std_2D])
	# end

	# if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	# else
	# 	f[1, 2] = Legend(
	#         f,
	#         [sc2, sc3, sc4, sc5, sc6, sc7,],
	#         ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	#         framevisible=false,
	#     )
	# end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ e4e5e57b-84be-4709-8c12-b1e5d8d8ee97
with_theme(medphys_theme) do
    dt_2D_felzenszwalb()
end

# ╔═╡ a1e36fb1-0313-4f3c-811a-630e37324558
md"""
The fastest `Felzenszwalb`s are
1. DepthFirstEx
2. WorkStealingEx
"""

# ╔═╡ 876532f2-4963-4b42-b9a9-75a9da037cd8
md"""
### Wenbo
"""

# ╔═╡ e1c79280-342d-4af6-849b-1d010c884b47
function dt_2D_wenbo()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Wenbo 2D)"

	df = df_dt_2D

	sc8 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_inplace_mean_2D])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_inplace_mean_2D], df[!, :wenbo_inplace_std_2D])
	
	sc9 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D], df[!, :wenbo_threaded_std_2D])
	
	sc10 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth], df[!, :wenbo_threaded_std_2D_depth])
	
	sc11 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_nonthread])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_nonthread], df[!, :wenbo_threaded_std_2D_nonthread])
	
	sc12 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_worksteal])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_worksteal], df[!, :wenbo_threaded_std_2D_worksteal])
	
	# if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D], df[!, :wenbo_gpu_std_2D])
	# end

	# if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	# else
	# 	f[1, 2] = Legend(
	#         f,
	#         [sc8, sc9, sc10, sc11, sc12],
	#         ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	#         framevisible=false,
	#     )
	# end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 0bc66118-964b-4225-9d51-69edb40f1f29
with_theme(medphys_theme) do
    dt_2D_wenbo()
end

# ╔═╡ 8c2c5919-8dd2-45e3-8345-c9f0196fd50e
md"""
The fastest `Wenbo`s are
1. Wenbo GPU
2. Wenbo Threaded
3. Wenbo DepthFirstEx
"""

# ╔═╡ f2b9b359-f7c8-433a-ad06-de1a95a78624
md"""
### Python
"""

# ╔═╡ 28324364-945a-43ff-a6c5-2b6b6db1e90b
function dt_2D_python()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Wenbo 2D)"

	df = python_df_dt_2D
	
	# sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :dt_mean_cpu_2D])
	sc1 = scatter!(ax1, python_sizes_2D, df[!, :dt_mean_cpu_2D])

			f[1, 2] = Legend(
	        f,
	        [sc1],
	        ["Python CPU"];
	        framevisible=false,
	    )
	
	return f
end

# ╔═╡ a8898897-4b11-4265-99b0-e9b8b84f962c
with_theme(medphys_theme) do
    dt_2D_python()
end

# ╔═╡ 182b9ad3-b59e-4f12-a3ba-704049fcdb9a
md"""
### Fastest + Python
"""

# ╔═╡ 6cbde940-9b99-40c5-9ca6-0c68776e6800
function dt_2D_fastest()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"

	df_py = python_df_dt_2D
	df = df_dt_2D
	
    sc1 = scatter!(ax1, python_sizes_2D, df_py[!, :dt_mean_cpu_2D])
	
	sc2 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D])

	sc2_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D])
	
	
	f[1, 2] = Legend(
		f,
		[sc1, sc2, sc2_gpu],
		["Python", "Wenbo Threaded", "Wenbo GPU"];
		framevisible=false,
	)

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=6e5)
	
	return f
end

# ╔═╡ cba8582d-c9ea-4d39-a197-537985b7bd29
with_theme(medphys_theme) do
    dt_2D_fastest()
end

# ╔═╡ 4c4c8444-f15e-4a59-951f-b3a65a923bd4
begin
	df_dt_2d_python_wenbo = DataFrame(
		input_size = round.(python_sizes_2D, sigdigits=1),
		wenbo_threaded_dt = df_dt_2D[!, :wenbo_threaded_mean_2D],
		wenbo_gpu_dt = df_dt_2D[!, :wenbo_gpu_mean_2D],
		python_dt = python_df_dt_2D[!, :dt_mean_cpu_2D]
	)
	df_dt_2d_python_wenbo[!, :speedup_threaded] = df_dt_2d_python_wenbo[!, :python_dt] ./ df_dt_2d_python_wenbo[!, :wenbo_threaded_dt]
	
	df_dt_2d_python_wenbo[!, :speedup_gpu] = df_dt_2d_python_wenbo[!, :python_dt] ./ df_dt_2d_python_wenbo[!, :wenbo_gpu_dt]
	
	df_dt_2d_python_wenbo
end

# ╔═╡ 69d55f25-7b57-4589-b0ad-3bdbf4a256e6
# function dt_2D_all()
#     f = Figure()
#     ax1 = Axis(f[1, 1])
# 	ax1.xlabel = "Number of Elements"
#     ax1.ylabel = "Time (ns)"
#     ax1.title = "Distance Transforms (All 2D)"

# 	df = df_dt_2D

	
#     sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D], df[!, :maurer_std_2D])

# 	sc2 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_mean_2D])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_mean_2D], df[!, :felzenszwalb_std_2D])

# 	sc3 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_inplace_mean_2D])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_inplace_mean_2D], df[!, :felzenszwalb_inplace_std_2D])

# 	sc4 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D], df[!, :felzenszwalb_threaded_std_2D])

# 	sc5 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth], df[!, :felzenszwalb_threaded_std_2D_depth])

# 	sc6 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_nonthread])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_nonthread], df[!, :felzenszwalb_threaded_std_2D_nonthread])

# 	sc7 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_worksteal])
#     # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_worksteal], df[!, :felzenszwalb_threaded_std_2D_worksteal])

# 	if has_cuda_gpu()
# 		sc7_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D])
# 	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D], df[!, :felzenszwalb_gpu_std_2D])
# 	end

# 	sc8 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_inplace_mean_2D])
# 	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_inplace_mean_2D], df[!, :wenbo_inplace_std_2D])
	
# 	sc9 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D])
# 	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D], df[!, :wenbo_threaded_std_2D])
	
# 	sc10 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth])
# 	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth], df[!, :wenbo_threaded_std_2D_depth])
	
# 	sc11 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_nonthread])
# 	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_nonthread], df[!, :wenbo_threaded_std_2D_nonthread])
	
# 	sc12 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_worksteal])
# 	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_worksteal], df[!, :wenbo_threaded_std_2D_worksteal])
	
# 	if has_cuda_gpu()
# 	    sc12_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D])
# 	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D], df[!, :wenbo_gpu_std_2D])
# 	end

# 	if has_cuda_gpu()
# 		f[1, 2] = Legend(
# 	        f,
# 	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu, sc8, sc9, sc10, sc11, sc12, sc12_gpu],
# 	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
# 	        framevisible=false,
# 	    )
# 	else
# 		f[1, 2] = Legend(
# 	        f,
# 	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12],
# 	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
# 	        framevisible=false,
# 	    )
# 	end

# 	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

# 	# xlims!(ax1; low=3.5e4, high=4.5e4)
# 	# ylims!(ax1; low=0, high=4e5)
	
# 	return f
# end

# ╔═╡ 12b298f3-bcd2-41bc-a95a-ed53f8996f36
# with_theme(medphys_theme) do
#     dt_2D_all()
# end

# ╔═╡ eedf9f9c-ba03-4453-9956-ce998c149faf
md"""
## 3D
"""

# ╔═╡ fe306014-5740-4f53-994a-2e2facfde635
df_dt_3D = CSV.read(joinpath(julia_dir, "DT and Loss", "dt_3D.csv"), DataFrame);

# ╔═╡ 628af84d-57be-4e3d-a685-a02e07f2a183
python_df_dt_3D = CSV.read(joinpath(python_dir, "DT and Loss", "purePython_DT_3D_Nov06.csv"), DataFrame);

# ╔═╡ 584fb533-17e1-414f-9497-a3fccfff291d
python_sizes_3D = python_df_dt_3D[!, :sizes_3D] .^3

# ╔═╡ 53304723-cbbb-4eed-98a1-5538e4a00f6c
df_dt_3D

# ╔═╡ 029d3260-3a46-46c4-81ba-5d3b5a17166c
python_df_dt_3D

# ╔═╡ bc9ce358-5684-45d5-84b9-e93fcd0324b5
md"""
### Felzenszwalb
"""

# ╔═╡ b40d34b1-8d01-404f-9df7-1d7babbe04f2
function dt_3D_felzenszwalb()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Felzenszwalb 3D)"

	df = df_dt_3D

	sc2 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_mean_3D], df[!, :felzenszwalb_std_3D])

	sc3 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_inplace_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_inplace_mean_3D], df[!, :felzenszwalb_inplace_std_3D])

	sc4 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D], df[!, :felzenszwalb_threaded_std_3D])

	sc5 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth], df[!, :felzenszwalb_threaded_std_3D_depth])

	sc6 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_nonthread])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_nonthread], df[!, :felzenszwalb_threaded_std_3D_nonthread])

	sc7 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_worksteal])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_worksteal], df[!, :felzenszwalb_threaded_std_3D_worksteal])

	# if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D], df[!, :felzenszwalb_gpu_std_3D])
	# end

	# if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	# else
	# 	f[1, 2] = Legend(
	#         f,
	#         [sc2, sc3, sc4, sc5, sc6, sc7,],
	#         ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	#         framevisible=false,
	#     )
	# end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 3df49e8b-87fe-431e-a124-37acd8bb297f
with_theme(medphys_theme) do
    dt_3D_felzenszwalb()
end

# ╔═╡ e55b10e6-acc2-4911-ace7-cb88c57db206
md"""
The fastest `Felzenszwalb`s are
1. GPU
2. DepthFirstEx
3. WorkStealingEx
"""

# ╔═╡ c59b4ba3-983f-4156-afb4-05d9b9e96923
md"""
### Wenbo
"""

# ╔═╡ d68c8eb4-5840-48e0-9758-f96ce4432714
function dt_3D_wenbo()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Wenbo 3D)"

	df = df_dt_3D

	sc8 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_inplace_mean_3D])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_inplace_mean_3D], df[!, :wenbo_inplace_std_3D])
	
	sc9 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D], df[!, :wenbo_threaded_std_3D])
	
	sc10 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth], df[!, :wenbo_threaded_std_3D_depth])
	
	sc11 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_nonthread])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_nonthread], df[!, :wenbo_threaded_std_3D_nonthread])
	
	sc12 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_worksteal])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_worksteal], df[!, :wenbo_threaded_std_3D_worksteal])
	
	# if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D], df[!, :wenbo_gpu_std_3D])
	# end

	# if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	# else
	# 	f[1, 2] = Legend(
	#         f,
	#         [sc8, sc9, sc10, sc11, sc12],
	#         ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	#         framevisible=false,
	#     )
	# end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 5edc4c62-b895-4d4e-890c-51f1cc36024c
with_theme(medphys_theme) do
    dt_3D_wenbo()
end

# ╔═╡ 31d50495-8bcc-4cd9-9e1f-dd75cb5447ce
md"""
### Python
"""

# ╔═╡ 6e9e98ef-b527-49db-96ee-695c3f066097
function dt_3D_python()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Wenbo 2D)"

	df = python_df_dt_3D

	sc1 = scatter!(ax1, python_sizes_3D, df[!, :dt_mean_cpu_3D])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_inplace_mean_2D], df[!, :wenbo_inplace_std_2D])

			f[1, 2] = Legend(
	        f,
	        [sc1],
	        ["Python CPU"];
	        framevisible=false,
	    )
	
	return f
end

# ╔═╡ f5e0137e-9f66-4acf-943d-bbfa93cacbc1
with_theme(medphys_theme) do
    dt_3D_python()
end

# ╔═╡ a6efcd4d-bf49-40f7-80c8-07234fc61d0d
md"""
### Fastest + Python
"""

# ╔═╡ a04fff7d-a885-46c3-acf5-095efbbb3227
function dt_3D_fastest()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (3D)"

	df_py = python_df_dt_3D
	df = df_dt_3D
	
    sc1 = scatter!(ax1, python_sizes_3D, df_py[!, :dt_mean_cpu_3D])
	
	sc2 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D])

	sc2_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D])
	
	
	f[1, 2] = Legend(
		f,
		[sc1, sc2, sc2_gpu],
		["Python", "Wenbo Threaded", "Wenbo GPU"];
		framevisible=false,
	)

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=6e5)
	
	return f
end

# ╔═╡ fce529ba-1f4c-4930-a980-ab88c0a516a5
with_theme(medphys_theme) do
    dt_3D_fastest()
end

# ╔═╡ ab0c6b51-15cf-4f1d-bf64-d8f2a3eeedc3
begin
	df_dt_3d_python_wenbo = DataFrame(
		input_size = round.(python_sizes_3D, sigdigits=1),
		wenbo_threaded_dt = df_dt_3D[!, :wenbo_threaded_mean_3D],
		wenbo_gpu_dt = df_dt_3D[!, :wenbo_gpu_mean_3D],
		python_dt = python_df_dt_3D[!, :dt_mean_cpu_3D]
	)
	df_dt_3d_python_wenbo[!, :speedup_threaded] = df_dt_3d_python_wenbo[!, :python_dt] ./ df_dt_3d_python_wenbo[!, :wenbo_threaded_dt]
	
	df_dt_3d_python_wenbo[!, :speedup_gpu] = df_dt_3d_python_wenbo[!, :python_dt] ./ df_dt_3d_python_wenbo[!, :wenbo_gpu_dt]
	
	df_dt_3d_python_wenbo
end

# ╔═╡ 288bcb17-f5d3-432e-bc1b-d27734c39c57
# function dt_3D_all()
#     f = Figure()
#     ax1 = Axis(f[1, 1])
# 	ax1.xlabel = "Number of Elements"
#     ax1.ylabel = "Time (ns)"
#     ax1.title = "Distance Transforms (All 3D)"

# 	df = df_dt_3D

	
#     sc1 = scatter!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D], df[!, :maurer_std_3D])

# 	sc2 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_mean_3D])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_mean_3D], df[!, :felzenszwalb_std_3D])

# 	sc3 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_inplace_mean_3D])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_inplace_mean_3D], df[!, :felzenszwalb_inplace_std_3D])

# 	sc4 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D], df[!, :felzenszwalb_threaded_std_3D])

# 	sc5 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth], df[!, :felzenszwalb_threaded_std_3D_depth])

# 	sc6 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_nonthread])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_nonthread], df[!, :felzenszwalb_threaded_std_3D_nonthread])

# 	sc7 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_worksteal])
#     # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_worksteal], df[!, :felzenszwalb_threaded_std_3D_worksteal])

# 	if has_cuda_gpu()
# 		sc7_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D])
# 	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D], df[!, :felzenszwalb_gpu_std_3D])
# 	end

# 	sc8 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_inplace_mean_3D])
# 	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_inplace_mean_3D], df[!, :wenbo_inplace_std_3D])
	
# 	sc9 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D])
# 	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D], df[!, :wenbo_threaded_std_3D])
	
# 	sc10 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth])
# 	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth], df[!, :wenbo_threaded_std_3D_depth])
	
# 	sc11 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_nonthread])
# 	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_nonthread], df[!, :wenbo_threaded_std_3D_nonthread])
	
# 	sc12 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_worksteal])
# 	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_worksteal], df[!, :wenbo_threaded_std_3D_worksteal])
	
# 	if has_cuda_gpu()
# 	    sc12_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D])
# 	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D], df[!, :wenbo_gpu_std_3D])
# 	end

# 	if has_cuda_gpu()
# 		f[1, 2] = Legend(
# 	        f,
# 	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu, sc8, sc9, sc10, sc11, sc12, sc12_gpu],
# 	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
# 	        framevisible=false,
# 	    )
# 	else
# 		f[1, 2] = Legend(
# 	        f,
# 	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12],
# 	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
# 	        framevisible=false,
# 	    )
# 	end

# 	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

# 	# xlims!(ax1; low=3.5e4, high=4.5e4)
# 	# ylims!(ax1; low=0, high=4e5)
	
# 	return f
# end

# ╔═╡ 199a5460-3b24-4b4e-b392-d60785166595
# with_theme(medphys_theme) do
#     dt_3D_all()
# end

# ╔═╡ a7c614f9-9f25-468f-9468-8c8de6bef1cb
md"""
# Loss Functions
"""

# ╔═╡ 465a96d1-39fa-48ee-a782-54855ece6258
md"""
## 2D
"""

# ╔═╡ cbc474d6-4c81-4d20-98c5-0ae9cf780a0a
df_loss_2D = CSV.read(joinpath(julia_dir, "DT and Loss", "loss_2D.csv"), DataFrame);

# ╔═╡ 5a43898a-766a-409d-b9e2-7514a417c202
python_df_loss_2D = CSV.read(joinpath(python_dir, "DT and Loss", "purePython_Loss_2D_nov6.csv"), DataFrame);

# ╔═╡ df2bb3cc-3bd7-4973-88f3-d1e575ada290
python_df_loss_2D

# ╔═╡ b505a057-32e7-4011-ba6a-366d0aee478d
function loss_2D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Loss (2D)"

	df = df_loss_2D

    sc1 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_2D])
	sc1_gpu = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_gpu_2D])

	sc2 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_2D])
	sc2_gpu = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_gpu_2D])

	sc3 = scatter!(ax1, df[!, :sizes_loss_2D], python_df_loss_2D[!, :hd_mean_2D_purePython])
	sc4 = scatter!(ax1, df[!, :sizes_loss_2D], python_df_loss_2D[!, :dice_mean_2D_purePython])
	
	f[1, 2] = Legend(
		f,
		[sc1, sc1_gpu, sc2, sc2_gpu, sc3, sc4],
		["Dice", "DICE GPU", "Hausdorff", "Hausdorff GPU", "Python Hausdorff", "Python DICE"];
		framevisible=false,
	)

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 764c9cfd-c0c2-44bd-8119-5d488bae4e1f
with_theme(medphys_theme) do
    loss_2D_all()
end

# ╔═╡ d2024756-0d6e-43ea-b833-f1a9b5bd15ee
md"""
## 3D
"""

# ╔═╡ 57a9afe8-60d4-41c6-b571-c9d3f835e23a
df_loss_3D = CSV.read(joinpath(julia_dir, "DT and Loss", "loss_3D.csv"), DataFrame);

# ╔═╡ 0183a4c2-9017-40ba-b28e-74370e51777c
python_df_loss_3D = CSV.read(joinpath(python_dir, "DT and Loss", "purePython_Loss_3D_nov6.csv"), DataFrame);

# ╔═╡ 6267b4f3-5658-4ad3-bf1b-918f35074d91
python_df_loss_3D

# ╔═╡ 7acbac7b-9009-4b9f-8d80-c6bd3de11baa
function loss_3D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Loss (3D)"

	df = df_loss_3D

    sc1 = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :dice_mean_3D])
	sc1_gpu = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :dice_mean_gpu_3D])

	sc2 = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_3D])
	sc2_gpu = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_gpu_3D])

	sc3 = scatter!(ax1, df[!, :sizes_loss_3D], python_df_loss_3D[!, :hd_mean_3D_purePython])
	sc4 = scatter!(ax1, df[!, :sizes_loss_3D], python_df_loss_3D[!, :dice_mean_3D_purePython])
	
	f[1, 2] = Legend(
		f,
		[sc1, sc1_gpu, sc2, sc2_gpu, sc3, sc4],
		["Dice", "DICE GPU", "Hausdorff", "Hausdorff GPU", "Python Hausdorff", "Python DICE"];
		framevisible=false,
	)

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 96674796-f25b-4957-a467-41bae3fdefe3
with_theme(medphys_theme) do
    loss_3D_all()
end

# ╔═╡ Cell order:
# ╠═9b9938f3-a49e-43e4-abf8-478216eefc58
# ╠═912daec8-3954-4b64-908d-75175ce2dfd2
# ╠═f1da69f4-72c4-40cc-b51e-d38e0cc8ecd1
# ╠═aafe75ac-98d1-4050-a2cf-45574c2613c5
# ╠═86a7ccf5-5de0-4da4-a6e8-6afc71b515a7
# ╟─25a6f92a-f171-49b1-b428-95f76dd852a8
# ╠═f2727c97-e288-4a49-a345-ade5058de27b
# ╠═3196c8b0-435e-41c7-83a9-c350dade38aa
# ╠═6382c6cd-9d68-4e65-862e-79af21b86ff6
# ╠═d3805664-6315-4129-831e-3d2738baeeb6
# ╠═da760574-b39f-4766-91fe-96ef9638d2fa
# ╠═bfae16ab-a6f9-4659-8300-ef3a4d8db9d8
# ╠═8284bfba-bf32-482e-8310-95578c75f64d
# ╟─a6bc73aa-573b-466c-bb8a-7bb54c376a0e
# ╟─fba53777-11fc-4e40-b780-6a83c44c8f93
# ╟─e1c30865-7110-4e67-8338-58596206c9c6
# ╟─e4e5e57b-84be-4709-8c12-b1e5d8d8ee97
# ╟─a1e36fb1-0313-4f3c-811a-630e37324558
# ╟─876532f2-4963-4b42-b9a9-75a9da037cd8
# ╟─e1c79280-342d-4af6-849b-1d010c884b47
# ╟─0bc66118-964b-4225-9d51-69edb40f1f29
# ╟─8c2c5919-8dd2-45e3-8345-c9f0196fd50e
# ╟─f2b9b359-f7c8-433a-ad06-de1a95a78624
# ╟─28324364-945a-43ff-a6c5-2b6b6db1e90b
# ╟─a8898897-4b11-4265-99b0-e9b8b84f962c
# ╟─182b9ad3-b59e-4f12-a3ba-704049fcdb9a
# ╟─6cbde940-9b99-40c5-9ca6-0c68776e6800
# ╟─cba8582d-c9ea-4d39-a197-537985b7bd29
# ╠═4c4c8444-f15e-4a59-951f-b3a65a923bd4
# ╟─69d55f25-7b57-4589-b0ad-3bdbf4a256e6
# ╟─12b298f3-bcd2-41bc-a95a-ed53f8996f36
# ╟─eedf9f9c-ba03-4453-9956-ce998c149faf
# ╠═fe306014-5740-4f53-994a-2e2facfde635
# ╠═628af84d-57be-4e3d-a685-a02e07f2a183
# ╠═584fb533-17e1-414f-9497-a3fccfff291d
# ╠═53304723-cbbb-4eed-98a1-5538e4a00f6c
# ╠═029d3260-3a46-46c4-81ba-5d3b5a17166c
# ╟─bc9ce358-5684-45d5-84b9-e93fcd0324b5
# ╟─b40d34b1-8d01-404f-9df7-1d7babbe04f2
# ╟─3df49e8b-87fe-431e-a124-37acd8bb297f
# ╟─e55b10e6-acc2-4911-ace7-cb88c57db206
# ╟─c59b4ba3-983f-4156-afb4-05d9b9e96923
# ╟─d68c8eb4-5840-48e0-9758-f96ce4432714
# ╠═5edc4c62-b895-4d4e-890c-51f1cc36024c
# ╟─31d50495-8bcc-4cd9-9e1f-dd75cb5447ce
# ╟─6e9e98ef-b527-49db-96ee-695c3f066097
# ╟─f5e0137e-9f66-4acf-943d-bbfa93cacbc1
# ╟─a6efcd4d-bf49-40f7-80c8-07234fc61d0d
# ╟─a04fff7d-a885-46c3-acf5-095efbbb3227
# ╟─fce529ba-1f4c-4930-a980-ab88c0a516a5
# ╠═ab0c6b51-15cf-4f1d-bf64-d8f2a3eeedc3
# ╟─288bcb17-f5d3-432e-bc1b-d27734c39c57
# ╟─199a5460-3b24-4b4e-b392-d60785166595
# ╟─a7c614f9-9f25-468f-9468-8c8de6bef1cb
# ╟─465a96d1-39fa-48ee-a782-54855ece6258
# ╠═cbc474d6-4c81-4d20-98c5-0ae9cf780a0a
# ╠═5a43898a-766a-409d-b9e2-7514a417c202
# ╠═df2bb3cc-3bd7-4973-88f3-d1e575ada290
# ╟─b505a057-32e7-4011-ba6a-366d0aee478d
# ╟─764c9cfd-c0c2-44bd-8119-5d488bae4e1f
# ╟─d2024756-0d6e-43ea-b833-f1a9b5bd15ee
# ╠═57a9afe8-60d4-41c6-b571-c9d3f835e23a
# ╠═0183a4c2-9017-40ba-b28e-74370e51777c
# ╠═6267b4f3-5658-4ad3-bf1b-918f35074d91
# ╟─7acbac7b-9009-4b9f-8d80-c6bd3de11baa
# ╠═96674796-f25b-4957-a467-41bae3fdefe3
