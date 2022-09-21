### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 9b9938f3-a49e-43e4-abf8-478216eefc58
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
	using DistanceTransforms
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

# ╔═╡ 25a6f92a-f171-49b1-b428-95f76dd852a8
md"""
# Distance Transforms
"""

# ╔═╡ 6382c6cd-9d68-4e65-862e-79af21b86ff6
df_dt_2D = CSV.read(current_dir * "/dt_2D.csv", DataFrame);

# ╔═╡ da760574-b39f-4766-91fe-96ef9638d2fa
df_dt_2D

# ╔═╡ a6bc73aa-573b-466c-bb8a-7bb54c376a0e
md"""
## 2D
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

	if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D], df[!, :felzenszwalb_gpu_std_2D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7,],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	        framevisible=false,
	    )
	end

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
2. Threaded
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
	
	if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D], df[!, :wenbo_gpu_std_2D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	        framevisible=false,
	    )
	end

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
1. Wenbo Threaded
2. Wenbo DepthFirstEx
"""

# ╔═╡ 6cbde940-9b99-40c5-9ca6-0c68776e6800
function dt_2D_fastest()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Fastest 2D)"

	df = df_dt_2D

	
    sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D], df[!, :maurer_std_2D])

	sc4 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D], df[!, :felzenszwalb_threaded_std_2D])

	sc5 = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_threaded_mean_2D_depth], df[!, :felzenszwalb_threaded_std_2D_depth])
	
	sc9 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D], df[!, :wenbo_threaded_std_2D])
	
	sc10 = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth])
	# errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_threaded_mean_2D_depth], df[!, :wenbo_threaded_std_2D_depth])
	
	f[1, 2] = Legend(
		f,
		[sc1, sc4, sc5, sc9, sc10],
		["Maurer", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Wenbo Threaded", "Wenbo DepthFirstEx"];
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

# ╔═╡ 69d55f25-7b57-4589-b0ad-3bdbf4a256e6
function dt_2D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (All 2D)"

	df = df_dt_2D

	
    sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :maurer_mean_2D], df[!, :maurer_std_2D])

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

	if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :felzenszwalb_gpu_mean_2D], df[!, :felzenszwalb_gpu_std_2D])
	end

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
	
	if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D])
	    # errorbars!(ax1, df[!, :sizes_2D], df[!, :wenbo_gpu_mean_2D], df[!, :wenbo_gpu_std_2D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu, sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	        framevisible=false,
	    )
	end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 12b298f3-bcd2-41bc-a95a-ed53f8996f36
with_theme(medphys_theme) do
    dt_2D_all()
end

# ╔═╡ eedf9f9c-ba03-4453-9956-ce998c149faf
md"""
## 3D
"""

# ╔═╡ fe306014-5740-4f53-994a-2e2facfde635
df_dt_3D = CSV.read(current_dir * "/dt_3D.csv", DataFrame);

# ╔═╡ 53304723-cbbb-4eed-98a1-5538e4a00f6c
df_dt_3D

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

	if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D], df[!, :felzenszwalb_gpu_std_3D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc2, sc3, sc4, sc5, sc6, sc7,],
	        ["Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx"];
	        framevisible=false,
	    )
	end

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
1. DepthFirstEx
2. Threaded
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
	
	if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D], df[!, :wenbo_gpu_std_3D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc8, sc9, sc10, sc11, sc12],
	        ["Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	        framevisible=false,
	    )
	end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 5edc4c62-b895-4d4e-890c-51f1cc36024c
with_theme(medphys_theme) do
    dt_3D_wenbo()
end

# ╔═╡ 22443020-3cf4-4ca3-ad18-0eb8b8017dba
md"""
The fastest `Wenbo`s are
1. Wenbo Threaded
2. Wenbo DepthFirstEx
"""

# ╔═╡ 3a15a80a-5409-42e6-b61e-623029fb3d2c
function dt_3D_fastest()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (Fastest 3D)"

	df = df_dt_3D

	
    sc1 = scatter!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D], df[!, :maurer_std_3D])

	sc4 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D], df[!, :felzenszwalb_threaded_std_3D])

	sc5 = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_threaded_mean_3D_depth], df[!, :felzenszwalb_threaded_std_3D_depth])
	
	sc9 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D], df[!, :wenbo_threaded_std_3D])
	
	sc10 = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth])
	# errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_threaded_mean_3D_depth], df[!, :wenbo_threaded_std_3D_depth])
	
	f[1, 2] = Legend(
		f,
		[sc1, sc4, sc5, sc9, sc10],
		["Maurer", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Wenbo Threaded", "Wenbo DepthFirstEx"];
		framevisible=false,
	)

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=6e5)
	
	return f
end

# ╔═╡ 2c8e3cff-73ae-439b-a0be-b1c2a934fa16
with_theme(medphys_theme) do
    dt_3D_fastest()
end

# ╔═╡ 288bcb17-f5d3-432e-bc1b-d27734c39c57
function dt_3D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (All 3D)"

	df = df_dt_3D

	
    sc1 = scatter!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :maurer_mean_3D], df[!, :maurer_std_3D])

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

	if has_cuda_gpu()
		sc7_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :felzenszwalb_gpu_mean_3D], df[!, :felzenszwalb_gpu_std_3D])
	end

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
	
	if has_cuda_gpu()
	    sc12_gpu = scatter!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D])
	    # errorbars!(ax1, df[!, :sizes_3D], df[!, :wenbo_gpu_mean_3D], df[!, :wenbo_gpu_std_3D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc7_gpu, sc8, sc9, sc10, sc11, sc12, sc12_gpu],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Felzenszwalb GPU", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx", "Wenbo GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2, sc3, sc4, sc5, sc6, sc7, sc8, sc9, sc10, sc11, sc12],
	        ["Maurer", "Felzenszwalb", "Felzenszwalb In-Place", "Felzenszwalb Threaded", "Felzenszwalb DepthFirstEx", "Felzenszwalb NonThreadedEx", "Felzenszwalb WorkStealingEx", "Wenbo In-Place", "Wenbo Threaded", "Wenbo DepthFirstEx", "Wenbo NonThreadedEx", "Wenbo WorkStealingEx"];
	        framevisible=false,
	    )
	end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 199a5460-3b24-4b4e-b392-d60785166595
with_theme(medphys_theme) do
    dt_3D_all()
end

# ╔═╡ d51b95e0-64f6-4c38-b1b3-2537d5f5c6da
df_dt_3D[!, :wenbo_threaded_mean_3D]

# ╔═╡ 0a5c4032-33e0-466c-a1bb-fa0b22d33cba
df_dt_3D[!, :maurer_mean_3D]

# ╔═╡ a7c614f9-9f25-468f-9468-8c8de6bef1cb
md"""
# Loss Functions
"""

# ╔═╡ 465a96d1-39fa-48ee-a782-54855ece6258
md"""
## 2D
"""

# ╔═╡ e92af4e2-d32c-4a8f-9f82-729ba50580e6
df_loss_2D = CSV.read(current_dir * "/loss_2D.csv", DataFrame);

# ╔═╡ b505a057-32e7-4011-ba6a-366d0aee478d
function loss_2D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (All 2D)"

	df = df_loss_2D

    sc1 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :dice_mean_2D], df[!, :dice_std_2D])

	if has_cuda_gpu()
	    sc1_gpu = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_gpu_2D])
	    # errorbars!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_gpu_2D], df[!, :dice_gpu_std_2D])
	end

	sc2 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_2D])
	# errorbars!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_2D], df[!, :hausdorff_std_2D])
	
	
	if has_cuda_gpu()
		sc2_gpu = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_gpu_2D])
		# errorbars!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_gpu_2D], df[!, :hausdorff_std_gpu_2D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc1, sc1_gpu, sc2, sc2_gpu],
	        ["Dice", "DICE GPU", "Hausdorff", "Hausdorff GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2],
	        ["Dice", "Hausdorff"];
	        framevisible=false,
	    )
	end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ 764c9cfd-c0c2-44bd-8119-5d488bae4e1f
with_theme(medphys_theme) do
    loss_2D_all()
end

# ╔═╡ 5e94a962-3d88-40c6-85ba-f94f8791b67d
md"""
## 3D
"""

# ╔═╡ b3f967b7-46ae-408e-b706-e3bdf1861c50
df_loss_3D = CSV.read(current_dir * "/loss_3D.csv", DataFrame);

# ╔═╡ cf43c4ab-731b-4d78-9c69-c98fb51c6416
function loss_3D_all()
    f = Figure()
    ax1 = Axis(f[1, 1])
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (All 3D)"

	df = df_loss_3D

    sc1 = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :dice_mean_3D])
    # errorbars!(ax1, df[!, :sizes_3D], df[!, :dice_mean_3D], df[!, :dice_std_3D])

	if has_cuda_gpu()
	    sc1_gpu = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :dice_mean_gpu_3D])
	    # errorbars!(ax1, df[!, :sizes_loss_3D], df[!, :dice_mean_gpu_3D], df[!, :dice_gpu_std_3D])
	end

	sc2 = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_3D])
	# errorbars!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_3D], df[!, :hausdorff_std_3D])
	
	
	if has_cuda_gpu()
		sc2_gpu = scatter!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_gpu_3D])
		# errorbars!(ax1, df[!, :sizes_loss_3D], df[!, :hausdorff_mean_gpu_3D], df[!, :hausdorff_std_gpu_3D])
	end

	if has_cuda_gpu()
		f[1, 2] = Legend(
	        f,
	        [sc1, sc1_gpu, sc2, sc2_gpu],
	        ["Dice", "DICE GPU", "Hausdorff", "Hausdorff GPU"];
	        framevisible=false,
	    )
	else
		f[1, 2] = Legend(
	        f,
	        [sc1, sc2],
	        ["Dice", "Hausdorff"];
	        framevisible=false,
	    )
	end

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)

	# xlims!(ax1; low=3.5e4, high=4.5e4)
	# ylims!(ax1; low=0, high=4e5)
	
	return f
end

# ╔═╡ e2e2eae2-70df-4dd9-84ff-5a60455f2822
with_theme(medphys_theme) do
    loss_3D_all()
end

# ╔═╡ Cell order:
# ╠═9b9938f3-a49e-43e4-abf8-478216eefc58
# ╠═912daec8-3954-4b64-908d-75175ce2dfd2
# ╠═f1da69f4-72c4-40cc-b51e-d38e0cc8ecd1
# ╠═aafe75ac-98d1-4050-a2cf-45574c2613c5
# ╟─25a6f92a-f171-49b1-b428-95f76dd852a8
# ╠═6382c6cd-9d68-4e65-862e-79af21b86ff6
# ╠═da760574-b39f-4766-91fe-96ef9638d2fa
# ╟─a6bc73aa-573b-466c-bb8a-7bb54c376a0e
# ╟─e1c30865-7110-4e67-8338-58596206c9c6
# ╠═e4e5e57b-84be-4709-8c12-b1e5d8d8ee97
# ╟─a1e36fb1-0313-4f3c-811a-630e37324558
# ╟─e1c79280-342d-4af6-849b-1d010c884b47
# ╠═0bc66118-964b-4225-9d51-69edb40f1f29
# ╟─8c2c5919-8dd2-45e3-8345-c9f0196fd50e
# ╠═6cbde940-9b99-40c5-9ca6-0c68776e6800
# ╠═cba8582d-c9ea-4d39-a197-537985b7bd29
# ╟─69d55f25-7b57-4589-b0ad-3bdbf4a256e6
# ╠═12b298f3-bcd2-41bc-a95a-ed53f8996f36
# ╟─eedf9f9c-ba03-4453-9956-ce998c149faf
# ╠═fe306014-5740-4f53-994a-2e2facfde635
# ╠═53304723-cbbb-4eed-98a1-5538e4a00f6c
# ╟─b40d34b1-8d01-404f-9df7-1d7babbe04f2
# ╠═3df49e8b-87fe-431e-a124-37acd8bb297f
# ╟─e55b10e6-acc2-4911-ace7-cb88c57db206
# ╟─d68c8eb4-5840-48e0-9758-f96ce4432714
# ╠═5edc4c62-b895-4d4e-890c-51f1cc36024c
# ╟─22443020-3cf4-4ca3-ad18-0eb8b8017dba
# ╟─3a15a80a-5409-42e6-b61e-623029fb3d2c
# ╠═2c8e3cff-73ae-439b-a0be-b1c2a934fa16
# ╟─288bcb17-f5d3-432e-bc1b-d27734c39c57
# ╠═199a5460-3b24-4b4e-b392-d60785166595
# ╠═d51b95e0-64f6-4c38-b1b3-2537d5f5c6da
# ╠═0a5c4032-33e0-466c-a1bb-fa0b22d33cba
# ╟─a7c614f9-9f25-468f-9468-8c8de6bef1cb
# ╟─465a96d1-39fa-48ee-a782-54855ece6258
# ╠═e92af4e2-d32c-4a8f-9f82-729ba50580e6
# ╟─b505a057-32e7-4011-ba6a-366d0aee478d
# ╠═764c9cfd-c0c2-44bd-8119-5d488bae4e1f
# ╟─5e94a962-3d88-40c6-85ba-f94f8791b67d
# ╠═b3f967b7-46ae-408e-b706-e3bdf1861c50
# ╟─cf43c4ab-731b-4d78-9c69-c98fb51c6416
# ╠═e2e2eae2-70df-4dd9-84ff-5a60455f2822
