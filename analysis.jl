### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 9b9938f3-a49e-43e4-abf8-478216eefc58
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

# ╔═╡ 25a6f92a-f171-49b1-b428-95f76dd852a8
md"""
# Distance Transforms
"""

# ╔═╡ 56ba90ce-f698-42c1-840d-b36af11e6de8
path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/"

# ╔═╡ 6382c6cd-9d68-4e65-862e-79af21b86ff6
df_dt_2D = CSV.read(string(path, "julia_timings_dt_2D.csv"), DataFrame);

# ╔═╡ a6bc73aa-573b-466c-bb8a-7bb54c376a0e
md"""
## 2D
"""

# ╔═╡ 69d55f25-7b57-4589-b0ad-3bdbf4a256e6
function dt_timings_2D()
    f = Figure()
    ax1 = Axis(f[1, 1])
	# xlims!(ax1; low=0, high=x_new[end] + 10)
	# ylims!(ax1; low=0)
	# ax1.xticks = vcat(0, 1:length(sizes))
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"

	df = df_dt_2D

	
    sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :edt_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :edt_mean_2D], df[!, :edt_std_2D])

	sc2 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_mean_2D], df[!, :sedt_std_2D])

	sc3 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_inplace_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_inplace_mean_2D], df[!, :sedt_inplace_std_2D])

	sc4 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D], df[!, :sedt_threaded_std_2D])

	sc5 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_depth])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_depth], df[!, :sedt_threaded_std_2D_depth])

	sc6 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_nonthread])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_nonthread], df[!, :sedt_threaded_std_2D_nonthread])

	sc7 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_worksteal])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D_worksteal], df[!, :sedt_threaded_std_2D_worksteal])

	f[1, 2] = Legend(
        f,
        [sc1, sc2, sc3, sc4, sc5, sc6, sc7],
        ["Euclidean", "Squared Euclidean", "Squared Euclidean In-Place", "Squared Euclidean Threaded", "Squared Euclidean DepthFirstEx", "Squared Euclidean NonThreadedEx", "Squared Euclidean WorkStealingEx"];
        framevisible=false,
    )

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)
	
	return f
end

# ╔═╡ 12b298f3-bcd2-41bc-a95a-ed53f8996f36
with_theme(medphys_theme) do
    dt_timings_2D()
end

# ╔═╡ 73d71b70-c19a-4d74-b703-237c9f1a7603
md"""
## 3D
"""

# ╔═╡ 31503e94-39de-4a1e-921e-8c5644438234
# function dt_timings_3D()
#     f = Figure()
#     ax1 = Axis(f[1, 1])
# 	# xlims!(ax1; low=0, high=x_new[end] + 10)
# 	# ylims!(ax1; low=0)
# 	# ax1.xticks = vcat(0, 1:length(sizes))
# 	ax1.xlabel = "Number of Elements"
#     ax1.ylabel = "Time (ns)"
#     ax1.title = "Distance Transforms (2D)"

	
#     sc1 = scatter!(ax1, df[!, :sizes_3D], df[!, :edt_mean_3D])
#     errorbars!(ax1, df[!, :sizes_3D], df[!, :edt_mean_3D], df[!, :edt_std_3D])

# 	sc2 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_mean_3D])
#     errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_mean_3D], df[!, :sedt_std_3D])

# 	sc3 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_inplace_mean_3D])
#     errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_inplace_mean_3D], df[!, :sedt_inplace_std_3D])

# 	sc4 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_threaded_mean_3D])
#     errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_threaded_mean_3D], df[!, :sedt_threaded_std_3D])

# 	f[1, 2] = Legend(
#         f,
#         [sc1, sc2, sc3, sc4],
#         ["Euclidean", "Squared Euclidean", "Squared Euclidean In-Place", "Squared Euclidean Threaded"];
#         framevisible=false,
#     )

# 	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)
	
# 	return f
# end

# ╔═╡ 584d05c6-c83d-45e5-973f-baf838bd16d3
# with_theme(medphys_theme) do
#     dt_timings_3D()
# end

# ╔═╡ a7c614f9-9f25-468f-9468-8c8de6bef1cb
md"""
# Loss Functions
"""

# ╔═╡ 465a96d1-39fa-48ee-a782-54855ece6258
md"""
## 2D
"""

# ╔═╡ 035f9a10-6425-442b-8760-c966e73252b9
df_loss_2D = CSV.read(string(path, "julia_timings_loss_2D.csv"), DataFrame);

# ╔═╡ d827edd3-842d-4027-81bc-3bb50e49aeee
df_loss_2D

# ╔═╡ 853844e9-cd7e-4130-95b9-01d3596c907c
function loss_timings_2D()
    f = Figure()
    ax1 = Axis(f[1, 1])
	# xlims!(ax1; low=0, high=x_new[end] + 10)
	# ylims!(ax1; low=0)
	# ax1.xticks = vcat(0, 1:length(sizes))
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"

	df = df_loss_2D

	
    sc1 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :dice_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :edt_mean_2D], df[!, :edt_std_2D])

	sc2 = scatter!(ax1, df[!, :sizes_loss_2D], df[!, :hausdorff_mean_2D])
    # errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_mean_2D], df[!, :sedt_std_2D])

	f[1, 2] = Legend(
        f,
        [sc1, sc2],
        ["DICE", "Hausdorff"];
        framevisible=false,
    )

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)
	
	return f
end

# ╔═╡ 17d02b26-ccbd-430a-8519-2465c6dcc518
with_theme(medphys_theme) do
    loss_timings_2D()
end

# ╔═╡ Cell order:
# ╠═9b9938f3-a49e-43e4-abf8-478216eefc58
# ╠═912daec8-3954-4b64-908d-75175ce2dfd2
# ╠═f1da69f4-72c4-40cc-b51e-d38e0cc8ecd1
# ╟─25a6f92a-f171-49b1-b428-95f76dd852a8
# ╠═56ba90ce-f698-42c1-840d-b36af11e6de8
# ╠═6382c6cd-9d68-4e65-862e-79af21b86ff6
# ╟─a6bc73aa-573b-466c-bb8a-7bb54c376a0e
# ╠═69d55f25-7b57-4589-b0ad-3bdbf4a256e6
# ╟─12b298f3-bcd2-41bc-a95a-ed53f8996f36
# ╟─73d71b70-c19a-4d74-b703-237c9f1a7603
# ╠═31503e94-39de-4a1e-921e-8c5644438234
# ╟─584d05c6-c83d-45e5-973f-baf838bd16d3
# ╟─a7c614f9-9f25-468f-9468-8c8de6bef1cb
# ╟─465a96d1-39fa-48ee-a782-54855ece6258
# ╠═035f9a10-6425-442b-8760-c966e73252b9
# ╠═d827edd3-842d-4027-81bc-3bb50e49aeee
# ╟─853844e9-cd7e-4130-95b9-01d3596c907c
# ╠═17d02b26-ccbd-430a-8519-2465c6dcc518
