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
	using CairoMakie
	using BenchmarkTools
	using DataFrames
	using CSV
	using CUDA
	using DistanceTransforms
	using Losers
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
# path = "/Users/daleblack/Google Drive/dev/MolloiLab/distance-transforms/julia_timings.csv"
# path = raw"C:\Users\wenbl13\Desktop\dale\distance-transforms\julia_timings.csv"
path = raw"C:\Users\wenbl13\Desktop\dale\distance-transforms\julia_timings_gpu.csv"

# ╔═╡ 6382c6cd-9d68-4e65-862e-79af21b86ff6
df = CSV.read(string(path), DataFrame);

# ╔═╡ b5a90a40-eb61-49c2-8115-78bf1d67f9bb
begin
	idxs_2D = zeros(size(names(df)))
	idxs_3D = zeros(size(names(df)))
	for i in 1:length(names(df))
		if occursin("2D", names(df)[i])
			idxs_2D[i] = 1
		elseif occursin("3D", names(df)[i])
			idxs_3D[i] = 1
		end
	end
	idxs_2D = Bool.(idxs_2D)
	idxs_3D = Bool.(idxs_3D)
end;

# ╔═╡ de5b393a-f4d3-4698-aa6f-83f042f8c9cb
df_2D = df[!, idxs_2D];

# ╔═╡ e866db41-1c5a-481c-be47-0f78900a93ce
df_3D = df[!, idxs_3D];

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

	
    sc1 = scatter!(ax1, df[!, :sizes_2D], df[!, :edt_mean_2D])
    errorbars!(ax1, df[!, :sizes_2D], df[!, :edt_mean_2D], df[!, :edt_std_2D])

	sc2 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_mean_2D])
    errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_mean_2D], df[!, :sedt_std_2D])

	sc3 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_inplace_mean_2D])
    errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_inplace_mean_2D], df[!, :sedt_inplace_std_2D])

	sc4 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D])
    errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_threaded_mean_2D], df[!, :sedt_threaded_std_2D])

	sc5 = scatter!(ax1, df[!, :sizes_2D], df[!, :sedt_gpu_mean_2D])
    errorbars!(ax1, df[!, :sizes_2D], df[!, :sedt_gpu_mean_2D], df[!, :sedt_gpu_std_2D])

	f[1, 2] = Legend(
        f,
        [sc1, sc2, sc3, sc4, sc5],
        ["Euclidean", "Squared Euclidean", "Squared Euclidean In-Place", "Squared Euclidean Threaded", "Squared Euclidean GPU"];
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
function dt_timings_3D()
    f = Figure()
    ax1 = Axis(f[1, 1])
	# xlims!(ax1; low=0, high=x_new[end] + 10)
	# ylims!(ax1; low=0)
	# ax1.xticks = vcat(0, 1:length(sizes))
	ax1.xlabel = "Number of Elements"
    ax1.ylabel = "Time (ns)"
    ax1.title = "Distance Transforms (2D)"

	
    sc1 = scatter!(ax1, df[!, :sizes_3D], df[!, :edt_mean_3D])
    errorbars!(ax1, df[!, :sizes_3D], df[!, :edt_mean_3D], df[!, :edt_std_3D])

	sc2 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_mean_3D])
    errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_mean_3D], df[!, :sedt_std_3D])

	sc3 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_inplace_mean_3D])
    errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_inplace_mean_3D], df[!, :sedt_inplace_std_3D])

	sc4 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_threaded_mean_3D])
    errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_threaded_mean_3D], df[!, :sedt_threaded_std_3D])

	sc5 = scatter!(ax1, df[!, :sizes_3D], df[!, :sedt_gpu_mean_3D])
    errorbars!(ax1, df[!, :sizes_3D], df[!, :sedt_gpu_mean_3D], df[!, :sedt_gpu_std_3D])

	f[1, 2] = Legend(
        f,
        [sc1, sc2, sc3, sc4, sc5],
        ["Euclidean", "Squared Euclidean", "Squared Euclidean In-Place", "Squared Euclidean Threaded", "Squared Euclidean GPU"];
        framevisible=false,
    )

	 # save("/Users/daleblack/Google Drive/Research/Papers/My Papers/cac-simulation/figures/linear_reg_norm.png", f)
	
	return f
end

# ╔═╡ 584d05c6-c83d-45e5-973f-baf838bd16d3
with_theme(medphys_theme) do
    dt_timings_3D()
end

# ╔═╡ Cell order:
# ╠═9b9938f3-a49e-43e4-abf8-478216eefc58
# ╠═912daec8-3954-4b64-908d-75175ce2dfd2
# ╟─f1da69f4-72c4-40cc-b51e-d38e0cc8ecd1
# ╟─25a6f92a-f171-49b1-b428-95f76dd852a8
# ╠═56ba90ce-f698-42c1-840d-b36af11e6de8
# ╠═6382c6cd-9d68-4e65-862e-79af21b86ff6
# ╠═b5a90a40-eb61-49c2-8115-78bf1d67f9bb
# ╠═de5b393a-f4d3-4698-aa6f-83f042f8c9cb
# ╠═e866db41-1c5a-481c-be47-0f78900a93ce
# ╟─a6bc73aa-573b-466c-bb8a-7bb54c376a0e
# ╟─69d55f25-7b57-4589-b0ad-3bdbf4a256e6
# ╟─12b298f3-bcd2-41bc-a95a-ed53f8996f36
# ╟─73d71b70-c19a-4d74-b703-237c9f1a7603
# ╟─31503e94-39de-4a1e-921e-8c5644438234
# ╟─584d05c6-c83d-45e5-973f-baf838bd16d3
