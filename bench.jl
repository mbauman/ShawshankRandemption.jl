import Random, ShawshankRandemption
using StatsPlots: StatsPlots, plot, plot!, savefig, @layout, ylims, ylims!
using Statistics: median, quantile
using BenchmarkTools: BenchmarkTools, BenchmarkGroup, @benchmark, @benchmarkable
import BenchmarkPlots

## Scalars
scalars = BenchmarkGroup()
scalars["Float16"] = BenchmarkGroup()
scalars["Float16"]["Random"] = @benchmarkable Random.rand(Float16)
scalars["Float16"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand(Float16)
scalars["Float32"] = BenchmarkGroup()
scalars["Float32"]["Random"] = @benchmarkable Random.rand(Float32)
scalars["Float32"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand(Float32)
scalars["Float64"] = BenchmarkGroup()
scalars["Float64"]["Random"] = @benchmarkable Random.rand(Float64)
scalars["Float64"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand(Float64)

BenchmarkTools.tune!(scalars)
ts = BenchmarkTools.run(scalars)
p = plot([plot(ts["$T"], title="rand($T)") for T in (Float16, Float32, Float64)]..., layout=@layout([a b c]), titlefontsize=8, tickfontsize=4, labelfontsize=4)
ylims!.(p.subplots, tuple.(0, 5 #=last.(ylims.(p.subplots)))=#))
savefig(p, "scalar.svg")

## Arrays
arrays = BenchmarkGroup()
sizes = round.(Int, exp2.(2:1.75:25))
arrays["Float16"] = BenchmarkGroup()
for n in sizes
    arrays["Float16"]["$n"] = BenchmarkGroup()
    arrays["Float16"]["$n"]["Random"] = @benchmarkable Random.rand!(A) setup=A=Array{Float16}(undef, $n)
    arrays["Float16"]["$n"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand!(A) setup=A=Array{Float16}(undef, $n)
end
arrays["Float32"] = BenchmarkGroup()
for n in sizes
    arrays["Float32"]["$n"] = BenchmarkGroup()
    arrays["Float32"]["$n"]["Random"] = @benchmarkable Random.rand!(A) setup=A=Array{Float32}(undef, $n)
    arrays["Float32"]["$n"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand!(A) setup=A=Array{Float32}(undef, $n)
end
arrays["Float64"] = BenchmarkGroup()
for n in sizes
    arrays["Float64"]["$n"] = BenchmarkGroup()
    arrays["Float64"]["$n"]["Random"] = @benchmarkable Random.rand!(A) setup=A=Array{Float64}(undef, $n)
    arrays["Float64"]["$n"]["ShawshankRandemption"] = @benchmarkable ShawshankRandemption.rand!(A) setup=A=Array{Float64}(undef, $n)
end

BenchmarkTools.tune!(arrays)
ta = BenchmarkTools.run(arrays)

# p = plot([plot(t["$T"]["$n"], title="$n x $T") for n in sizes, T in (Float16, Float32, Float64)]..., titlefontsize=8, tickfontsize=4, labelfontsize=4)
# ylims!.(p.subplots, tuple.(0, last.(ylims.(p.subplots))))

p = plot(xaxis=:log, xlabel="Array length", ylabel="Average ns/elt (min - median - 90%)")
colors = StatsPlots.palette(:Paired_6)
for (i,T) in enumerate((Float16, Float32, Float64))
    r =  [f(ta["$T"]["$n"]["Random"]).time/n for n in sizes, f in (minimum, median, x->(;time=quantile(x.times, .9)))]
    plot!(p, sizes, r[:,2], ribbon=(r[:,2].-r[:,1], r[:,3].-r[:,2]), label="Random $T", color=colors[i*2])
    s =  [f(ta["$T"]["$n"]["ShawshankRandemption"]).time/n for n in sizes, f in (minimum, median, x->(;time=quantile(x.times, .9)))]
    plot!(p, sizes, s[:,2], ribbon=(s[:,2].-s[:,1], s[:,3].-s[:,2]), label="ShawshankRandemption $T", color=colors[i*2-1], linestyle=:dash)
end
ylims!(p, (0, last(ylims(p))))

savefig(p, "array.svg")
