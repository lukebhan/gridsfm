#!/usr/bin/env julia
# One shard of scenarios: perturb -> solve AC-OPF (Ipopt) -> export .pyg.json.
#
# Workers are independent processes rather than Distributed workers, so a worker death
# costs one shard instead of the run, and every case is a separate file, so a run is
# resumable and killable.
#
#   julia --project=<env> worker.jl <config.json> <shard.txt> <outdir> [recdir]
using JSON3, PowerModels, Ipopt, JuMP, OrderedCollections, Polynomials, Printf, Random
PowerModels.silence()
include(joinpath(@__DIR__, "export_gridsfm.jl"))     # build_gridsfm_data: the schema authority
include(joinpath(@__DIR__, "perturb_modes.jl"))

cfgpath, shardpath, outdir = ARGS[1], ARGS[2], ARGS[3]
recdir = length(ARGS) > 3 ? ARGS[4] : joinpath(dirname(outdir), "_records")
mkpath(recdir)
# One durable record line per ATTEMPTED case, feasible or not. Survives staging deletion,
# so end-of-run statistics describe the infeasible cases too and --append can tell which
# scenario indices have already been explored.
recf = joinpath(recdir, replace(basename(shardpath), ".txt" => ".jsonl"))
cfg = JSON3.read(read(cfgpath, String), Dict{String,Any})
P   = cfg["perturbations"]; S = cfg["solver"]
grid_id = cfg["grid_id"]; topo = cfg["topology_abs"]
master  = Int(cfg["seeds"]["master"])
mkpath(outdir)

solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer,
    "print_level" => 0, "sb" => "yes",
    "max_iter" => Int(S["max_iters"]),
    "tol" => Float64(S["tol"]), "acceptable_tol" => Float64(S["acceptable_tol"]))

# Stop marker, written by the orchestrator the instant the run reaches its feasible
# target. Checked before EVERY case: batches deliberately over-provision, so without this
# the workers would solve their whole allocation and overshoot. Because the check happens
# between cases, no signal handling is needed and no partially written case is produced.
const STOPDIR = dirname(abspath(shardpath))
stopped(mode) = isfile(joinpath(STOPDIR, "_stop_$(mode)"))

"""
    set_midpoint_start!(d)

Set the solver's start point to the neutral midpoint prior: `pg = (pmin+pmax)/2`,
`qg = (qmin+qmax)/2`, and `vm` at the middle of each bus's band.

PowerModels otherwise starts `pg`/`qg` at the case file's shipped dispatch, which every
perturbed scenario has invalidated -- loads have moved and `killgen` may have tripped a
unit, so that dispatch no longer balances and Ipopt spends its first iterations walking
away from it. This is also the prior `export_gridsfm.jl` writes into the generator
features, so the solver's start and the model's input prior are the same point.
"""
function set_midpoint_start!(d)
    for (_, g) in get(d, "gen", Dict())
        Int(get(g, "gen_status", get(g, "status", 1))) == 1 || continue
        g["pg_start"] = 0.5 * (g["pmin"] + g["pmax"])
        g["qg_start"] = 0.5 * (g["qmin"] + g["qmax"])
    end
    for (_, b) in get(d, "bus", Dict())
        b["vm_start"] = 0.5 * (get(b, "vmin", 0.9) + get(b, "vmax", 1.1))
        b["va_start"] = 0.0
    end
    return d
end

n_skipped_stopped = 0
for line in readlines(shardpath)
    isempty(strip(line)) && continue
    mode, sidx_s = split(strip(line))
    sidx = parse(Int, sidx_s)
    if mode != "base" && stopped(mode)
        global n_skipped_stopped += 1
        continue
    end
    fname = mode == "base" ? "base_unperturbed.pyg.json" : @sprintf("%s_%05d.pyg.json", mode, sidx)
    outp  = joinpath(outdir, fname)
    isfile(outp) && (println("SKIP $fname"); flush(stdout); continue)
    try
        rng  = MersenneTwister(scenario_seed(master, grid_id, mode, sidx))
        data = PowerModels.parse_file(topo; import_all=false, validate=true)
        extra = mode == "base" ? Dict{String,Any}() : apply_composed!(data, rng, P)
        set_midpoint_start!(data)
        pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
        t  = @elapsed (res = PowerModels.optimize_model!(pm, optimizer=solver))
        iters = try Int(JuMP.barrier_iterations(pm.model)) catch; -1 end
        opf, feas = build_gridsfm_data(pm, res, data)
        md = opf["metadata"]
        md["scenario_id"] = sidx; md["perturb_mode"] = mode; md["feasible"] = feas
        md["grid_id"] = grid_id; md["seed_master"] = master
        md["scenario_seed"] = string(scenario_seed(master, grid_id, mode, sidx))
        md["solve_seconds"] = round(t, digits=3)
        md["solver_iters"] = iters
        for (k, v) in extra; md[k] = v; end
        rec = Dict{String,Any}("mode"=>mode, "sidx"=>sidx, "feasible"=>feas,
                   "iters"=>iters, "seconds"=>round(t, digits=3),
                   "status"=>string(get(md, "termination_status", "?")),
                   "objective"=>get(md, "objective", nothing))
        # Every perturbation descriptor goes into the record: `modes_applied` is what
        # yield is reported marginally over, since a composed scenario has no single mode
        # to attribute to, and the per-mode severity scalars are read from here too.
        for (k, v) in extra; rec[k] = v; end
        open(recf, "a") do io; println(io, JSON3.write(rec)); end
        if feas                                        # only feasible cases are published
            tmp = outp * ".tmp"                        # atomic: never leave a half file
            open(tmp, "w") do io; JSON3.pretty(io, opf); end
            mv(tmp, outp; force=true)
        end
        @printf("DONE %s feasible=%s iters=%d %.1fs\n", fname, feas, iters, t)
    catch e
        open(recf, "a") do io
            rec = Dict{String,Any}("mode"=>mode, "sidx"=>sidx, "feasible"=>false,
                                   "iters"=>-1, "seconds"=>0.0, "status"=>"ERROR",
                                   "objective"=>nothing, "modes_applied"=>String[],
                                   "n_modes_applied"=>0, "perturb_param"=>nothing)
            println(io, JSON3.write(rec))
        end
        @printf("FAIL %s %s\n", fname, first(sprint(showerror, e), 120))
    end
    flush(stdout)
end
@printf("SHARD_COMPLETE skipped_at_target=%d\n", n_skipped_stopped)
