using PowerModels, Ipopt, JSON3, JuMP

include("recon.jl")

function solve_gridsfm_ac_opf(data, solver; prepare! = nothing)
    t0 = time_ns()
    pm = PowerModels.instantiate_model(data, ACPPowerModel, PowerModels.build_opf)
    local result
    seconds = @elapsed begin
        if prepare! === nothing
            result = PowerModels.optimize_model!(pm, optimizer=solver)
        else
            JuMP.set_optimizer(pm.model, solver)
            prepare!(pm)
            result = PowerModels.optimize_model!(pm)
        end
    end
    total_seconds = (time_ns() - t0) / 1e9
    iterations = try Int(JuMP.barrier_iterations(pm.model)) catch; -1 end
    return (; model=pm, result, iterations, seconds, total_seconds)
end

function ipopt_attributes(options, integer_options)
    attrs = Any[]
    for (key, value) in options
        name = String(key)
        typed = name in integer_options ? Int(value) : value isa Number ? Float64(value) : value
        push!(attrs, name => typed)
    end
    return attrs
end

function optimizer(arm, settings)
    options = settings["ipopt"]
    attrs = ipopt_attributes(options["common"], options["integer_options"])
    mu = options["mu_init"][arm == "cold" ? "cold" : "warm"]
    mu === nothing || push!(attrs, "mu_init" => Float64(mu))
    return PowerModels.optimizer_with_attributes(Ipopt.Optimizer, attrs...)
end

function flat_start!(net, settings)
    start = settings["starts"]["cold"]
    for b in values(net["bus"])
        b["vm"] = b["vm_start"] = Float64(start["vm"])
        b["va"] = b["va_start"] = Float64(start["va"])
    end
    for g in values(net["gen"])
        g["pg"] = g["pg_start"] = Float64(start["pg"] isa AbstractString ? g[start["pg"]] : start["pg"])
        g["qg"] = g["qg_start"] = Float64(start["qg"] isa AbstractString ? g[start["qg"]] : start["qg"])
    end
end

function dc_start!(net, settings)
    start = settings["starts"]["dc"]
    solver = PowerModels.optimizer_with_attributes(Ipopt.Optimizer,
        ipopt_attributes(settings["ipopt"]["dc"], settings["ipopt"]["integer_options"])...)
    local result
    seconds = @elapsed result = PowerModels.solve_opf(net, DCPPowerModel, solver)
    status = string(result["termination_status"])
    if !(status in settings["evaluation"]["solved_statuses"])
        flat_start!(net, settings)
        return seconds, "dc_FAILED:$status"
    end
    sol = result["solution"]
    for (i, b) in net["bus"]
        s = get(get(sol, "bus", Dict()), i, Dict())
        b["va"] = b["va_start"] = Float64(get(s, "va", start["va"]))
        b["vm"] = b["vm_start"] = Float64(start["vm"])
    end
    for (i, g) in net["gen"]
        s = get(get(sol, "gen", Dict()), i, Dict())
        g["pg"] = g["pg_start"] = Float64(get(s, "pg", start["pg"]))
        g["qg"] = g["qg_start"] = Float64(start["qg"])
    end
    return seconds, "dc"
end

function model_start!(net, pred, settings)
    start = settings["starts"]["model"]
    buses = sort(collect(values(net["bus"])); by=b -> Int(b["index"]))
    gens = sort(collect(values(net["gen"])); by=g -> Int(g["index"]))
    length(pred["vm"]) == length(buses) || error("Prediction bus count differs from scenario")
    length(pred["pg"]) == length(gens) || error("Prediction generator count differs from scenario")
    for (k, b) in enumerate(buses)
        b["vm"] = b["vm_start"] = Float64(pred["vm"][k])
        b["va"] = b["va_start"] = haskey(pred, "va") ? Float64(pred["va"][k]) : Float64(start["va"])
    end
    for (k, g) in enumerate(gens)
        g["pg"] = g["pg_start"] = Float64(pred["pg"][k])
        g["qg"] = g["qg_start"] = haskey(pred, "qg") ? Float64(pred["qg"][k]) : Float64(start["qg"])
    end
end

function projection!(pm, net, settings)
    projection = settings["projection"]
    pg, vm = PowerModels.var(pm, 0, :pg), PowerModels.var(pm, 0, :vm)
    gids, bids = PowerModels.ids(pm, 0, :gen), PowerModels.ids(pm, 0, :bus)
    pgt = Dict(g => Float64(net["gen"][string(g)]["pg"]) for g in gids)
    ctrl = [b for b in bids if Int(net["bus"][string(b)]["bus_type"]) in projection["control_bus_types"]]
    vmt = Dict(b => Float64(net["bus"][string(b)]["vm"]) for b in ctrl)
    pg_weight, vm_weight = Float64(projection["pg_weight"]), Float64(projection["vm_weight"])
    JuMP.@objective(pm.model, Min, pg_weight * sum((pg[g] - pgt[g])^2 for g in gids) +
        vm_weight * sum((vm[b] - vmt[b])^2 for b in ctrl))
end

function gen_cost(net, sol; start=false)
    cost = 0.0
    for (i, g) in net["gen"]
        Int(get(g, "gen_status", 1)) == 1 || continue
        p = start ? Float64(get(g, "pg_start", 0.0)) :
            Float64(get(get(get(sol, "gen", Dict()), i, Dict()), "pg", 0.0))
        c = get(g, "cost", Float64[])
        cost += length(c) >= 3 ? c[1]*p^2 + c[2]*p + c[3] :
            length(c) == 2 ? c[1]*p + c[2] : 0.0
    end
    return cost
end

function dist_to_start(net, sol)
    d = 0.0
    for (i, b) in net["bus"]
        s = get(get(sol, "bus", Dict()), i, Dict())
        d += (Float64(get(s, "va", 0.0)) - b["va_start"])^2
        d += (Float64(get(s, "vm", 1.0)) - b["vm_start"])^2
    end
    for (i, g) in net["gen"]
        Int(get(g, "gen_status", 1)) == 1 || continue
        s = get(get(sol, "gen", Dict()), i, Dict())
        d += (Float64(get(s, "pg", 0.0)) - g["pg_start"])^2
        d += (Float64(get(s, "qg", 0.0)) - g["qg_start"])^2
    end
    return sqrt(d)
end

csvcell(x) = '"' * replace(string(x), '"' => "\"\"") * '"'
write_row(io, xs) = (println(io, join(csvcell.(xs), ',')); flush(io))

function evaluation_main(args=ARGS)
    length(args) == 3 || error("Usage: solve.jl CONFIG.json TASKS.jsonl OUT.csv")
    cfgpath, taskpath, outpath = args
    cfg = JSON3.read(read(cfgpath, String), Dict{String,Any})
    settings = cfg["settings"]
    PowerModels.silence()
    pred_cache = Dict{String,Dict{Int,Any}}()
    errors = 0
    open(outpath, "w") do io
        write_row(io, ["arm", "run", "case_idx", "mode", "scenario_id", "objective_kind",
            "iterations", "seconds", "start_seconds", "status", "solver_objective",
            "gen_cost", "gt_objective", "dist_start", "start_cost", "note"])
        for line in eachline(taskpath)
            isempty(strip(line)) && continue
            t = JSON3.read(line, Dict{String,Any})
            arm, i = String(t["arm"]), Int(t["case_idx"])
            run = get(t, "run", nothing)
            obj, start = String(t["objective"]), String(t["start"])
            println("START arm=$arm case=$i objective=$obj"); flush(stdout)
            try
                offset = Int(cfg["offsets"][i + 1])
                offset >= 0 || error("Missing byte offset for scenario $i")
                pyg = open(cfg["split_jsonl"]) do f
                    seek(f, offset)
                    JSON3.read(readline(f), Dict{String,Any})
                end
                md = pyg["metadata"]
                net = build_net_from_pyg(cfg["topology"], pyg)
                start_seconds, note = 0.0, start
                if start == "flat"
                    flat_start!(net, settings)
                elseif start == "dc"
                    start_seconds, note = dc_start!(net, settings)
                elseif start in ("model", "base", "ft", "lf")
                    run === nothing && error("Model start requires a prediction run")
                    preds = get!(pred_cache, String(run)) do
                        Dict(Int(p["i"]) => p for p in
                            (JSON3.read(l, Dict{String,Any}) for l in eachline(joinpath(cfg["pred_dir"], run * ".jsonl"))))
                    end
                    pred = preds[i]
                    model_start!(net, pred, settings)
                    note = get(pred, "dc_fallback", false) ? "model_dcfallback" : "model"
                else
                    error("Unknown start: $start")
                end
                obj in ("cost", "dist_ctrl") || error("Unknown objective: $obj")
                solved = solve_gridsfm_ac_opf(net, optimizer(arm, settings); prepare! = pm -> begin
                    obj == "dist_ctrl" && projection!(pm, net, settings)
                    deadline = time() + Float64(settings["ipopt"]["callback_time_limit"])
                    JuMP.MOI.set(pm.model, Ipopt.CallbackFunction(), (args...) -> time() < deadline)
                end)
                pm, result, seconds = solved.model, solved.result, solved.total_seconds
                status = string(result["termination_status"])
                iterations = solved.iterations
                sol = get(result, "solution", Dict())
                write_row(io, [arm, something(run, ""), i, get(md, "perturb_mode", ""),
                    get(md, "scenario_id", ""), obj, iterations, seconds, start_seconds, status,
                    get(result, "objective", NaN), gen_cost(net, sol), get(md, "objective", NaN),
                    dist_to_start(net, sol), gen_cost(net, Dict(); start=true), note])
                println("DONE arm=$arm case=$i status=$status iterations=$iterations"); flush(stdout)
            catch e
                errors += 1
                msg = replace(sprint(showerror, e), '\n' => ' ')
                write_row(io, [arm, something(run, ""), i, "", "", obj, "", "", "", "ERROR",
                    "", "", "", "", "", msg])
                showerror(stderr, e, catch_backtrace()); println(stderr)
            end
        end
    end
    errors == 0 || error("$errors task(s) failed; see ERROR rows in $outpath")
end

if abspath(PROGRAM_FILE) == @__FILE__
    evaluation_main()
end
