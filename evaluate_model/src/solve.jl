using PowerModels, Ipopt, JSON3, JuMP

include("recon.jl")

function ipopt_attributes(options, integer_options)
    attrs = Any[]
    for (key, value) in options
        name = String(key)
        typed = name in integer_options ? Int(value) : value isa Number ? Float64(value) : value
        push!(attrs, name => typed)
    end
    return attrs
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

extra_columns(::Val{:cpu}) = String[]
extra_columns(::Val{:gpu}) = ["build_seconds"]

function run_solver(backend, args=ARGS)
    length(args) == 3 || error("Usage: solver.jl CONFIG.json TASKS.jsonl OUT.csv")
    cfgpath, taskpath, outpath = args
    cfg = JSON3.read(read(cfgpath, String), Dict{String,Any})
    settings = cfg["settings"]
    PowerModels.silence()
    offsets = Vector{Int}(cfg["offsets"])
    pred_cache = Dict{String,Dict{Int,Any}}()
    read_case(i::Int) = open(cfg["split_jsonl"]) do stream
        offset = offsets[i + 1]
        offset >= 0 || error("Missing byte offset for scenario $i")
        seek(stream, offset)
        JSON3.read(readline(stream), Dict{String,Any})
    end
    predictions(run::AbstractString) = get!(pred_cache, String(run)) do
        records = Dict{Int,Any}()
        for line in eachline(joinpath(cfg["pred_dir"], run * ".jsonl"))
            isempty(strip(line)) && continue
            pred = JSON3.read(line, Dict{String,Any})
            records[Int(pred["i"])] = pred
        end
        records
    end
    state = setup_backend(backend, cfg, read_case)
    columns = ["arm", "run", "case_idx", "mode", "scenario_id", "objective_kind",
        "iterations", "seconds", extra_columns(backend)..., "start_seconds", "status",
        "solver_objective", "gen_cost", "gt_objective", "dist_start", "start_cost", "note"]
    errors = 0
    open(outpath, "w") do io
        write_row(io, columns)
        for line in eachline(taskpath)
            isempty(strip(line)) && continue
            task = JSON3.read(line, Dict{String,Any})
            arm, i = String(task["arm"]), Int(task["case_idx"])
            run = get(task, "run", nothing)
            objective, start = String(task["objective"]), String(task["start"])
            row = Dict{String,Any}("arm" => arm, "run" => something(run, ""),
                "case_idx" => i, "objective_kind" => objective)
            println("START arm=$arm case=$i objective=$objective")
            flush(stdout)
            try
                pyg = read_case(i)
                md = get(pyg, "metadata", Dict{String,Any}())
                net = build_net_from_pyg(cfg["topology"], pyg)
                start_seconds, note = 0.0, start
                if start == "flat"
                    flat_start!(net, settings)
                elseif start == "dc"
                    start_seconds, note = dc_start!(net, settings)
                elseif start in ("model", "base", "ft", "lf")
                    run === nothing && error("Model start requires a prediction run")
                    pred = predictions(String(run))[i]
                    model_start!(net, pred, settings)
                    note = get(pred, "dc_fallback", false) ? "model_dcfallback" : "model"
                else
                    error("Unknown start: $start")
                end
                objective in ("cost", "dist_ctrl") || error("Unknown objective: $objective")
                result = solve_case(backend, net, arm, start, objective, settings, state)
                for (key, value) in pairs(result)
                    row[String(key)] = value
                end
                row["mode"] = get(md, "perturb_mode", "")
                row["scenario_id"] = get(md, "scenario_id", "")
                row["gt_objective"] = get(md, "objective", "")
                row["start_seconds"] = start_seconds
                row["note"] = note
                write_row(io, [get(row, name, "") for name in columns])
                println("DONE arm=$arm case=$i status=$(result.status) iterations=$(result.iterations)")
                flush(stdout)
            catch e
                errors += 1
                row["status"] = "ERROR"
                row["note"] = replace(sprint(showerror, e), '\n' => ' ')
                write_row(io, [get(row, name, "") for name in columns])
                showerror(stderr, e, catch_backtrace())
                println(stderr)
            end
        end
    end
    errors == 0 || error("$errors task(s) failed; see ERROR rows in $outpath")
end
