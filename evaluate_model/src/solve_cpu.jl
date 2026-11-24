include("solve.jl")

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

function optimizer(arm, settings)
    options = settings["ipopt"]
    attrs = ipopt_attributes(options["common"], options["integer_options"])
    mu = options["mu_init"][arm == "cold" ? "cold" : "warm"]
    mu === nothing || push!(attrs, "mu_init" => Float64(mu))
    return PowerModels.optimizer_with_attributes(Ipopt.Optimizer, attrs...)
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

setup_backend(::Val{:cpu}, cfg, read_case) = nothing

function solve_case(::Val{:cpu}, net, arm, start, objective, settings, state)
    solved = solve_gridsfm_ac_opf(net, optimizer(arm, settings); prepare! = pm -> begin
        objective == "dist_ctrl" && projection!(pm, net, settings)
        deadline = time() + Float64(settings["ipopt"]["callback_time_limit"])
        JuMP.MOI.set(pm.model, Ipopt.CallbackFunction(), (args...) -> time() < deadline)
    end)
    pm, result = solved.model, solved.result
    sol = get(result, "solution", Dict())
    return (; iterations=solved.iterations, seconds=solved.total_seconds,
        status=string(result["termination_status"]), solver_objective=get(result, "objective", NaN),
        gen_cost=gen_cost(net, sol), dist_start=dist_to_start(net, sol),
        start_cost=gen_cost(net, Dict(); start=true))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_solver(Val(:cpu))
end
