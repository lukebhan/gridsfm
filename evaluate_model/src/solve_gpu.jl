using ExaModels, MadNLP, MadNLPGPU, CUDA, CUDSS

include("solve.jl")
include("opf_gpu.jl")

function gen_cost(data, x, nb, ng)
    pg = @view x[2nb+1 : 2nb+ng]
    sum(g.cost1*pg[g.i]^2 + g.cost2*pg[g.i] + g.cost3 for g in data.gen)
end

function start_vector(net, nb, ng, warm::Bool)
    ref = PowerModels.build_ref(net)[:it][:pm][:nw][0]
    x = zeros(Float64, 2nb + 2ng)
    for (k, (_, v)) in enumerate(ref[:bus])
        x[k] = warm ? Float64(get(v, "va", 0.0)) : 0.0
        x[nb + k] = warm ? Float64(get(v, "vm", 1.0)) : 1.0
    end
    for (k, (_, v)) in enumerate(ref[:gen])
        x[2nb + k] = warm ? Float64(get(v, "pg", 0.0)) : 0.0
        x[2nb + ng + k] = warm ? Float64(get(v, "qg", 0.0)) : 0.0
    end
    return x
end

function dist_to_start(x, x0, nb, ng)
    n = min(length(x), length(x0), 2nb + 2ng)
    d = 0.0
    @inbounds for k in 1:n
        d += (x[k] - x0[k])^2
    end
    return sqrt(d)
end

function start_gen_cost(data, x0, nb, ng)
    pg = @view x0[2nb+1 : 2nb+ng]
    sum(g.cost1*pg[g.i]^2 + g.cost2*pg[g.i] + g.cost3 for g in data.gen)
end

function setup_backend(::Val{:gpu}, cfg, read_case)
    CUDA.functional() || error("GPU evaluation requires a working CUDA device; check CUDA_VISIBLE_DEVICES and the NVIDIA driver")
    backend = CUDABackend()
    println("CUDA: ", CUDA.name(CUDA.device()))
    flush(stdout)
    settings = cfg["settings"]
    mn = settings["madnlp"]
    try
        pyg = read_case(0)
        net = build_net_from_pyg(cfg["topology"], pyg)
        flat_start!(net, settings)
        model, = build_opf(net; backend=backend, warm=false, objective=:cost,
                           pq_box=Bool(get(mn, "pq_box", true)))
        madnlp(model; tol=Float64(mn["tol"]), max_iter=50, print_level=MadNLP.ERROR,
               nlp_scaling=Bool(mn["nlp_scaling_cost"]))
        println("JIT warmup done")
        flush(stdout)
    catch e
        println("warmup skipped: ", first(sprint(showerror, e), 140))
        flush(stdout)
    end
    return backend
end

function solve_case(::Val{:gpu}, net, arm, start, objective, settings, backend)
    mn = settings["madnlp"]
    is_projection = objective == "dist_ctrl"
    obj = is_projection ? :dist_ctrl : :cost
    scaling = is_projection ? Bool(mn["nlp_scaling_dist"]) : Bool(mn["nlp_scaling_cost"])
    dscale = is_projection ? Float64(mn["proj_dscale"]) : 1.0
    vw = Float64(get(mn, "proj_vw", 1.0))
    build_seconds = @elapsed ((model, nb, ng, busdict, gendict, data) =
        build_opf(net; backend=backend, warm=(start != "flat"),
                  objective=obj, vw=vw, dscale=dscale,
                  pq_box=Bool(get(mn, "pq_box", true))))
    local result
    time_limit = Float64(get(mn, "max_wall_time", 0.0))
    seconds = @elapsed (result = time_limit > 0 ?
        madnlp(model; tol=Float64(mn["tol"]), max_iter=Int(mn["max_iter"]),
               max_wall_time=time_limit, nlp_scaling=scaling,
               print_level=MadNLP.ERROR) :
        madnlp(model; tol=Float64(mn["tol"]), max_iter=Int(mn["max_iter"]),
               nlp_scaling=scaling, print_level=MadNLP.ERROR))
    x = Array(result.solution)
    x0 = start_vector(net, nb, ng, start != "flat")
    return (; iterations=result.iter, seconds, build_seconds, status=string(result.status),
        solver_objective=result.objective, gen_cost=gen_cost(data, x, nb, ng),
        dist_start=dist_to_start(x, x0, nb, ng), start_cost=start_gen_cost(data, x0, nb, ng))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_solver(Val(:gpu))
end
