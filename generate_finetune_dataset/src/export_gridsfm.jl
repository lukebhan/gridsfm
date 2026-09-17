#!/usr/bin/env julia
# export_gridsfm_data.jl
#
# Solve a power-systems case with strict AC-OPF and export a single-scenario
# training-data file (.pyg.json) in the gridSFM schema.
#
# Input accepts anything PowerModels.parse_file handles:
#   - .m     MATPOWER (e.g. pglib cases)
#   - .json  PowerModels-native JSON (e.g. solve_topo_json.jl output)
#   - .raw / .psse / ... anything parse_file supports
#
# Typical workflow when the raw topology needs cold-strict solving first:
#   julia solve_topo_json.jl         raw.json raw.solvable.json
#   julia export_gridsfm_data.jl     raw.solvable.json out.pyg.json
#
# Or directly on a solvable source file:
#   julia export_gridsfm_data.jl     pglib_opf_case500_goc.m out.pyg.json
#
# ════════════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA (.pyg.json)
# ════════════════════════════════════════════════════════════════════════
# {
#   "grid": {
#     "nodes": {
#       "bus":       [ [base_kv, bus_type, vmin, vmax], ... ]          # 4 cols
#       "generator": [ [mbase, pg, pmin, pmax, qg, qmin, qmax, vg,     # 11 cols
#                       cp2, cp1, cp0], ... ]
#       "load":      [ [pd, qd], ... ]                                 # 2 cols
#       "shunt":     [ [bs, gs], ... ]                                 # 2 cols
#     },
#     "edges": {
#       "ac_line": {
#         "senders":   [from_bus_idx, ...]                   # 0-indexed row positions
#         "receivers": [to_bus_idx,   ...]                   #   into the bus node array
#         "features":  [ [angmin, angmax, b_fr, b_to,                  # 9 cols
#                         br_r, br_x, rate_a, rate_b, rate_c], ... ]
#                      # rate_a/b/c are NORMALIZED by _rates(): if rate_b is
#                      # missing or zero it falls back to rate_a, rate_c
#                      # falls back to rate_b. Consumers never need to
#                      # re-apply this fallback.
#       },
#       "transformer": {
#         "senders":   [from_bus_idx, ...]
#         "receivers": [to_bus_idx,   ...]
#         "features":  [ [angmin, angmax, br_r, br_x,                  # 11 cols
#                         rate_a, rate_b, rate_c, tap, shift,
#                         b_fr, b_to], ... ]
#                      # Same rate_a/b/c normalization as ac_line. `tap`
#                      # defaults to 1.0, `shift` to 0.0 when absent.
#       },
#       "generator_link": { "senders": [gen_idx…],   "receivers": [bus_idx…] },
#       "load_link":      { "senders": [load_idx…],  "receivers": [bus_idx…] },
#       "shunt_link":     { "senders": [shunt_idx…], "receivers": [bus_idx…] }
#     },
#     "context": [ [ [baseMVA] ] ]                           # scalar metadata
#   },
#   "solution": {
#     "nodes": {
#       "bus":       [ [va, vm], ... ]                                 # 2 cols
#       "generator": [ [pg, qg], ... ]                                 # 2 cols
#     },
#     "edges": {
#       "ac_line":     { "senders":[…], "receivers":[…],
#                        "features":[ [pt, qt, pf, qf], ... ] }        # 4 cols
#       "transformer": same structure
#     },
#     "duals": {
#       "bus":         [ [λ_p, λ_q, μ_vmin, μ_vmax], ... ]             # 4 cols
#                      # Bus real / reactive power-balance equality duals +
#                      # voltage-magnitude bound multipliers.
#       "generator":   [ [μ_pmin, μ_pmax, μ_qmin, μ_qmax], ... ]       # 4 cols
#       "ac_line":     [ [μ_therm_f, μ_therm_t], ... ]                 # 2 cols
#                      # Thermal-limit multipliers at from / to end.
#       "transformer": [ [μ_therm_f, μ_therm_t], ... ]                 # 2 cols
#     }
#   },
#   "metadata": {
#     "objective":              Float,           # AC-OPF objective ($)
#     "termination_status":     String,          # Ipopt/JuMP status
#     "bus_id_map":             [int, ...],      # row i → original PowerModels bus id
#     "gen_id_map":             [int, ...],      # row i → original gen id
#     "load_id_map":            [int, ...],      # row i → original load id
#     "gen_bus_map":            [int, ...],      # row i → gen's bus (orig id)
#     "load_bus_map":           [int, ...],      # row i → load's bus (orig id)
#     "ac_line_branch_ids":     [int, ...],      # row i → orig branch id
#     "transformer_branch_ids": [int, ...]       # row i → orig branch id
#   }
# }
#
# Conventions:
#   - All numeric values are per-unit on the model's baseMVA (exceptions:
#     `base_kv` is nameplate kV, `tap` is a nameplate ratio).
#   - All angles are radians.
#   - Edge `senders` / `receivers` are 0-indexed row positions into the bus
#     node array, handy for direct PyG tensor construction on the Python
#     side. Original PowerModels ids are retained in metadata id_maps for
#     round-tripping.
#   - Solution / dual fields are zeroed out when `termination_status` is NOT
#     a success (shape preserved so downstream loaders always see the same
#     tensor layout).
#
# Consumer: the Python training pipeline's build_hetero_data_from_json()
# loads this file directly as a PyG HeteroData.
# ════════════════════════════════════════════════════════════════════════
#
# Usage:
#   julia --project=<repo> export_gridsfm_data.jl <input.{m,json,...}> <output.pyg.json>
#
using PowerModels, Ipopt, JuMP
using JSON3
using Printf
using OrderedCollections
using Polynomials

PowerModels.silence()


# Is a branch actually a transformer? Off-nominal tap or non-zero phase
# shift both qualify. (PowerModels sometimes has an explicit "transformer"
# flag, fall back on tap/shift if absent.)
_is_xfmr(b) = haskey(b,"transformer") ? b["transformer"] :
              (abs(get(b,"tap",1.0)-1.0) > 1e-8 || abs(get(b,"shift",0.0)) > 1e-8)


# Normalize rate_a/b/c: missing rate_b defaults to rate_a, missing rate_c
# defaults to rate_b. Mirrors PowerModels' handling.
function _rates(b)
    ra = get(b,"rate_a",0.0)
    rb = get(b,"rate_b",ra); rb == 0 && (rb = ra)
    rc = get(b,"rate_c",rb); rc == 0 && (rc = rb)
    (ra, rb, rc)
end


# Gen cost coefficients → (cp2, cp1, cp0) in the quadratic form
#   cost = cp2 * P² + cp1 * P + cp0
#
# Accepts ONLY exact polynomial costs (PowerModels cost model 2, degree ≤ 2).
# Anything that would have to be APPROXIMATED is a hard error:
#   - model 1 (piecewise-linear, ≥6 entries)
#   - model 2 length 4 (ad-hoc linear-segment)
# Why the hard stop: `build_gridsfm_data` receives a model instantiated from
# the ORIGINAL cost, so PowerModels solved against it. Silently approximating
# here would make the exported (cp2, cp1, cp0) encode a different objective
# than the stored metadata.objective was computed from, and a round-trip
# re-solve would reconstruct a different optimization problem. To support
# non-polynomial costs, normalize the network BEFORE instantiating the model
# and export those normalized coefficients.
function _gcost(g)
    rc = try Float64.(g["cost"]) catch; Float64[] end
    isempty(rc) && return (0.0, 0.0, 0.0)   # zero-cost generator (e.g. wind/solar)

    model = Int(get(g, "model", 2))
    if model != 2
        error("unsupported generator cost model $(model) " *
              "(name=$(get(g,"name","?")), index=$(get(g,"index","?"))). " *
              "Only polynomial costs (PowerModels cost model 2) with " *
              "degree ≤ 2 can be exported exactly. Piecewise-linear costs " *
              "(model 1) would have to be approximated, which breaks " *
              "round-trip objective consistency.")
    end
    if length(rc) > 3
        error("unsupported polynomial generator cost with $(length(rc)) " *
              "coefficients (gen index=$(get(g,"index","?"))). Expected " *
              "constant, linear, or quadratic (length ≤ 3). A length-4 " *
              "\"linear segment\" cost would have to be approximated by " *
              "fitting a line, breaking round-trip objective consistency.")
    end

    p = vcat(zeros(3-length(rc)), rc)
    return (p[1], p[2], p[3])
end


"""
    build_gridsfm_data(pm, result, data) -> (opf::OrderedDict, feas::Bool)

Build the training-data OrderedDict matching the schema in the header.
`pm` is the PowerModels model (needed for dual extraction), `result` is
the optimize_model! return, `data` is the PowerModels data dict.
"""
function build_gridsfm_data(pm, result, data)
    r = result["solution"]
    ts = string(get(result, "termination_status", "UNKNOWN"))
    feas = any(s -> occursin(s, uppercase(ts)),
               ["LOCALLY_SOLVED","OPTIMAL","ALMOST_LOCALLY_SOLVED"])

    # ── Buses ──────────────────────────────────────────────────
    # ON/OFF: every node/edge row gets a trailing availability column
    # (1.0 = in service, 0.0 = out). We keep ALL elements (no active-only
    # filter) so topology is stable across outage scenarios and the model
    # sees an outaged element rather than a vanished one. Availability is
    # APPENDED last so all existing gridsfm/schema.py column indices are
    # unchanged.
    bv = sort([(parse(Int,k), b) for (k,b) in data["bus"]]; by=first)
    bf = [[b["base_kv"], b["bus_type"], b["vmin"], b["vmax"],
           Int(b["bus_type"]) == 4 ? 0.0 : 1.0] for (_,b) in bv]
    bids = [id for (id,_) in bv]
    bmap = Dict(id => i-1 for (i,id) in enumerate(bids))  # orig id → 0-indexed row

    # ── Generators (ALL rows, availability = gen_status) ──────
    # LEAK GUARD: pg/qg/vg are the OPF SOLUTION (the dispatch being PREDICTED),
    # not problem inputs, so feeding them would leak the answer. They carry a
    # neutral, solution-free MIDPOINT prior instead: pg=(pmin+pmax)/2,
    # qg=(qmin+qmax)/2, vg=1.0. That prior stays in-distribution, matches the
    # model's own 0.5*(Pmin+Pmax) base, and is the same point the solver starts
    # from. The true pg/qg/vm live only in `solution`, as the supervision
    # labels. Limits, cost, mbase and availability are genuine problem data and
    # remain as features.
    gv = sort([(parse(Int,k), g) for (k,g) in data["gen"]]; by=first)
    gf = [Float64[g["mbase"], 0.5*(g["pmin"]+g["pmax"]), g["pmin"], g["pmax"],
                  0.5*(g["qmin"]+g["qmax"]), g["qmin"], g["qmax"], 1.0, _gcost(g)...,
                  Int(get(g, "gen_status", get(g, "status", 1))) == 1 ? 1.0 : 0.0]
          for (_,g) in gv]

    # ── Loads (ALL rows, availability = status) ────────────────
    lv = sort([(parse(Int,k), l) for (k,l) in get(data, "load", Dict{String,Any}())]; by=first)
    lf = [[l["pd"], l["qd"], get(l, "status", 1) != 0 ? 1.0 : 0.0] for (_,l) in lv]

    # ── Shunts (ALL rows, availability = status) ───────────────
    sv = sort([(parse(Int,k), s) for (k,s) in get(data, "shunt", Dict{String,Any}())]; by=first)
    sf = [[s["bs"], s["gs"], get(s, "status", 1) != 0 ? 1.0 : 0.0] for (_,s) in sv]

    # ── Branches: split ac_line vs transformer (ALL rows) ─────
    brv = sort([(parse(Int,k), b) for (k,b) in get(data, "branch", Dict{String,Any}())]; by=first)
    acs, acr, acf = Int[], Int[], Vector{Float64}[]           # ac_line topology
    trs, trr, trf = Int[], Int[], Vector{Float64}[]           # transformer topology
    acid, trid    = Int[], Int[]                              # orig branch ids
    sacs, sacr, sacf = Int[], Int[], Vector{Float64}[]        # ac_line flow sol
    strs, strr, strf = Int[], Int[], Vector{Float64}[]        # xfmr flow sol
    rb = get(r, "branch", nothing)
    for (bid,b) in brv
        u, v = bmap[b["f_bus"]], bmap[b["t_bus"]]
        ra, rbb, rc = _rates(b)
        avail = get(b, "br_status", 1) != 0 ? 1.0 : 0.0   # ON/OFF (appended last)
        # Default ±π/2 RADIANS: PowerModels stores angles in radians after
        # parsing, so a degree-valued default would export ≈20626° of slack.
        amin, amax = get(b, "angmin", -π/2), get(b, "angmax", π/2)
        f4 = Float64[0,0,0,0]   # [pt, qt, pf, qf], zeros if no solution
        if rb !== nothing
            s = get(rb, string(bid), nothing)
            s !== nothing && (f4 = Float64[s["pt"], s["qt"], s["pf"], s["qf"]])
        end
        if _is_xfmr(b)
            push!(trs, u); push!(trr, v); push!(trid, bid)
            push!(trf, Float64[amin, amax, b["br_r"], b["br_x"], ra, rbb, rc,
                               get(b,"tap",1.0), get(b,"shift",0.0),
                               get(b,"b_fr",0.0), get(b,"b_to",0.0), avail])
            push!(strs, u); push!(strr, v); push!(strf, f4)
        else
            push!(acs, u); push!(acr, v); push!(acid, bid)
            push!(acf, Float64[amin, amax, get(b,"b_fr",0.0), get(b,"b_to",0.0),
                               b["br_r"], b["br_x"], ra, rbb, rc, avail])
            push!(sacs, u); push!(sacr, v); push!(sacf, f4)
        end
    end

    # ── Link edges (generator→bus, load→bus, shunt→bus) ───────
    # senders = 0-indexed row into the respective node array,
    # receivers = 0-indexed bus row the component is attached to.
    gls = collect(0:length(gv)-1); glr = [bmap[g["gen_bus"]]   for (_,g) in gv]
    lls = collect(0:length(lv)-1); llr = [bmap[l["load_bus"]]  for (_,l) in lv]
    sls = collect(0:length(sv)-1); slr = [bmap[s["shunt_bus"]] for (_,s) in sv]

    # ── Solution: bus (va, vm) + gen (pg, qg) ─────────────────
    # Bus solution is ID-KEYED over all buses (bv) so it aligns row-for-row with
    # the feature array `bf`. The solver reports only solved buses, so building
    # this positionally over r["bus"] would shift every label after the first
    # isolated (type-4) bus. Those buses have no solved va/vm and get [0,0].
    rbus = get(r, "bus", Dict{String,Any}())
    sb = [ (s = get(rbus, string(bid), nothing);
            s !== nothing ? Float64[s["va"], s["vm"]] : Float64[0.0, 0.0])
           for (bid,_) in bv ]
    sg = [[0.0, 0.0] for _ in gv]
    if haskey(r, "gen")
        gd = r["gen"]
        for (i,(gid,_)) in enumerate(gv)
            s = get(gd, string(gid), nothing)
            s !== nothing && (sg[i] = [s["pg"], s["qg"]])
        end
    end

    # ── Duals ─────────────────────────────────────────────────
    # Extracted from JuMP via the PowerModels nw(0) variable refs.
    jm = pm.model
    nwv = pm.var[:it][:pm][:nw][0]

    db, dg, da, dt = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]

    # Bus: [λ_p, λ_q, μ_vmin, μ_vmax]. JuMP orders its nonlinear equality
    # constraints so the two per-bus balance eqs come in pairs. The position
    # match assumes every bus has a balance pair, that holds only when all
    # buses are in service. If it doesn't (e.g. a type-4 isolated bus), fall
    # back to a FULL-LENGTH zero `db` so it still aligns row-for-row with the
    # bus feature array `bf` (leaving it empty misaligned the va/vm labels).
    # (Duals are not consumed by the training pipeline, correctness here is
    # about tensor-shape alignment, not dual values.)
    nle = all_constraints(jm, NonlinearExpr, MOI.EqualTo{Float64})
    nb = length(bv)
    if length(nle) >= 2nb
        vmv = nwv[:vm]
        for (bi,(bid,_)) in enumerate(bv)
            v = vmv[bid]
            push!(db, Float64[
                JuMP.dual(nle[2bi-1]),
                JuMP.dual(nle[2bi]),
                has_lower_bound(v) ? max(0.0, JuMP.dual(LowerBoundRef(v)))  : 0.0,
                has_upper_bound(v) ? max(0.0, -JuMP.dual(UpperBoundRef(v))) : 0.0,
            ])
        end
    else
        for _ in bv
            push!(db, Float64[0.0, 0.0, 0.0, 0.0])
        end
    end

    # Gen: [μ_pmin, μ_pmax, μ_qmin, μ_qmax]
    pgv, qgv = nwv[:pg], nwv[:qg]
    for (gid,_) in gv
        try
            pg, qg = pgv[gid], qgv[gid]
            push!(dg, Float64[
                has_lower_bound(pg) ? max(0.0, JuMP.dual(LowerBoundRef(pg)))  : 0.0,
                has_upper_bound(pg) ? max(0.0, -JuMP.dual(UpperBoundRef(pg))) : 0.0,
                has_lower_bound(qg) ? max(0.0, JuMP.dual(LowerBoundRef(qg)))  : 0.0,
                has_upper_bound(qg) ? max(0.0, -JuMP.dual(UpperBoundRef(qg))) : 0.0,
            ])
        catch
            push!(dg, Float64[0,0,0,0])
        end
    end

    # Branch thermal limits: [μ_therm_f, μ_therm_t] per branch.
    # Expect two quadratic ≤ constraints per branch (from-end, to-end).
    qlt = all_constraints(jm, QuadExpr, MOI.LessThan{Float64})
    if length(qlt) == 2length(brv)
        for (bi,(_,b)) in enumerate(brv)
            tf  = max(0.0, -JuMP.dual(qlt[2bi-1]))
            tr2 = max(0.0, -JuMP.dual(qlt[2bi]))
            _is_xfmr(b) ? push!(dt, Float64[tf, tr2]) : push!(da, Float64[tf, tr2])
        end
    else
        for (_,b) in brv
            _is_xfmr(b) ? push!(dt, Float64[0,0]) : push!(da, Float64[0,0])
        end
    end

    # ── Assemble OrderedDict ──────────────────────────────────
    opf = OrderedDict(
        "grid" => OrderedDict(
            "nodes" => OrderedDict("bus"=>bf, "generator"=>gf, "load"=>lf, "shunt"=>sf),
            "edges" => OrderedDict(
                "ac_line"        => OrderedDict("senders"=>acs,  "receivers"=>acr,  "features"=>acf),
                "transformer"    => OrderedDict("senders"=>trs,  "receivers"=>trr,  "features"=>trf),
                "generator_link" => OrderedDict("senders"=>gls,  "receivers"=>glr),
                "load_link"      => OrderedDict("senders"=>lls,  "receivers"=>llr),
                "shunt_link"     => OrderedDict("senders"=>sls,  "receivers"=>slr),
            ),
            "context" => [[[data["baseMVA"]]]],
        ),
        "solution" => OrderedDict(
            "nodes" => OrderedDict("bus"=>sb, "generator"=>sg),
            "edges" => OrderedDict(
                "ac_line"     => OrderedDict("senders"=>sacs, "receivers"=>sacr, "features"=>sacf),
                "transformer" => OrderedDict("senders"=>strs, "receivers"=>strr, "features"=>strf),
            ),
            "duals" => OrderedDict("bus"=>db, "generator"=>dg, "ac_line"=>da, "transformer"=>dt),
        ),
        "metadata" => OrderedDict(
            "objective"              => get(result, "objective", nothing),
            "termination_status"     => ts,
            "bus_id_map"             => bids,
            "gen_id_map"             => [id for (id,_) in gv],
            "load_id_map"            => [id for (id,_) in lv],
            "gen_bus_map"            => [g["gen_bus"]  for (_,g) in gv],
            "load_bus_map"           => [l["load_bus"] for (_,l) in lv],
            "ac_line_branch_ids"     => acid,
            "transformer_branch_ids" => trid,
        ),
    )

    # Zero solution fields on infeasible, preserves tensor shape so the
    # Python loader never needs to handle missing fields.
    if !feas
        sol = opf["solution"]
        for nt in ("bus","generator")
            haskey(sol["nodes"], nt) && (sol["nodes"][nt] = [zeros(length(v)) for v in sol["nodes"][nt]])
        end
        for et in ("ac_line","transformer")
            haskey(sol["edges"], et) && (sol["edges"][et]["features"] = [zeros(length(v)) for v in sol["edges"][et]["features"]])
        end
        for k in keys(sol["duals"])
            sol["duals"][k] = [zeros(length(v)) for v in sol["duals"][k]]
        end
    end

    return opf, feas
end


function main()
    if length(ARGS) < 2
        println("""
Usage: julia export_gridsfm_data.jl <input.{m,json,...}> <output.pyg.json>

Solves <input> with strict AC-OPF (instantiate_model + optimize_model! so
dual values are extractable), then writes a gridSFM training-data
.pyg.json to <output>. See the header docstring for the complete schema.

Input should already be cold-strict solvable. For raw topologies that
aren't, run through solve_topo_json.jl first to produce a .solvable.json
with any required relaxation applied.
""")
        exit(2)
    end
    input_path  = ARGS[1]
    output_path = ARGS[2]
    if !isfile(input_path)
        println("Input not found: $input_path"); exit(1)
    end
    mkpath(dirname(output_path))

    @info "Loading $input_path"
    net = PowerModels.parse_file(input_path; import_all=false, validate=true)

    n_bus = length(get(net, "bus", Dict()))
    max_iter = n_bus > 5000 ? 10000 : (n_bus > 1000 ? 5000 : 3000)
    solver = optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level"    => 0,
        "max_iter"       => max_iter,
        "tol"            => 1e-6,
        "acceptable_tol" => 1e-4,
    )

    t0 = time()
    pm = PowerModels.instantiate_model(net, ACPPowerModel, PowerModels.build_opf)
    result = PowerModels.optimize_model!(pm, optimizer=solver)
    elapsed = time() - t0

    term = string(get(result, "termination_status", "UNKNOWN"))
    obj  = try Float64(get(result, "objective", NaN)) catch; NaN end
    @printf("Solve: status=%s  obj=%.2f  elapsed=%.2fs\n", term, obj, elapsed)

    opf, feas = build_gridsfm_data(pm, result, net)
    open(output_path, "w") do io
        JSON3.pretty(io, opf)
    end
    @info "Wrote gridSFM pyg.json to $output_path (feas=$feas)"
    exit(feas ? 0 : 1)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
