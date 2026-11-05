function build_net_from_pyg(solvable_path::AbstractString, pyg::AbstractDict)
    net = PowerModels.parse_file(solvable_path; import_all=false, validate=true)
    grid = pyg["grid"]; meta = pyg["metadata"]

    _sort_by_index(d::Dict{String,<:Any}) = sort(collect(values(d)); by = x -> Int(x["index"]))

    _gen_active(g) = Int(get(g, "gen_status", get(g, "status", 1))) == 1
    loads_pm  = _sort_by_index(get(net, "load", Dict{String,Any}()))
    buses_pm  = _sort_by_index(get(net, "bus", Dict{String,Any}()))

    loads_json = grid["nodes"]["load"]::Vector
    if length(loads_json) != length(loads_pm)
        error("load count mismatch: pyg=$(length(loads_json)) solvable=$(length(loads_pm))")
    end
    for (ld_pm, row) in zip(loads_pm, loads_json)
        ld_pm["pd"] = Float64(row[1])
        ld_pm["qd"] = Float64(row[2])
    end

    if haskey(pyg, "solution") && haskey(pyg["solution"], "nodes") &&
       haskey(pyg["solution"]["nodes"], "bus")
        bus_sol = pyg["solution"]["nodes"]["bus"]::Vector
        if !isempty(bus_sol)
            if length(bus_sol) != length(buses_pm)
                error("bus warm-start row count mismatch: pyg.solution=$(length(bus_sol)) " *
                      "solvable=$(length(buses_pm)). Did the exporter schema change?")
            end
            for (b_pm, s) in zip(buses_pm, bus_sol)
                b_pm["va"] = Float64(s[1]); b_pm["va_start"] = Float64(s[1])
                b_pm["vm"] = Float64(s[2]); b_pm["vm_start"] = Float64(s[2])
            end
        end
    end

    bus_ids = [Int(x) for x in meta["bus_id_map"]]
    n_bus_pyg = length(grid["nodes"]["bus"])
    if length(bus_ids) != n_bus_pyg
        error("bus_id_map length $(length(bus_ids)) != grid.nodes.bus rows $n_bus_pyg")
    end
    for (i, row) in enumerate(grid["nodes"]["bus"])
        bid = bus_ids[i]
        b   = get(net["bus"], string(bid), nothing)
        b === nothing && continue
        b["vmin"] = Float64(row[3])
        b["vmax"] = Float64(row[4])
    end

    GEN_AVAIL_COL = 12
    gen_nodes = grid["nodes"]["generator"]
    gen_ids   = [Int(x) for x in meta["gen_id_map"]]
    live_ids  = Set(gen_ids)

    for (_, g) in net["gen"]
        if !(Int(g["index"]) in live_ids) && _gen_active(g)
            g["gen_status"] = 0
        end
    end

    sol_gen = haskey(pyg, "solution") && haskey(pyg["solution"], "nodes") ?
              get(pyg["solution"]["nodes"], "generator", nothing) : nothing
    for (i, row) in enumerate(gen_nodes)
        gid = gen_ids[i]
        g   = get(net["gen"], string(gid), nothing)
        g === nothing && continue

        if length(row) >= GEN_AVAIL_COL
            g["gen_status"] = Float64(row[GEN_AVAIL_COL]) > 0.5 ? 1 : 0
        end

        cp2, cp1, cp0 = Float64(row[9]), Float64(row[10]), Float64(row[11])
        g["cost"]  = Float64[cp2, cp1, cp0]
        g["model"] = 2
        g["ncost"] = 3

        g["pmin"] = Float64(row[3]); g["pmax"] = Float64(row[4])
        g["qmin"] = Float64(row[6]); g["qmax"] = Float64(row[7])

        if sol_gen !== nothing && i <= length(sol_gen) && !isempty(sol_gen[i])
            g["pg"] = Float64(sol_gen[i][1]); g["pg_start"] = Float64(sol_gen[i][1])
            g["qg"] = Float64(sol_gen[i][2]); g["qg_start"] = Float64(sol_gen[i][2])
        end
    end

    set_rate!(b, key, v) = (v > 0 ? (b[key] = Float64(v)) : delete!(b, key))

    ac_ids = [Int(x) for x in meta["ac_line_branch_ids"]]
    for (i, feats) in enumerate(grid["edges"]["ac_line"]["features"])
        bid = ac_ids[i]
        b   = get(net["branch"], string(bid), nothing)
        b === nothing && continue
        set_rate!(b, "rate_a", feats[7])
        set_rate!(b, "rate_b", feats[8])
        set_rate!(b, "rate_c", feats[9])
    end
    tr_ids = [Int(x) for x in meta["transformer_branch_ids"]]
    for (i, feats) in enumerate(grid["edges"]["transformer"]["features"])
        bid = tr_ids[i]
        b   = get(net["branch"], string(bid), nothing)
        b === nothing && continue
        set_rate!(b, "rate_a", feats[5])
        set_rate!(b, "rate_b", feats[6])
        set_rate!(b, "rate_c", feats[7])
    end

    return net
end
