# Scenario perturbations.
#
# A scenario applies several modes to the same grid, each gated by its own probability
# in `perturbations.mode_probs`. Composition is the point: the model sees a grid that is
# simultaneously more loaded, re-dispatched and short a unit, which is what an operator
# sees. Each mode function is independent and also usable alone.
#
# SEEDING (the reproducibility contract):
#   Every scenario's RNG is seeded from SHA-256 of "master|grid_id|mode|sidx", where
#   `mode` is "multi" for a composed scenario and "base" for the unperturbed one. The
#   seed depends on the grid's NAME, never on its file path, so the same config
#   reproduces the same dataset on any machine, Julia version or checkout location.
using SHA, Random

function scenario_seed(master::Integer, grid_id::AbstractString,
                       mode::AbstractString, sidx::Integer)::UInt64
    h = SHA.sha256("$(master)|$(grid_id)|$(mode)|$(sidx)")
    s = UInt64(0)
    for i in 1:8
        s = (s << 8) | UInt64(h[i])
    end
    return s
end

"""
    mode_loads!(d, rng, p)

Demand variation. Draws one system-wide factor `sf ~ U[load_sf_lo, load_sf_hi]`, then
scales every load's `pd`/`qd` by `sf * U[1-jitter, 1+jitter]`: a correlated system move
plus independent per-bus noise.
"""
function mode_loads!(d, rng, p)
    lo, hi, j = p["load_sf_lo"], p["load_sf_hi"], p["load_jitter"]
    sf = lo + rand(rng) * (hi - lo)
    for (_, ld) in get(d, "load", Dict())
        ld["pd"] *= sf * ((1 - j) + rand(rng) * 2j)
        ld["qd"] *= sf * ((1 - j) + rand(rng) * 2j)
    end
    Dict("system_load_factor" => round(sf, digits=4))
end

"""
    mode_costs!(d, rng, p)

Merit-order reshuffle. Takes `cost_frac` of the online generators, buckets them by cost
degree (`ncost`), and permutes the cost vectors within each bucket. Degree is preserved,
so a quadratic curve never becomes linear; what changes is which units are cheap.
"""
function mode_costs!(d, rng, p)
    act = [(k, g) for (k, g) in get(d, "gen", Dict())
           if Int(get(g, "gen_status", get(g, "status", 1))) == 1]
    minpool = Int(get(p, "costs_min_pool", 2))
    length(act) < minpool && return Dict("cost_shuffle_pct" => 0.0)
    ns = clamp(round(Int, length(act) * p["cost_frac"]), minpool, length(act))
    sel = [act[i] for i in randperm(rng, length(act))[1:ns]]
    by = Dict{Int,Vector{Int}}()
    for (i, (_, g)) in enumerate(sel)
        push!(get!(by, get(g, "ncost", length(get(g, "cost", []))), Int[]), i)
    end
    n = 0
    for (_, ix) in by                       # shuffle only within equal-degree pools
        length(ix) < minpool && continue
        cs = [try Float64.(sel[i][2]["cost"]) catch; Float64[] end for i in ix]
        pm = randperm(rng, length(ix))
        for (j, i) in enumerate(ix)
            !isempty(cs[pm[j]]) && (sel[i][2]["cost"] = cs[pm[j]])
        end
        n += length(ix)
    end
    Dict("cost_shuffle_pct" => round(n / max(1, length(d["gen"])) * 100, digits=2))
end

"""
    mode_killgen!(d, rng, p)

Generator outages (N-k). Trips `k` units, `k` drawn from `killgen_nk` by inverse CDF over
`killgen_probs`. Candidates are online units above `killgen_pmax_threshold`, and the draw
is clamped so at least `killgen_min_online` stay online.
"""
function mode_killgen!(d, rng, p)
    nks   = p["killgen_nk"]
    probs = p["killgen_probs"]
    thr   = Float64(get(p, "killgen_pmax_threshold", 0.01))
    minon = Int(get(p, "killgen_min_online", 2))
    r = rand(rng); nk = nks[end]; c = 0.0
    for (i, pr) in enumerate(probs)           # inverse-CDF pick over killgen_nk
        c += Float64(pr)
        if r < c
            nk = nks[i]; break
        end
    end
    act = [(k, g) for (k, g) in get(d, "gen", Dict())
           if Int(get(g, "gen_status", get(g, "status", 1))) == 1 && get(g, "pmax", 0.0) > thr]
    length(act) <= maximum(nks) && return Dict("n_gens_killed" => 0)
    nk = min(nk, length(act) - minon)
    nk < 1 && return Dict("n_gens_killed" => 0)
    for i in randperm(rng, length(act))[1:nk]
        act[i][2]["gen_status"] = 0
    end
    Dict("n_gens_killed" => nk)
end

"""
    mode_derate!(d, rng, p)

Thermal congestion. Scales `rate_a`/`rate_b`/`rate_c` down by `U[derate_lo, derate_hi]`,
drawn independently per branch, on `derate_frac` of the in-service rated branches.
Branches with `rate_a == 0` are never candidates.
"""
function mode_derate!(d, rng, p)
    br = [(k, b) for (k, b) in get(d, "branch", Dict())
          if Int(get(b, "br_status", 1)) == 1 && get(b, "rate_a", 0.0) > 0]
    isempty(br) && return Dict("n_lines_derated" => 0)
    nd = max(1, round(Int, length(br) * p["derate_frac"]))
    lo, hi = p["derate_lo"], p["derate_hi"]
    fs = Float64[]
    for i in randperm(rng, length(br))[1:nd]
        f = lo + rand(rng) * (hi - lo); _, b = br[i]
        push!(fs, f)
        b["rate_a"] *= f
        haskey(b, "rate_b") && (b["rate_b"] *= f)
        haskey(b, "rate_c") && (b["rate_c"] *= f)
    end
    # Severity is the MEAN derate factor; the affected-branch count is fixed by
    # derate_frac and so carries no information.
    Dict("n_lines_derated" => nd,
         "derate_factor_mean" => round(sum(fs)/length(fs), digits=4),
         "derate_factor_min" => round(minimum(fs), digits=4))
end

"""
    mode_vsqueeze!(d, rng, p)

Voltage-band tightening. On `vsqueeze_frac` of buses, raises `vmin` and lowers `vmax` by
independent draws in `[0, vsqueeze_delta]`. A bus whose band would invert reverts to its
original band and is excluded from the mean-shrink statistic.
"""
function mode_vsqueeze!(d, rng, p)
    bs = collect(get(d, "bus", Dict()))
    isempty(bs) && return Dict("n_buses_vsqueezed" => 0)
    ns = max(1, round(Int, length(bs) * p["vsqueeze_frac"]))
    δ = p["vsqueeze_delta"]
    shrink = Float64[]
    for i in randperm(rng, length(bs))[1:ns]
        _, b = bs[i]
        lo, hi = get(b, "vmin", 0.9), get(b, "vmax", 1.1)
        b["vmin"] = lo + δ * rand(rng); b["vmax"] = hi - δ * rand(rng)
        if b["vmin"] >= b["vmax"]
            b["vmin"] = lo; b["vmax"] = hi                          # revert if inverted
        else
            push!(shrink, (hi - lo) - (b["vmax"] - b["vmin"]))
        end
    end
    Dict("n_buses_vsqueezed" => ns,
         "band_shrink_mean" => round(isempty(shrink) ? 0.0 : sum(shrink)/length(shrink),
                                     digits=5))
end

const MODE_FNS = Dict("loads" => mode_loads!, "costs" => mode_costs!,
                      "killgen" => mode_killgen!, "derate" => mode_derate!,
                      "vsqueeze" => mode_vsqueeze!)

# Application order. NOT alphabetical and not the config's key order: `costs` reshuffles
# the merit order over the units that are still online, so it must run BEFORE `killgen`
# takes any of them out. `derate`/`vsqueeze` touch branches and buses and are
# order-independent, but are pinned here so the RNG stream stays reproducible.
const COMPOSE_ORDER = ("loads", "costs", "killgen", "derate", "vsqueeze")

# Descriptor defaults, so a scenario where a mode did not fire records an explicit
# neutral value rather than a missing key: statistics compute marginal yields off these,
# and a missing key and a real zero would otherwise be indistinguishable.
const COMPOSE_DEFAULTS = Dict{String,Any}(
    "system_load_factor" => 1.0, "cost_shuffle_pct" => 0.0, "n_gens_killed" => 0,
    "n_lines_derated" => 0, "derate_factor_mean" => 1.0, "derate_factor_min" => 1.0,
    "n_buses_vsqueezed" => 0, "band_shrink_mean" => 0.0)

"""
    apply_composed!(d, rng, p) -> Dict

Apply every mode in `COMPOSE_ORDER` whose `mode_probs` draw succeeds to the same grid `d`,
and return the merged descriptors plus `modes_applied`.

A gate is drawn for every mode even when its probability is >= 1.0, so the RNG stream does
not depend on the probability VALUES: lowering `killgen` from 0.30 to 0.20 changes which
scenarios trip a unit without also reshuffling every load factor.

`perturb_param` is the system load factor. `loads` is configured to fire on every
scenario, so it is the only severity scalar defined for all of them; the per-mode
descriptors carry the rest.
"""
function apply_composed!(d, rng, p)
    probs = get(p, "mode_probs", Dict{String,Any}())
    extra = copy(COMPOSE_DEFAULTS)
    fired = String[]
    for name in COMPOSE_ORDER
        pr = Float64(get(probs, name, 0.0))
        gate = rand(rng)                      # drawn unconditionally; see docstring
        (pr <= 0.0 || gate >= pr) && continue
        for (k, v) in MODE_FNS[name](d, rng, p)
            extra[k] = v
        end
        push!(fired, name)
    end
    extra["modes_applied"] = fired
    extra["n_modes_applied"] = length(fired)
    extra["perturb_param"] = extra["system_load_factor"]
    return extra
end
