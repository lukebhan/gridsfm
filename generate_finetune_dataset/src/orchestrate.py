"""Spawn and monitor independent Julia workers until the feasible-case target is met.

SCHEDULING: one global pool, every worker on it, stop the instant the target is met.

That is deliberately the whole policy. Nothing here estimates or tracks yield, because
nothing needs to: the target is a count of FEASIBLE cases, and the only way to know
whether a scenario is feasible is to solve it. Batches deliberately over-provision and
the stop marker ends the round, which bounds the wasted work to whatever is in flight at
that instant -- at most `num_workers - 1` solves, the floor for any parallel scheme.

Scenarios compose their perturbations (`mode_probs`), so every case draws from the same
distribution and there is nothing to balance: one pool, one target, one cutoff.

  * Workers are independent processes, not Distributed workers, so one dying costs one
    shard rather than the run.
  * The stop marker (`_work/_stop_multi`) carries the cutoff to the workers: each checks
    it before every case, so no signal handling or mid-solve kill is needed and no
    partially written case is produced.
  * Every case is a separate file, so a run is resumable and killable.
"""
from __future__ import annotations
import json, os, subprocess, time
import dataset, stats as statsmod

# The label every composed scenario carries: its record `mode`, its staging filename
# prefix, and the `mode` field of its scenario seed. The single unperturbed case uses
# "base" instead.
COMPOSED = "multi"

# Mode application order, mirroring COMPOSE_ORDER in perturb_modes.jl. Used for
# reporting only, the Julia side is the authority on what actually runs.
MODE_ORDER = ("loads", "costs", "killgen", "derate", "vsqueeze")

# Poll interval, and the waste knob: after the target is reached, workers keep going
# until they next check the marker, so a slow poll costs extra solves. 3 s against a
# multi-second solve keeps that to roughly the in-flight set.
POLL_SECONDS = 3
# How long a worker gets to finish its in-flight case once the target is met.
DRAIN_TIMEOUT = 120
# Give up after this many batches rather than looping forever on a config whose yield is
# too low to ever reach the target.
MAX_ROUNDS = 12
# --dry_run: enough cases to exercise the whole path without a meaningful solve budget.
DRY_RUN_CASES = 5
DRY_RUN_ROUNDS = 2


def _shard(tasks: list[tuple[str, int]], n: int) -> list[list[tuple[str, int]]]:
    """Deal the scenarios round-robin across all workers."""
    out = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        out[i % n].append(t)
    return [s for s in out if s]


def generate(cfg: dict, outdir: str, num_workers: int, total_num_feasible: int,
             julia: str, project: str, max_rounds: int = MAX_ROUNDS,
             log=print,
             published: int = 0) -> dict:
    os.makedirs(outdir, exist_ok=True)
    workdir = os.path.join(outdir, "_work"); os.makedirs(workdir, exist_ok=True)
    recdir = os.path.join(os.path.dirname(os.path.abspath(outdir)), "_records")
    os.makedirs(recdir, exist_ok=True)
    # config handed to Julia as JSON (no YAML dependency on the Julia side)
    cfg_json = os.path.join(workdir, "config.json")
    with open(cfg_json, "w") as f:
        json.dump({"grid_id": cfg["grid_id"], "topology_abs": cfg["_topology_abs"],
                   "seeds": cfg["seeds"], "solver": cfg["solver"],
                   "perturbations": cfg["perturbations"]}, f, indent=1)

    # Never reuse a scenario index that has already been EXPLORED (feasible or not), or
    # the same perturbation would be regenerated and duplicated.
    explored = statsmod.load_records(recdir)
    next_idx = 1 + max((r["sidx"] for r in explored if r.get("mode") == COMPOSED),
                       default=0)
    if explored:
        log(f"resuming: {len(explored)} scenarios already explored; "
            f"continuing from sidx {next_idx}")
    if published:
        log(f"already published: {published}  (target {total_num_feasible})")
    first_new_idx = next_idx           # everything from here on is this run's work

    def feasible_now() -> int:
        """Feasible cases available: what is already published plus what this run has
        staged. Records alone over-count, because the publish step trims to the target and
        deletes the surplus staging files, so on an --append run they would claim cases
        that no longer exist."""
        recs = statsmod.load_records(recdir)
        staged = sum(1 for r in recs if r.get("mode") == COMPOSED and r.get("feasible")
                     and r["sidx"] >= first_new_idx)
        return published + staged

    def run_batch(mode: str, tasks: list[tuple[str, int]], tag: str,
                  target: int) -> int:
        """Put every worker on the pool; stop as soon as the target is met.
        Returns the feasible count when the batch ends."""
        sm = os.path.join(workdir, f"_stop_{mode}")
        os.path.exists(sm) and os.remove(sm)          # stale marker from an earlier batch
        shards = _shard(tasks, num_workers)
        procs = []
        for i, sh in enumerate(shards):
            sp = os.path.join(workdir, f"shard_{tag}_{i}.txt")
            with open(sp, "w") as f:
                f.write("\n".join(f"{mm} {ss}" for mm, ss in sh))
            lp = open(os.path.join(workdir, f"worker_{tag}_{i}.log"), "w")
            procs.append((subprocess.Popen(
                [julia, f"--project={project}", os.path.join(os.path.dirname(__file__), "worker.jl"),
                 cfg_json, sp, outdir, recdir], stdout=lp, stderr=subprocess.STDOUT), lp))

        t0 = time.time()
        got = feasible_now()
        tick = 0
        while any(p.poll() is None for p, _ in procs):
            time.sleep(POLL_SECONDS)
            tick += 1
            got = feasible_now()
            alive = sum(1 for p, _ in procs if p.poll() is None)
            if got >= target:
                if not os.path.exists(sm):
                    open(sm, "w").write(str(got))
                    log(f"\n  target met ({got}/{target}) at "
                        f"{(time.time()-t0)/60:.1f} min; {alive} workers draining "
                        f"(they skip the rest)")
                for p, _ in procs:                   # in-flight cases finish, the rest skip
                    try:
                        p.wait(timeout=DRAIN_TIMEOUT)
                    except Exception:
                        p.kill()
                break
            if tick % 10 == 0:                        # log every ~30 s, not every poll
                log(f"  {got}/{target} feasible | {alive}/{len(procs)} "
                    f"workers | {(time.time()-t0)/60:.1f} min", end="\r")
        for p, lp in procs:
            if p.poll() is None:
                p.terminate()
            lp.close()
        return feasible_now()

    # the unperturbed base case, once, before the pool
    if not os.path.isfile(os.path.join(outdir, "base_unperturbed.pyg.json")):
        run_batch("base", [("base", 0)], "base", target=1)

    # ---- ONE POOL, every worker on it, stop at the target ----
    got = feasible_now()
    if got >= total_num_feasible:
        log(f"already at {got}/{total_num_feasible} feasible, nothing to generate")
    else:
        log(f"\n{got}/{total_num_feasible} feasible, {num_workers} workers on it")
        for attempt_round in range(1, max_rounds + 1):
            need = total_num_feasible - got
            # No yield model: ask for a batch big enough that no worker idles, and let the
            # stop marker end it. If the batch runs dry before the target (a low-yield
            # config), the loop simply asks for another.
            batch = max(num_workers, need * 2)
            tasks = [(COMPOSED, next_idx + k) for k in range(batch)]
            next_idx += batch
            log(f"  batch {attempt_round}: {batch} scenarios "
                f"(sidx {tasks[0][1]}-{tasks[-1][1]}) for {need} more feasible")
            got = run_batch(COMPOSED, tasks, f"{COMPOSED}_{attempt_round}",
                            target=total_num_feasible)
            log(f"  {got}/{total_num_feasible} feasible after batch {attempt_round}")
            if got >= total_num_feasible:
                break
        else:
            log(f"  WARNING: stopped at {got}/{total_num_feasible} after "
                f"{max_rounds} batches")

    final = dataset.scan(outdir)
    return dict(feasible=sum(final["feasible"].values()),
                per_mode=dict(final["feasible"]), infeasible=dict(final["infeasible"]),
                bad=final["bad"], feasible_files=final["feasible_files"], recdir=recdir)
