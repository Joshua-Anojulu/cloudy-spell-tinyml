"""All computations of the paper. Each sub-command writes one JSON file under results/.

    python3 experiments.py grid 1,2,5,10      # the 60 settings (fixed, cascade, optimal), by battery size
    python3 experiments.py grid 20
    python3 experiments.py checks             # Monte Carlo and boundary checks
    python3 experiments.py night 1,2,5,10     # night-heavy events
    python3 experiments.py night 20
    python3 experiments.py robust             # battery-reading error, capacity loss
    python3 experiments.py sites              # replay of recorded daily irradiance (NASA POWER)
    python3 experiments.py menu               # the findings under other model menus
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np

import solar_node as sn

HS = (6, 8, 12)
PS = (0.5, 0.7, 0.85, 0.95)
CS = (1, 2, 5, 10, 20)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")

# share of the day's events per hour: 70% in the twelve dark hours (18:00 to 06:00), 30% in daylight
NIGHT_SHARE = 0.7
NIGHT_EVENTS = tuple((NIGHT_SHARE / 12.0) if (h < 6 or h >= 18) else ((1.0 - NIGHT_SHARE) / 12.0) for h in range(24))


def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    return path


def k_out(k):
    return "never" if k is sn.NEVER else int(k)


def boundary_switches(ev, policy_first_of_level):
    """Expected model switches per day at the stationary law: inside the day plus across midnight."""
    pi, tabs = ev["pi"], ev["tabs"]
    total = ev["switches_in_day"]
    for w in (sn.SUN, sn.CLOUD):
        nxt_first = policy_first_of_level(tabs[w]["end"])
        total += float(pi[w] @ (nxt_first != tabs[w]["last"]))
    return total


def structure_of_optimal(act, pi_mid, setting):
    """How the optimal table looks along the battery axis: number of action changes per (weather, hour)
    over all levels, and whether the action never decreases with the level."""
    changes = (np.diff(act, axis=2) != 0).sum(axis=2)          # [w, h]
    monotone = bool((np.diff(act, axis=2) >= 0).all())
    return {"max_changes": int(changes.max()), "mean_changes": float(changes.mean()), "monotone_in_level": monotone,
            "table_entries": int(act.size)}


def one_setting(H, p, C, event_weights=sn.UNIFORM_EVENTS, save_policy=None):
    s = sn.Setting(H=H, p=p, C=C, event_weights=event_weights)
    rec = {"H": H, "p": p, "C": C, "cap_units": s.cap}
    for m in sn.MODELS:
        ev = sn.evaluate(s, sn.fixed_policy(m))
        rec[f"fixed_{m}"] = ev["rate"]
        rec[f"spill_{m}"] = ev["spill_frac"]
        rec[f"brownout_hours_{m}"] = float(ev["hours"][sn.SLEEP])
    t0 = time.time()
    lo, hi, act = sn.optimal_policy(s)
    ev = sn.evaluate(s, sn.table_policy(act))
    rec.update(opt_gain_lo=lo, opt_gain_hi=hi, opt=ev["rate"], opt_spill=ev["spill_frac"],
               opt_hours=[float(x) for x in ev["hours"]], opt_seconds=time.time() - t0)
    rec["opt_structure"] = structure_of_optimal(act, ev["pi"], s)
    if save_policy:
        np.save(save_policy, act.astype(np.int8))
    t0 = time.time()
    bc = sn.best_cascade(s)
    u1, u2 = sn.cascade_thresholds(bc["k1"], bc["k2"], s.cap)
    pol = sn.cascade_policy_units(u1, u2)
    ev = sn.evaluate(s, pol)
    first = lambda levels: pol(levels, 0, 0)
    rec.update(cascade=ev["rate"], cascade_k1=k_out(bc["k1"]), cascade_k2=k_out(bc["k2"]), cascade_u1=int(u1), cascade_u2=int(u2),
               cascade_pairs=bc["pairs"], cascade_spill=ev["spill_frac"], cascade_hours=[float(x) for x in ev["hours"]],
               cascade_switches_per_day=boundary_switches(ev, first), cascade_seconds=time.time() - t0)
    return rec


def cmd_grid(cs, tag="grid", event_weights=sn.UNIFORM_EVENTS):
    out = []
    for H, p, C in itertools.product(HS, PS, cs):
        pol_path = os.path.join(RES, f"opt_policy_{tag}_H{H}_p{p}_C{C}.npy") if tag == "grid" else None
        rec = one_setting(H, p, C, event_weights, save_policy=pol_path)
        out.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()
                          if k not in ("opt_hours", "cascade_hours", "opt_structure")}), flush=True)
        save(f"{tag}_C{'-'.join(str(c) for c in cs)}.json", out)


def cmd_checks():
    """Monte Carlo on 4 settings drawn at random (fixed seed), boundary cases, dominance, energy balance."""
    rng = np.random.default_rng(20261125)
    allset = list(itertools.product(HS, PS, CS))
    picks = [allset[i] for i in rng.choice(len(allset), size=4, replace=False)]
    mc = []
    for H, p, C in picks:
        s = sn.Setting(H=H, p=p, C=C)
        bc = sn.best_cascade(s)
        pols = {"S": sn.fixed_policy("S"), "M": sn.fixed_policy("M"), "L": sn.fixed_policy("L"),
                "cascade": sn.cascade_policy(bc["k1"], bc["k2"], s.cap)}
        lo, hi, act = sn.optimal_policy(s)
        pols["optimal"] = sn.table_policy(act)
        for name, pol in pols.items():
            exact = sn.evaluate(s, pol)["rate"]
            rates, bal = sn.simulate(s, pol, days=3000, replicas=400, rng=rng)
            se = float(rates.std(ddof=1) / np.sqrt(len(rates)))
            z = (float(rates.mean()) - exact) / se if se > 0 else 0.0
            mc.append({"H": H, "p": p, "C": C, "policy": name, "exact": exact, "mc_mean": float(rates.mean()),
                       "mc_se": se, "z": z, "energy_balance_violation_units": bal})
            print(json.dumps(mc[-1]), flush=True)
    # boundary cases
    bound = {}
    s = sn.Setting(H=60.0, p=0.5, C=20.0, cloud_factor=1.0)          # abundant sun, battery carries L overnight
    bound["abundant"] = {m: sn.evaluate(s, sn.fixed_policy(m))["rate"] for m in sn.MODELS}
    lo, hi, act = sn.optimal_policy(s)
    bound["abundant"]["optimal"] = sn.evaluate(s, sn.table_policy(act))["rate"]
    s = sn.Setting(H=60.0, p=0.5, C=2.0, cloud_factor=1.0)           # the same sun with a 2 J battery
    bound["abundant_small_battery_L"] = sn.evaluate(s, sn.fixed_policy("L"))["rate"]
    s = sn.Setting(H=0.0, p=0.5, C=10.0)
    bound["no_sun"] = {m: sn.evaluate(s, sn.fixed_policy(m))["rate"] for m in sn.MODELS}
    # power iteration against the direct solve
    worst = 0.0
    for H, p, C in picks:
        s = sn.Setting(H=H, p=p, C=C)
        for m in sn.MODELS:
            tabs = sn.day_tables(s, sn.fixed_policy(m))
            a = sn.stationary_midnight(s, [t["end"] for t in tabs])
            b = sn.stationary_midnight_direct(s, [t["end"] for t in tabs])
            ra = sum(float(a[w] @ tabs[w]["det"]) for w in (0, 1))
            rb = sum(float(b[w] @ tabs[w]["det"]) for w in (0, 1))
            worst = max(worst, abs(ra - rb))
    out = {"monte_carlo": mc, "max_abs_z": max(abs(r["z"]) for r in mc), "boundary": bound,
           "max_energy_balance_violation_units": max(r["energy_balance_violation_units"] for r in mc),
           "power_vs_direct_max_abs_diff": worst, "picked_settings": picks, "mc_days": 3000, "mc_replicas": 400}
    print(json.dumps({k: v for k, v in out.items() if k != "monte_carlo"}))
    save("checks.json", out)


def load_grid(tag="grid"):
    recs = []
    for name in sorted(os.listdir(RES)):
        if name.startswith(tag + "_C") and name.endswith(".json"):
            recs += json.load(open(os.path.join(RES, name)))
    return {(r["H"], r["p"], r["C"]): r for r in recs}


def cmd_robust():
    """At the cascade thresholds tuned for the nominal node: (i) the level is read with an error,
    (ii) the battery has lost capacity. Uses the grid results (run 'grid' first)."""
    grid = load_grid()
    out = []
    for (H, p, C), r in sorted(grid.items()):
        s = sn.Setting(H=H, p=p, C=C)
        u1, u2 = r["cascade_u1"], r["cascade_u2"]
        rec = {"H": H, "p": p, "C": C, "cascade": r["cascade"], "best_fixed": max(r["fixed_S"], r["fixed_M"], r["fixed_L"]),
               "opt": r["opt"]}
        for pct in (2, 5, 10):
            err = max(1, int(round(pct / 100.0 * s.cap)))
            rec[f"noisy_{pct}"] = sn.evaluate_noisy_cascade(s, u1, u2, err)["rate"]
        # 20% of the capacity lost; thresholds stay where they were set (in joules)
        s80 = sn.Setting(H=H, p=p, C=0.8 * C)
        rec["faded_old_thresholds"] = sn.rate_only(s80, sn.cascade_policy_units(u1, u2))
        rec["faded_retuned"] = sn.best_cascade(s80)["rate"]
        rec["faded_best_fixed"] = max(sn.rate_only(s80, sn.fixed_policy(m)) for m in sn.MODELS)
        out.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        save("robust.json", out)


# ---------------------------------------------------------------------------------------------
# recorded irradiance
# ---------------------------------------------------------------------------------------------

SITES = {"dhaka": ("Dhaka, Bangladesh", 23.81, 90.41), "ibadan": ("Ibadan, Nigeria", 7.38, 3.95),
         "niamey": ("Niamey, Niger", 13.51, 2.11)}
TRAIN_DAYS = 730            # 2021 and 2022; the remaining days (2023 to 2025) are the test period


def load_site(name):
    digits = "".join(line.strip() for line in open(os.path.join(DATA, f"{name}_2021_2025.digits")))
    return np.array([int(digits[i:i + 3]) for i in range(0, len(digits), 3)], dtype=float) / 100.0   # kWh/m2/day


def fit_two_state(g):
    """Two-state fit of a daily record: days above the median are 'sunny', the others 'cloudy'."""
    med = float(np.median(g))
    cls = np.where(g > med, sn.SUN, sn.CLOUD)
    sun = float(g[cls == sn.SUN].mean())
    cloud = float(g[cls == sn.CLOUD].mean())
    stay = float((cls[1:] == cls[:-1]).mean())
    return {"median": med, "sun_level": sun, "cloud_level": cloud, "cloud_factor": cloud / sun, "p_stay": stay}, cls


def spell_lengths(cls, which):
    runs, n = [], 0
    for c in cls:
        if c == which:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def cmd_sites(n_shuffles=200):
    rng = np.random.default_rng(244)
    out = {"train_days": TRAIN_DAYS, "n_shuffles": n_shuffles, "sites": {}}
    for name, (label, lat, lon) in SITES.items():
        g = load_site(name)
        fit_all, cls_all = fit_two_state(g)
        fit_tr, _ = fit_two_state(g[:TRAIN_DAYS])
        cls_by_train_median = np.where(g > fit_tr["median"], sn.SUN, sn.CLOUD)
        cloudy_runs = spell_lengths(cls_all, sn.CLOUD)
        site = {"label": label, "lat": lat, "lon": lon, "days": int(len(g)), "mean_kwh": float(g.mean()),
                "sd_kwh": float(g.std()), "cv": float(g.std() / g.mean()), "min_kwh": float(g.min()), "max_kwh": float(g.max()),
                "lag1": float(np.corrcoef(g[:-1], g[1:])[0, 1]), "fit_all": fit_all, "fit_train": fit_tr,
                "cloudy_spell_mean": float(np.mean(cloudy_runs)), "cloudy_spell_max": int(max(cloudy_runs)),
                "monthly_mean_kwh": None, "settings": []}
        # calendar months over the five years (2021-01-01 is day 0)
        import datetime
        d0 = datetime.date(2021, 1, 1)
        months = np.array([(d0 + datetime.timedelta(days=int(i))).month for i in range(len(g))])
        site["monthly_mean_kwh"] = [float(g[months == m].mean()) for m in range(1, 13)]
        for H, C in itertools.product(HS, CS):
            t0 = time.time()
            cap = sn.to_units(C)
            mean_j = 0.6 * H                               # the same mean harvest as the two-state model with this H
            dj = g * (mean_j / g.mean())                   # joules per day
            te = slice(TRAIN_DAYS, None)
            tr = slice(0, TRAIN_DAYS)
            rec = {"H": H, "C": C, "mean_harvest_j": mean_j}
            fixed_actions = np.array([sn.RUN_S, sn.RUN_M, sn.RUN_L], dtype=np.int64)
            pol_fixed3 = lambda b, w, h: np.broadcast_to(fixed_actions, np.shape(b))
            # (a) the record in its order, start full: all five years, and the test years alone
            rp = sn.replay(dj, pol_fixed3, cap, np.full(3, cap))
            for i, m in enumerate(sn.MODELS):
                rec[f"fixed_{m}"] = float(rp["daily"][i].mean())
                rec[f"spill_{m}"] = float(rp["spilled"][i].sum() / rp["harvested"][i].sum())
                rec[f"train_fixed_{m}"] = float(rp["daily"][i, tr].mean())
                rec[f"test_fixed_{m}"] = float(rp["daily"][i, te].mean())
            # (b) the same days in shuffled order
            perm = np.stack([rng.permutation(len(dj)) for _ in range(n_shuffles)])
            rp = sn.replay(dj[perm][:, None, :], pol_fixed3, cap, np.full((n_shuffles, 3), cap))
            for i, m in enumerate(sn.MODELS):
                r = rp["daily"][:, i, :].mean(axis=1)
                rec[f"shuffled_{m}_mean"] = float(r.mean())
                rec[f"shuffled_{m}_sd"] = float(r.std(ddof=1))
                rec[f"shuffled_{m}_min"] = float(r.min())
                rec[f"shuffled_{m}_max"] = float(r.max())
            # (c) cascade: thresholds chosen on the first two years, scored on the last three
            grid = list(sn.cascade_grid(cap))
            u1s = np.array([x[2] for x in grid], dtype=np.int64)
            u2s = np.array([x[3] for x in grid], dtype=np.int64)
            def pol_grid(b, w, h):
                e = b - sn.SLEEP_U
                return np.where(e >= u2s, sn.RUN_L, np.where(e >= u1s, sn.RUN_M, sn.RUN_S)).astype(np.int64)
            rp = sn.replay(dj, pol_grid, cap, np.full(len(grid), cap))
            train_rate = rp["daily"][:, tr].mean(axis=1)
            test_rate = rp["daily"][:, te].mean(axis=1)
            j = int(np.argmax(train_rate))
            rec.update(cascade_train=float(train_rate[j]), cascade_test=float(test_rate[j]),
                       cascade_test_spill=float(rp["spilled"][j, te].sum() / rp["harvested"][j, te].sum()),
                       cascade_k1=k_out(grid[j][0]), cascade_k2=k_out(grid[j][1]),
                       cascade_test_oracle_thresholds=float(test_rate.max()), cascade_all_years_best=float(rp["daily"].mean(axis=1).max()))
            # (d) cascade thresholds from the two-state model fitted to the training years (no replay used to tune)
            sfit = sn.Setting(H=mean_j * 2.0 / (1.0 + fit_tr["cloud_factor"]), p=fit_tr["p_stay"], C=C,
                              cloud_factor=fit_tr["cloud_factor"])
            bc = sn.best_cascade(sfit)
            jm = [i for i, x in enumerate(grid) if x[0] == bc["k1"] and x[1] == bc["k2"]][0]
            rec.update(cascade_modeltuned_test=float(test_rate[jm]), cascade_modeltuned_model_value=bc["rate"])
            # (e) the fitted model's optimal policy, run with the day's class known at midnight
            lo, hi, act = sn.optimal_policy(sfit)
            rp = sn.replay(dj, sn.table_policy(act), cap, cap, weather=cls_by_train_median)
            rec["mdp_fit_test"] = float(rp["daily"][te].mean())
            rec["mdp_fit_model_value"] = float(sn.rate_only(sfit, sn.table_policy(act)))
            rec["fit_model_fixed"] = {m: float(sn.rate_only(sfit, sn.fixed_policy(m))) for m in sn.MODELS}
            # (f) the future known: upper bound on the test years (from a full battery, the best possible start)
            rec["clairvoyant_test"] = sn.clairvoyant(dj[te], cap, cap)
            rec["seconds"] = time.time() - t0
            site["settings"].append(rec)
            print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                                    if k not in ("fit_model_fixed",)}), flush=True)
            out["sites"][name] = site
            save("sites.json", out)
    save("sites.json", out)


MENUS = {
    # name: (sleep floor units/h, inference units/h of S, M, L, recalls of S, M, L); one unit is 6 mJ
    "paper": (6, (5, 20, 80), (0.80, 0.90, 0.95)),
    "energy x2 per size": (6, (5, 10, 20), (0.80, 0.90, 0.95)),
    "energy x8 per size": (6, (5, 40, 320), (0.80, 0.90, 0.95)),
    "sleep floor halved": (3, (5, 20, 80), (0.80, 0.90, 0.95)),
    "sleep floor doubled": (12, (5, 20, 80), (0.80, 0.90, 0.95)),
    "recalls 0.70/0.85/0.95": (6, (5, 20, 80), (0.70, 0.85, 0.95)),
    "recalls 0.85/0.90/0.93": (6, (5, 20, 80), (0.85, 0.90, 0.93)),
}
MENU_CS = (1, 2, 5, 10)


def cmd_menu():
    """The three findings under other model menus (batteries up to 10 J, all suns and persistences)."""
    out = {}
    for name, (sleep_u, infer, recall) in MENUS.items():
        menu = sn.Menu(sleep_u=sleep_u, infer_u=(0,) + tuple(infer), recall=(0.0,) + tuple(recall))
        recs = []
        t0 = time.time()
        for H, p, C in itertools.product(HS, PS, MENU_CS):
            s = sn.Setting(H=H, p=p, C=C, menu=menu)
            rec = {"H": H, "p": p, "C": C}
            for m in sn.MODELS:
                ev = sn.evaluate(s, sn.fixed_policy(m))
                rec[f"fixed_{m}"] = ev["rate"]
                rec[f"spill_{m}"] = ev["spill_frac"]
            lo, hi, act = sn.optimal_policy(s)
            rec["opt"] = sn.rate_only(s, sn.table_policy(act))
            rec["cascade"] = sn.best_cascade(s)["rate"]
            recs.append(rec)
        out[name] = {"sleep_u": sleep_u, "infer_u": list(infer), "recall": list(recall), "settings": recs,
                     "seconds": time.time() - t0}
        print(name, round(time.time() - t0), "s", flush=True)
        save("menu.json", out)


def main():
    os.makedirs(RES, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "grid":
        cmd_grid([int(x) for x in sys.argv[2].split(",")])
    elif cmd == "night":
        cmd_grid([int(x) for x in sys.argv[2].split(",")], tag="night", event_weights=NIGHT_EVENTS)
    elif cmd == "checks":
        cmd_checks()
    elif cmd == "robust":
        cmd_robust()
    elif cmd == "sites":
        cmd_sites()
    elif cmd == "menu":
        cmd_menu()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
