"""Turn the result files into the macros and tables of the paper, and check the claims the text makes.

Writes paper/numbers.tex, paper/tab_main.tex, paper/tab_sites.tex, paper/tab_robust.tex and
results/numbers_catalogue.tsv (macro, value, where it comes from). Every statement of the text that is
qualitative ("never", "always", "at most") is asserted here; the build stops if one fails."""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

import experiments as ex
import solar_node as sn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.join(HERE, "..", "paper")
CODE_URL = os.environ.get("AIOT_CODE_URL", "https://github.com/Joshua-Anojulu/cloudy-spell-tinyml")

MACROS = {}
CATALOGUE = []
CHECKS = []


def fmt(x, nd=1):
    """Round half up after removing floating-point noise."""
    d = Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = f"{d:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def put(name, value, source):
    assert name.isalpha(), name
    assert name not in MACROS, f"duplicate macro {name}"
    value = str(value)
    if len(value) > 1 and value[0] == "-" and value[1].isdigit():
        value = "$-$" + value[1:]              # a minus sign, not a hyphen
    MACROS[name] = value
    CATALOGUE.append((name, value.replace("$-$", "-"), source))


def guard(cond, what):
    CHECKS.append((bool(cond), what))
    if not cond:
        raise SystemExit("CLAIM NOT SUPPORTED: " + what)


def pct(x, nd=1):
    return fmt(100.0 * x, nd)


def words(n):
    return {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten"}.get(n, str(n))


def join_list(items):
    items = [str(i) for i in items]
    return ", ".join(items)


def join_and(items):
    items = [str(i) for i in items]
    return ", ".join(items[:-1]) + " and " + items[-1]


def best_fixed(r):
    return max(r["fixed_S"], r["fixed_M"], r["fixed_L"])


def best_fixed_name(r):
    return max("SML", key=lambda m: r[f"fixed_{m}"])


def main():
    os.makedirs(PAPER, exist_ok=True)
    grid = ex.load_grid("grid")
    night = ex.load_grid("night")
    robust = {(r["H"], r["p"], r["C"]): r for r in json.load(open(os.path.join(RES, "robust.json")))}
    sites = json.load(open(os.path.join(RES, "sites.json")))
    checks = json.load(open(os.path.join(RES, "checks.json")))
    keys = sorted(grid)
    guard(len(keys) == len(ex.HS) * len(ex.PS) * len(ex.CS) == 60, "60 settings computed")
    guard(sorted(night) == keys and sorted(robust) == keys, "night and robustness runs cover the same 60 settings")

    # ----- model constants -----
    m = sn.PAPER_MENU
    mj = lambda units: units * sn.Q_J * 1000.0
    put("NQuantumMj", fmt(sn.Q_J * 1000, 0), "solar_node.Q_J")
    put("NCloudFactor", fmt(sn.Setting(H=1, p=0.5, C=1).cloud_factor, 1), "solar_node.Setting.cloud_factor")
    put("NSleepUw", fmt(sn.SLEEP_J_PER_H / 3600 * 1e6, 0), "solar_node.SLEEP_J_PER_H")
    put("NSleepMjH", fmt(mj(m.sleep_u), 0), "menu")
    put("NInferPerHour", sn.INFER_PER_H, "solar_node.INFER_PER_H")
    for i, name in ((sn.RUN_S, "S"), (sn.RUN_M, "M"), (sn.RUN_L, "L")):
        put(f"NRecall{name}", fmt(m.recall[i], 2), "menu")
        e = sn.E_INF_J[name] * 1000
        put(f"NInfer{name}", fmt(e, 1) if e < 1 else fmt(e, 0), "menu")
        put(f"NInferHour{name}", fmt(mj(m.infer_u[i]), 0), "menu")
        put(f"NHour{name}", fmt(mj(m.cost_u[i]), 0), "menu")
        put(f"NDayJ{name}", fmt(24 * m.cost_u[i] * sn.Q_J, 2), "menu")
    put("NDayJSleep", fmt(24 * m.sleep_u * sn.Q_J, 2), "menu")
    guard(m.cost_u[sn.RUN_S] > m.sleep_u, "an hour of S costs more than an hour of sleep")
    put("NHList", join_list(ex.HS), "experiments.HS")
    put("NPList", join_list(ex.PS), "experiments.PS")
    put("NCList", join_list(ex.CS), "experiments.CS")
    means = [sn.Setting(H=h, p=0.5, C=1).mean_harvest_j for h in ex.HS]
    put("NMeanHarvestList", join_and([fmt(x, 1) for x in means]), "0.6 H")
    put("NMeanShare", fmt(means[0] / ex.HS[0], 1), "mean harvest / H")
    dayM = 24 * m.cost_u[sn.RUN_M] * sn.Q_J
    dayS = 24 * m.cost_u[sn.RUN_S] * sn.Q_J
    dayL = 24 * m.cost_u[sn.RUN_L] * sn.Q_J
    guard(means[0] < dayM < means[1], "M is unaffordable on average at the lowest sun and affordable at the others")
    guard(dayS < means[0] and dayL > means[-1], "S always fits the mean harvest and L never does")
    put("NnSettings", len(keys), "grid")

    # ----- checks -----
    put("NnMcSettings", words(len(checks["picked_settings"])), "checks.json")
    put("NMcReplicas", checks["mc_replicas"], "checks.json")
    put("NMcDays", checks["mc_days"], "checks.json")
    put("NMcMaxZ", fmt(checks["max_abs_z"] + 0.005, 2), "checks.json, rounded up")
    guard(checks["max_abs_z"] < 3.5, "Monte Carlo agrees with the exact rates (|z| < 3.5 on 20 comparisons)")
    guard(checks["max_energy_balance_violation_units"] == 0, "energy balance holds in simulation")
    guard(checks["power_vs_direct_max_abs_diff"] < 1e-9, "power iteration equals the direct solve")
    ab = checks["boundary"]["abundant"]
    guard(all(abs(ab[x] - sn.RECALL[x]) < 1e-9 for x in "SML"), "abundant sun: every fixed policy scores its recall")
    put("NAbundantOpt", fmt(ab["optimal"], 2), "checks.json")
    guard(abs(ab["optimal"] - sn.RECALL["L"]) < 1e-6, "abundant sun: the optimum is the recall of L")
    put("NAbundantSmallC", 2, "checks: C of the small-battery case")
    put("NAbundantSmallL", pct(checks["boundary"]["abundant_small_battery_L"], 0), "checks.json")
    guard(checks["boundary"]["abundant_small_battery_L"] < 0.95 - 0.1, "a 2 J battery cannot carry L through the night")
    guard(all(abs(v) < 1e-9 for v in checks["boundary"]["no_sun"].values()), "no sun: nothing detected")

    # ----- weather model: headline numbers -----
    flips = 0
    for H in ex.HS:
        for C in ex.CS:
            flips += best_fixed_name(grid[(H, 0.5, C)]) != best_fixed_name(grid[(H, 0.95, C)])
    put("NnFlip", flips, "grid: best fixed model at p=0.5 vs p=0.95")
    put("NnSunBattery", len(ex.HS) * len(ex.CS), "grid")
    spillS = [grid[k]["spill_S"] for k in keys]
    put("NSpillSMin", pct(min(spillS), 0), "grid: spill of always-S")
    put("NSpillSMax", pct(max(spillS), 0), "grid: spill of always-S")
    gains = [100 * (grid[k]["opt"] - best_fixed(grid[k])) for k in keys]
    put("NOptGainMin", fmt(min(gains), 1), "grid: optimum minus best fixed")
    put("NOptGainMax", fmt(max(gains), 1), "grid")
    put("NOptGainMedian", fmt(st.median(gains), 1), "grid")
    guard(all(best_fixed_name(grid[k]) != "L" for k in keys), "L is never the best fixed model")
    usesL = sum(grid[k]["opt_hours"][sn.RUN_L] > 1e-9 for k in keys)
    put("NnOptUsesL", usesL, "grid: hours of L under the optimal policy")
    guard(usesL == len(keys), "the optimal policy runs L in every setting")
    worst_spill = max(keys, key=lambda k: grid[k]["opt_spill"])
    put("NOptSpillMax", pct(grid[worst_spill]["opt_spill"], 0), "grid: spill of the optimal policy")
    put("NOptSpillMaxC", worst_spill[2], "grid")
    guard(all(grid[k]["opt_spill"] < min(spillS) for k in keys), "the optimal policy always spills less than always-S ever does")
    guard(all(grid[k]["opt_gain_hi"] - grid[k]["opt_gain_lo"] < 1e-7 for k in keys), "optimum certified to 1e-7")
    guard(all(grid[k]["opt"] >= max(best_fixed(grid[k]), grid[k]["cascade"]) - 1e-7 for k in keys),
          "the optimum is never below a fixed policy or the cascade")
    guard(all(grid[k]["cascade"] >= best_fixed(grid[k]) - 1e-9 for k in keys), "the cascade is never worse than the best fixed model")
    shares = {k: (grid[k]["cascade"] - best_fixed(grid[k])) / (grid[k]["opt"] - best_fixed(grid[k])) for k in keys}
    guard(min(gains) > 0.5, "the optimum gains more than half a point everywhere, so the share is defined in all settings")
    put("NCascShareMedian", pct(st.median(shares.values()), 0), "grid: (cascade - best fixed)/(optimum - best fixed)")
    kw = min(keys, key=lambda k: shares[k])
    put("NCascShareWorst", pct(shares[kw], 0), "grid")
    put("NCascShareWorstC", kw[2], "grid")
    cgap = [100 * (grid[k]["opt"] - grid[k]["cascade"]) for k in keys]
    put("NCascGapMax", fmt(max(cgap) + 0.05, 1), "grid: optimum minus cascade, rounded up")
    sw = [grid[k]["cascade_switches_per_day"] for k in keys]
    put("NSwitchMedian", fmt(st.median(sw), 1), "grid: model changes per day under the cascade")
    put("NSwitchMax", fmt(max(sw), 1), "grid")
    guard(max(sw) <= 6, "the cascade changes model a few times a day at most")
    put("NSwitchCostPct", fmt(100 * max(sw) * sn.E_INF_J["L"] / min(means) + 0.05, 1),
        "max switches per day x energy of one L inference / smallest mean daily harvest, rounded up")
    # example setting of the text and the figure
    exH, exC, exP = 8, 10, 0.95
    put("NExH", exH, "example"); put("NExC", exC, "example"); put("NExP", exP, "example")
    put("NFigH", exH, "figure")
    a, b = grid[(exH, 0.5, exC)], grid[(exH, exP, exC)]
    put("NExMInd", pct(a["fixed_M"]), "grid"); put("NExSInd", pct(a["fixed_S"]), "grid")
    put("NExMPers", pct(b["fixed_M"]), "grid")
    guard(a["fixed_M"] > a["fixed_S"] and b["fixed_S"] > b["fixed_M"], "example: M beats S at p=0.5 and S beats M at p=0.95")
    put("NExCasc", pct(b["cascade"]), "grid"); put("NExOpt", pct(b["opt"]), "grid")
    lmax = max(grid[(exH, p, C)]["fixed_L"] for p in (0.5, 0.95) for C in ex.CS)
    put("NFigLMax", int(np.ceil(100 * lmax)), "grid: always-L at H=8, rounded up")
    tk = max(keys, key=lambda k: k[2] == 10)
    put("NOptTableEntries", f"{grid[(8, 0.95, 10)]['opt_structure']['table_entries']:,}".replace(",", "{,}"), "grid")
    put("NOptTableC", 10, "grid")

    # ----- Table: main -----
    rows = []
    for H in ex.HS:
        for p in (0.5, 0.95):
            for C in (2, 10):
                r = grid[(H, p, C)]
                rows.append(f"{H} & {p:.2f} & {C} & {pct(r['fixed_S'])} & {pct(r['fixed_M'])} & {pct(r['fixed_L'])} & "
                            f"{pct(r['cascade'])} & {pct(r['opt'])} \\\\")
            if p == 0.95 and H != ex.HS[-1]:
                rows.append("\\addlinespace")
    tab = ("\\begin{tabular}{@{}rrr|rrr|rr@{}}\n\\toprule\n$H$ & $p$ & $C$ & S & M & L & casc. & opt. \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(PAPER, "tab_main.tex"), "w").write(tab)

    # ----- Table: what the policies do (hours per day), at the example sun and battery -----
    rows = []
    for p_ in (0.5, exP):
        r = grid[(exH, p_, exC)]
        idle_m = r["brownout_hours_M"]
        rows.append(f"{p_:.2f} & always M & {fmt(idle_m, 1)} & {fmt(0, 1)} & {fmt(24 - idle_m, 1)} & {fmt(0, 1)} & {pct(r['spill_M'], 0)} \\\\")
        for label, hk, sk in (("cascade", "cascade_hours", "cascade_spill"), ("optimal", "opt_hours", "opt_spill")):
            h = r[hk]
            rows.append(f" & {label} & {fmt(h[0], 1)} & {fmt(h[1], 1)} & {fmt(h[2], 1)} & {fmt(h[3], 1)} & {pct(r[sk], 0)} \\\\")
        if p_ == 0.5:
            rows.append("\\addlinespace")
    tab = ("\\begin{tabular}{@{}rlccccc@{}}\n\\toprule\n$p$ & policy & idle & S & M & L & spill (\\%) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(PAPER, "tab_hours.tex"), "w").write(tab)
    rb = grid[(exH, exP, exC)]
    put("NExMIdlePers", fmt(rb["brownout_hours_M"], 1), "grid: hours per day browned out, always-M, example with spells")
    put("NExMIdleInd", fmt(grid[(exH, 0.5, exC)]["brownout_hours_M"], 1), "grid: the same with independent days")
    put("NExOptLPers", fmt(rb["opt_hours"][sn.RUN_L], 1), "grid: hours of L per day, optimal policy, example with spells")
    put("NExOptSleepPers", fmt(rb["opt_hours"][sn.SLEEP], 1), "grid: hours asleep per day, optimal policy, example with spells")
    put("NExCascSPers", fmt(rb["cascade_hours"][sn.RUN_S], 1), "grid: hours of S per day, cascade, example with spells")
    put("NExMSpillPers", pct(rb["spill_M"], 0), "grid: spill of always-M, example with spells")
    guard(rb["cascade_hours"][sn.SLEEP] < 0.05 and rb["opt_hours"][sn.SLEEP] < 0.05 and
          grid[(exH, 0.5, exC)]["cascade_hours"][sn.SLEEP] < 0.05 and grid[(exH, 0.5, exC)]["opt_hours"][sn.SLEEP] < 0.05,
          "example: cascade and optimal policy are never idle")
    guard(rb["cascade_spill"] < 0.01 and rb["opt_spill"] < 0.01, "example with spells: cascade and optimum spill almost nothing")
    guard(rb["cascade_hours"][sn.RUN_S] > rb["cascade_hours"][sn.RUN_M] and rb["opt_hours"][sn.RUN_L] > 1.0,
          "example with spells: the adaptive policies run mostly S and several hours of L")
    idle_keys = [k for k in keys if grid[k]["opt_hours"][sn.SLEEP] > 1e-6]
    guard(all(k[0] == min(ex.HS) for k in idle_keys) and len(idle_keys) == len(ex.PS) * len(ex.CS),
          "the optimal policy sleeps exactly in the settings with the lowest sun")
    put("NOptSleepH", min(ex.HS), "grid")
    nsleep = sum(grid[k]["opt_hours"][sn.SLEEP] > 1e-6 for k in keys)
    put("NnOptSleeps", nsleep, "grid: settings in which the optimal policy sleeps or browns out at all")
    put("NOptSleepMaxH", fmt(max(grid[k]["opt_hours"][sn.SLEEP] for k in keys), 1), "grid: most hours asleep per day under the optimal policy")

    # ----- recorded irradiance -----
    S = sites["sites"]
    names = list(S)
    put("NSiteDays", f"{S[names[0]]['days']:,}".replace(",", "{,}"), "sites.json")
    put("NSiteShuffles", sites["n_shuffles"], "sites.json")
    cfs = [S[n]["fit_all"]["cloud_factor"] for n in names]
    ps = [S[n]["fit_all"]["p_stay"] for n in names]
    put("NSiteCfMin", fmt(min(cfs), 2), "sites: lower half / upper half of days"); put("NSiteCfMax", fmt(max(cfs), 2), "sites")
    put("NSitePMin", fmt(min(ps), 2), "sites: P(stay in the same half)"); put("NSitePMax", fmt(max(ps), 2), "sites")
    put("NSiteSpellMax", max(S[n]["cloudy_spell_max"] for n in names), "sites: longest run of below-median days")
    cases = [(n, r) for n in names for r in S[n]["settings"]]
    put("NSiteCases", len(cases), "sites")
    guard(len(cases) == 45, "45 (site, sun, battery) cases")
    loss = [(100 * (r["shuffled_M_mean"] - r["fixed_M"]), n, r) for n, r in cases]
    lmax_, ln, lr = max(loss, key=lambda t: t[0])
    put("NSiteOrderLossMax", fmt(lmax_, 1), "sites: always-M, shuffled mean minus recorded order")
    put("NSiteOrderLossMaxWhere", f"{S[ln]['label'].split(',')[0]}, $H={lr['H']}$~J, $C={lr['C']}$~J", "sites")
    lossS = [100 * (r["shuffled_S_mean"] - r["fixed_S"]) for n, r in cases]
    put("NSiteOrderLossSMax", fmt(max(lossS), 1), "sites: always-S, shuffled minus recorded")
    flip = sum(max("SML", key=lambda m_: r[f"fixed_{m_}"]) != max("SML", key=lambda m_: r[f"shuffled_{m_}_mean"]) for n, r in cases)
    put("NSiteFlip", flip, "sites: best fixed model, recorded vs shuffled order")
    put("NSiteSameBest", f"all {len(cases)}" if flip == 0 else f"{len(cases) - flip} of the {len(cases)}",
        "sites: cases with the same best fixed model in both orders")
    third = sn.Setting(H=1, p=0.5, C=1).cloud_factor / (0.5 * (1 + sn.Setting(H=1, p=0.5, C=1).cloud_factor))
    guard(abs(third - 1 / 3) < 1e-12, "the model's cloudy day harvests a third of the mean")
    below = [float((ex.load_site(n) < third * ex.load_site(n).mean()).mean()) for n in names]
    put("NSiteBelowThirdMax", fmt(100 * max(below) + 0.05, 1), "data: share of recorded days below a third of the site mean, largest over sites, rounded up")
    # is the loss beyond what shuffles alone produce? count cases where recorded M is below every shuffle
    below_all = sum(r["fixed_M"] < r["shuffled_M_min"] for n, r in cases)
    put("NSiteMBelowAll", below_all, "sites: cases where recorded-order always-M is below all shuffles")
    sp = [r["spill_S"] for n, r in cases]
    put("NSiteSpillSMin", pct(min(sp), 0), "sites"); put("NSiteSpillSMax", pct(max(sp), 0), "sites")
    tbf = lambda r: max(r["test_fixed_S"], r["test_fixed_M"], r["test_fixed_L"])
    cg = [100 * (r["cascade_test"] - tbf(r)) for n, r in cases]
    put("NSiteCascGainMedian", fmt(st.median(cg), 1), "sites: tuned cascade minus best fixed, test years")
    put("NSiteCascGainMin", fmt(min(cg), 1), "sites"); put("NSiteCascGainMax", fmt(max(cg), 1), "sites")
    put("NSiteCascBelowFixed", words(sum(x < 0 for x in cg)), "sites: cases where the tuned cascade is below the best fixed model")
    guard(sum(x < 0 for x in cg) == 1, "text says 'in one case' (singular)")
    put("NSiteCascWorstBelow", fmt(-min(cg), 2), "sites: by how much the tuned cascade is below the best fixed model in that case")
    share = [(r["cascade_test"] - tbf(r)) / (r["clairvoyant_test"] - tbf(r)) for n, r in cases]
    put("NSiteShareMedian", pct(st.median(share), 0), "sites: (cascade - best fixed)/(bound - best fixed), test years")
    fit_err = [100 * (r["fit_model_fixed"]["M"] - r["fixed_M"]) for n, r in cases]
    put("NSiteFitErrMax", fmt(max(fit_err), 1), "sites: fitted two-state model minus replay, always-M")
    guard(max(fit_err) > 1.0 and min(fit_err) > -0.5, "the fitted model overestimates always-M and hardly ever underestimates it")
    put("NSiteModelTunedBelowFixed", sum(r["cascade_modeltuned_test"] < tbf(r) for n, r in cases), "sites: model-tuned cascade below best fixed")
    cc = [100 * (r["clairvoyant_test"] - r["cascade_test"]) for n, r in cases]
    guard(min(cc) > -1e-6, "the clairvoyant bound is never below the cascade")
    put("NSiteCascToClairMedian", fmt(st.median(cc), 1), "sites: clairvoyant bound minus tuned cascade, test years")
    put("NSiteCascToClairMax", fmt(max(cc), 1), "sites")
    co = [100 * (r["cascade_test_oracle_thresholds"] - r["cascade_test"]) for n, r in cases]
    put("NSiteCascOracleMax", fmt(max(co) + 0.05, 1), "sites: thresholds tuned on the test years minus tuned on training, rounded up")
    ml = [100 * (r["cascade_test"] - r["cascade_modeltuned_test"]) for n, r in cases]
    put("NSiteModelTunedLossMedian", fmt(st.median(ml), 1), "sites: record-tuned minus model-tuned cascade, test years")
    put("NSiteModelTunedLossMax", fmt(max(ml), 1), "sites")
    put("NSiteModelTunedLossMin", fmt(min(ml), 1), "sites")
    mg = [100 * (r["mdp_fit_test"] - r["cascade_test"]) for n, r in cases]
    put("NSiteMdpMinusCascMedian", fmt(st.median(mg), 1), "sites: fitted-model optimal policy minus tuned cascade, test years")
    put("NSiteMdpMinusCascMin", fmt(min(mg), 1), "sites"); put("NSiteMdpMinusCascMax", fmt(max(mg), 1), "sites")
    # reference case for the table
    refH, refC = 6, 10
    put("NSiteRefH", refH, "table"); put("NSiteRefC", refC, "table")
    rows = []
    for n in names:
        s = S[n]
        r = [x for x in s["settings"] if x["H"] == refH and x["C"] == refC][0]
        rows.append(f"{s['label'].split(',')[0]} & {fmt(s['fit_all']['cloud_factor'], 2)} & {fmt(s['fit_all']['p_stay'], 2)} & "
                    f"{s['cloudy_spell_max']} & {pct(r['fixed_S'])} & {pct(r['fixed_M'])} & {pct(r['shuffled_M_mean'])} & "
                    f"{pct(tbf(r))} & {pct(r['cascade_test'])} & {pct(r['clairvoyant_test'])} \\\\")
    tab = ("\\begin{tabular}{@{}lccc|ccc|ccc@{}}\n\\toprule\n"
           " & \\multicolumn{3}{c|}{record} & \\multicolumn{3}{c|}{order} & \\multicolumn{3}{c}{test} \\\\\n"
           "site & $c$ & $p$ & run & S & M & M\\,sh. & fix. & casc. & bound \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(PAPER, "tab_sites.tex"), "w").write(tab)

    # ----- robustness -----
    put("NNightSharePct", fmt(100 * ex.NIGHT_SHARE, 0), "experiments.NIGHT_SHARE")
    nflip = [(best_fixed_name(grid[k]), best_fixed_name(night[k])) for k in keys]
    put("NNightFlip", sum(a_ != b_ for a_, b_ in nflip), "night vs grid: best fixed model")
    order = {"S": 0, "M": 1, "L": 2}
    put("NNightFlipSmaller", sum(order[b_] < order[a_] for a_, b_ in nflip), "night: changes towards a smaller model")
    put("NNightFlipLarger", sum(order[b_] > order[a_] for a_, b_ in nflip), "night: changes towards a larger model")
    guard(sum(order[b_] > order[a_] for a_, b_ in nflip) == 0, "night: every change of the best fixed model is towards a smaller model")
    cl, ch = 2, 10
    put("NNightCLo", cl, "robust"); put("NNightCHi", ch, "robust")
    bat_n = [100 * (night[(H, p, ch)]["opt"] - night[(H, p, cl)]["opt"]) for H in ex.HS for p in ex.PS]
    bat_u = [100 * (grid[(H, p, ch)]["opt"] - grid[(H, p, cl)]["opt"]) for H in ex.HS for p in ex.PS]
    put("NNightBatteryGain", fmt(st.median(bat_n), 1), "night: optimum at C=10 minus C=2, median over (H, p)")
    put("NUniformBatteryGain", fmt(st.median(bat_u), 1), "grid: the same with uniform events")
    guard(st.median(bat_n) > st.median(bat_u), "night-heavy events make the larger battery worth more (median over sun and persistence)")
    nshares = [(night[k]["cascade"] - best_fixed(night[k])) / (night[k]["opt"] - best_fixed(night[k])) for k in keys
               if 100 * (night[k]["opt"] - best_fixed(night[k])) > 0.5]
    put("NNightShareMedian", pct(st.median(nshares), 0), "night: share of the gap recovered by the retuned cascade")
    put("NNightShareN", len(nshares), "night: settings where the optimum gains more than half a point")
    guard(all(night[k]["cascade"] >= best_fixed(night[k]) - 1e-9 for k in keys), "night: the cascade is never worse than the best fixed model")
    for pc, tag in ((2, "Lo"), (5, "Mid"), (10, "Hi")):
        ls = [100 * (robust[k]["cascade"] - robust[k][f"noisy_{pc}"]) for k in keys]
        put(f"NNoise{tag}Pct", pc, "robust")
        put(f"NNoise{tag}Median", fmt(st.median(ls), 1), f"robust: loss with a reading error of +-{pc}% of capacity")
        put(f"NNoise{tag}Max", fmt(max(ls), 1), "robust")
        put(f"NNoise{tag}Min", fmt(min(ls), 1), "robust")
        put(f"NNoise{tag}Beats", sum(robust[k][f"noisy_{pc}"] > robust[k]["best_fixed"] for k in keys), "robust: still above the best fixed model")
        if pc == 5:
            guard(all(robust[k]["noisy_5"] > robust[k]["best_fixed"] for k in keys), "+-5% reading error: still above the best fixed model everywhere")
    put("NFadePct", 20, "experiments.cmd_robust")
    fl = [100 * (robust[k]["faded_retuned"] - robust[k]["faded_old_thresholds"]) for k in keys]
    guard(min(fl) > -1e-7, "retuning never hurts")
    put("NFadeLossMedian", fmt(st.median(fl), 1), "robust: retuned minus old thresholds at 80% capacity")
    put("NFadeLossMax", fmt(max(fl), 1), "robust")
    put("NFadeBeats", sum(robust[k]["faded_old_thresholds"] >= robust[k]["faded_best_fixed"] - 1e-9 for k in keys), "robust")
    guard(all(robust[k]["faded_old_thresholds"] >= robust[k]["faded_best_fixed"] - 1e-9 for k in keys),
          "aged battery, old thresholds: never below the best fixed model of the aged node")
    kn = max(keys, key=lambda k: robust[k]["cascade"] - robust[k]["noisy_5"])
    put("NNoiseWorstC", kn[2], "robust: battery of the worst case at +-5%")
    # table
    base_gain = [100 * (grid[k]["cascade"] - best_fixed(grid[k])) for k in keys]
    night_gain = [100 * (night[k]["cascade"] - best_fixed(night[k])) for k in keys]
    night_gap = [100 * (night[k]["opt"] - night[k]["cascade"]) for k in keys]
    def row(label, gain, gap):
        return f"{label} & {fmt(st.median(gain), 1)} & {fmt(min(gain), 1)} & {fmt(st.median(gap), 1)} & {fmt(max(gap), 1)} \\\\"
    rows = [row("uniform events (Section~\\ref{sec:results})", base_gain, cgap),
            row(f"{fmt(100 * ex.NIGHT_SHARE, 0)}\\% of events at night, retuned", night_gain, night_gap)]
    for pc in (2, 5, 10):
        g = [100 * (robust[k][f"noisy_{pc}"] - robust[k]["best_fixed"]) for k in keys]
        gp = [100 * (robust[k]["opt"] - robust[k][f"noisy_{pc}"]) for k in keys]
        rows.append(row(f"level read within $\\pm{pc}\\%$ of $C$", g, gp))
    tab = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{casc.\\ $-$ best fixed} & \\multicolumn{2}{c}{optimum $-$ casc.} \\\\\n"
           " & median & min & median & max \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    import re as _re
    tab = _re.sub(r"& -(\d)", r"& $-$\1", tab)
    open(os.path.join(PAPER, "tab_robust.tex"), "w").write(tab)

    # ----- other menus -----
    menus = json.load(open(os.path.join(RES, "menu.json")))
    guard(list(menus)[0] == "paper" and len(menus) == 7, "the paper's menu and six others")
    label = {"paper": "as in Table~\\ref{tab:energy}", "energy x2 per size": "energy $\\times2$ per size",
             "energy x8 per size": "energy $\\times8$ per size", "sleep floor halved": "sleep floor halved",
             "sleep floor doubled": "sleep floor doubled", "recalls 0.70/0.85/0.95": "recalls 0.70, 0.85, 0.95",
             "recalls 0.85/0.90/0.93": "recalls 0.85, 0.90, 0.93"}
    rows, stats = [], {}
    for name, mm in menus.items():
        R = {(r["H"], r["p"], r["C"]): r for r in mm["settings"]}
        guard(len(R) == len(ex.HS) * len(ex.PS) * len(ex.MENU_CS), f"menu '{name}': 48 settings")
        fl = sum(best_fixed_name(R[(H, 0.5, C)]) != best_fixed_name(R[(H, 0.95, C)]) for H in ex.HS for C in ex.MENU_CS)
        sp_ = [r["spill_S"] for r in R.values()]
        gn = [100 * (r["opt"] - best_fixed(r)) for r in R.values()]
        sh = [(r["cascade"] - best_fixed(r)) / (r["opt"] - best_fixed(r)) for r in R.values() if 100 * (r["opt"] - best_fixed(r)) > 0.5]
        stats[name] = {"flips": fl, "spill_min": min(sp_), "spill_max": max(sp_), "gain_median": st.median(gn),
                       "share_median": st.median(sh), "L_best": sum(best_fixed_name(r) == "L" for r in R.values()),
                       "S_best_everywhere": all(best_fixed_name(r) == "S" for r in R.values()),
                       "cascade_never_worse": all(r["cascade"] >= best_fixed(r) - 1e-9 for r in R.values())}
        rows.append(f"{label[name]} & {fl} & {pct(min(sp_), 0)}--{pct(max(sp_), 0)} & {fmt(st.median(gn), 1)} & {pct(st.median(sh), 0)} \\\\")
        if name == "paper":
            rows.append("\\addlinespace")
    tab = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n & & spill of & optimum & gap re- \\\\\n"
           "menu & changes & S (\\%) & $-$ fixed & covered (\\%) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(PAPER, "tab_menu.tex"), "w").write(tab)
    others = {k: v for k, v in stats.items() if k != "paper"}
    put("NMenuCMax", max(ex.MENU_CS), "experiments.MENU_CS")
    put("NMenuPairs", len(ex.HS) * len(ex.MENU_CS), "menu: (sun, battery) pairs")
    put("NMenuOthers", words(len(others)), "menu.json")
    with_flips = [v["flips"] for v in others.values() if v["flips"] > 0]
    put("NMenuFlipMenus", words(len(with_flips)), "menu: other menus under which persistence changes the best fixed model")
    put("NMenuFlipMin", min(with_flips), "menu"); put("NMenuFlipMax", max(with_flips), "menu")
    no_flip = [k for k, v in others.items() if v["flips"] == 0]
    guard(no_flip == ["energy x8 per size"] and others["energy x8 per size"]["S_best_everywhere"],
          "the only menu without a change is the eightfold one, where S is the best fixed model in every setting")
    put("NMenuSpillMin", pct(min(v["spill_min"] for v in others.values()), 0), "menu: spill of always-S, all other menus")
    put("NMenuSpillMax", pct(max(v["spill_max"] for v in others.values()), 0), "menu")
    put("NMenuShareMin", pct(min(v["share_median"] for v in others.values()), 0), "menu: median share of the gap recovered, smallest over menus")
    put("NMenuShareMax", pct(max(v["share_median"] for v in others.values()), 0), "menu")
    guard(all(v["cascade_never_worse"] for v in stats.values()), "under every menu the cascade is never below the best fixed model")
    lb = {k: v["L_best"] for k, v in others.items() if v["L_best"] > 0}
    guard(list(lb) == ["energy x2 per size"], "L is the best fixed model somewhere only under the twofold menu")
    put("NMenuLBest", lb["energy x2 per size"], "menu: settings in which L is the best fixed model, twofold menu")
    put("NMenuSettings", len(ex.HS) * len(ex.PS) * len(ex.MENU_CS), "menu: settings per menu")

    # ----- code location -----
    put("NCodeUrl", "\\url{" + CODE_URL + "}" if CODE_URL else "\\textbf{CODE-URL-MISSING}", "environment variable AIOT_CODE_URL")

    with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
        f.write("% generated by code/make_numbers.py; do not edit\n")
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(RES, "numbers_catalogue.tsv"), "w") as f:
        f.write("macro\tvalue\tsource\n")
        for row_ in CATALOGUE:
            f.write("\t".join(row_) + "\n")
    with open(os.path.join(RES, "claims_checked.txt"), "w") as f:
        for ok, what in CHECKS:
            f.write(("OK   " if ok else "FAIL ") + what + "\n")
    print(len(MACROS), "macros,", len(CHECKS), "claims checked, all hold")


if __name__ == "__main__":
    main()
