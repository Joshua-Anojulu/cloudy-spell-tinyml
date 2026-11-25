"""Solar-powered TinyML node: exact long-run detection rates.

Model of the paper "Size the Model to the Cloudy Spell, Not the Average Sun" (IEEE AIoT 2026, paper 244).

  * Time is in hours. The battery level is an integer number of units of Q = 6 mJ between 0 and C.
  * Weather is sunny or cloudy for a whole day and changes at midnight, staying the same with
    probability p. A sunny day harvests H joules along a half-sine from 06:00 to 18:00, a cloudy day
    c*H (c = 0.2 in the paper). The harvest of an hour is rounded to the nearest unit.
  * Every hour the node pays the sleep floor (10 uW for the hour = 36 mJ = 6 units). An hour in which
    a model runs costs, on top of the floor, 60 inferences: S 0.5 mJ, M 2 mJ, L 8 mJ each
    (30, 120 and 480 mJ per hour).
  * Harvest and load act in the same hour: a model runs if battery + harvest of the hour covers floor
    + inference. Otherwise the node browns out: it detects nothing and pays the floor only (the level
    cannot go below zero). Charge above C spills.
  * An event is detected if it falls in an hour in which a model runs and the model catches it
    (recall S 0.80, M 0.90, L 0.95). Events are uniform over the day unless event weights are given.

Policies:
  * fixed: always S, always M or always L;
  * battery cascade (weather-blind, two comparisons per hour): with e = level left after the floor is
    paid, run L if e >= k2*E_L, else M if e > k1*E_M, else S, where E_M and E_L are the inference
    energies of one hour of M and of L; k1 and k2 are whole hours (or 'never');
  * optimal: average-reward optimal policy over (level, weather, hour of day), by relative value
    iteration on the one-day operator.

The energies and recalls are illustrative; nothing in this file is a hardware measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q_J = 0.006                      # joules per battery unit
SLEEP_J_PER_H = 10e-6 * 3600.0   # 10 uW for one hour
INFER_PER_H = 60
MODELS = ("S", "M", "L")
E_INF_J = {"S": 0.5e-3, "M": 2.0e-3, "L": 8.0e-3}
RECALL = {"S": 0.80, "M": 0.90, "L": 0.95}
SLEEP, RUN_S, RUN_M, RUN_L = 0, 1, 2, 3
ACTIONS = (SLEEP, RUN_S, RUN_M, RUN_L)
ACTION_NAME = {SLEEP: "sleep", RUN_S: "S", RUN_M: "M", RUN_L: "L"}
SUN, CLOUD = 0, 1
UNIFORM_EVENTS = tuple([1.0 / 24.0] * 24)


def to_units(joules: float) -> int:
    """Joules to battery units, rounded to the nearest unit."""
    return int(round(joules / Q_J))


@dataclass(frozen=True)
class Menu:
    """Energies (battery units per hour) and recalls of the sleep floor and the three models."""
    sleep_u: int = to_units(SLEEP_J_PER_H)
    infer_u: tuple = (0, to_units(INFER_PER_H * E_INF_J["S"]), to_units(INFER_PER_H * E_INF_J["M"]),
                      to_units(INFER_PER_H * E_INF_J["L"]))
    recall: tuple = (0.0, RECALL["S"], RECALL["M"], RECALL["L"])

    @property
    def cost_u(self) -> np.ndarray:
        """Total cost of an hour for each action: floor plus inference."""
        return np.array([self.sleep_u + e for e in self.infer_u], dtype=np.int64)

    @property
    def reward(self) -> np.ndarray:
        return np.array(self.recall)


PAPER_MENU = Menu()
# module-level shortcuts for the paper's menu
SLEEP_U = PAPER_MENU.sleep_u
COST_U = PAPER_MENU.cost_u
REWARD = PAPER_MENU.reward
E_M_U = PAPER_MENU.infer_u[RUN_M]
E_L_U = PAPER_MENU.infer_u[RUN_L]


def half_sine_fractions() -> np.ndarray:
    """Share of a day's harvest that falls in each of the 12 hours from 06:00 to 18:00."""
    k = np.arange(12)
    return (np.cos(np.pi * k / 12.0) - np.cos(np.pi * (k + 1) / 12.0)) / 2.0


def harvest_profile_units(day_joules) -> np.ndarray:
    """Hourly harvests in units for days that harvest day_joules along the half-sine.

    A scalar gives an array [24]; an array [...] of day totals gives [..., 24]."""
    dj = np.asarray(day_joules, dtype=float)
    prof = np.zeros(dj.shape + (24,), dtype=np.int64)
    prof[..., 6:18] = np.rint(dj[..., None] * half_sine_fractions() / Q_J).astype(np.int64)
    return prof


@dataclass(frozen=True)
class Setting:
    H: float                     # harvest of a sunny day, joules
    p: float                     # probability that the weather stays the same at midnight
    C: float                     # battery capacity, joules
    cloud_factor: float = 0.2    # harvest of a cloudy day as a share of a sunny day
    event_weights: tuple = UNIFORM_EVENTS   # share of the day's events in each hour
    menu: Menu = PAPER_MENU

    @property
    def cap(self) -> int:
        return to_units(self.C)

    def profiles(self) -> np.ndarray:
        """Array [weather, hour] of hourly harvest in units."""
        return np.stack([harvest_profile_units(self.H), harvest_profile_units(self.cloud_factor * self.H)])

    def weather_matrix(self) -> np.ndarray:
        return np.array([[self.p, 1.0 - self.p], [1.0 - self.p, self.p]])

    @property
    def mean_harvest_j(self) -> float:
        return 0.5 * (1.0 + self.cloud_factor) * self.H


def step(b, g, action, cap: int, menu: Menu = PAPER_MENU):
    """One hour for an array of battery levels b with harvest g (scalar or array) and actions.

    Returns (new level, ran, spilled units, units paid). A model that level plus harvest cannot cover
    does not run: the node browns out and pays the sleep floor only."""
    cost = menu.cost_u[action]
    avail = b + g
    ran = (action != SLEEP) & (avail >= cost)
    pay = np.where(ran, cost, menu.sleep_u)
    pay = np.minimum(pay, avail)             # the level cannot go below zero
    after = avail - pay
    spilled = np.maximum(after - cap, 0)
    return np.minimum(after, cap), ran, spilled, pay


# ---------------------------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------------------------

def fixed_policy(model: str):
    act = {"S": RUN_S, "M": RUN_M, "L": RUN_L}[model]

    def policy(b, w, h):
        return np.full(np.shape(b), act, dtype=np.int64)
    return policy


NEVER = None


def cascade_thresholds(k1, k2, cap: int, menu: Menu = PAPER_MENU):
    """Integer thresholds (u1, u2) on e = level - floor: M if e >= u1, L if e >= u2.

    The rule of the paper is M if e > k1*E_M, i.e. e >= k1*E_M + 1, and L if e >= k2*E_L."""
    never = cap + menu.sleep_u + 1
    u1 = never if k1 is NEVER else k1 * menu.infer_u[RUN_M] + 1
    u2 = never if k2 is NEVER else k2 * menu.infer_u[RUN_L]
    return u1, u2


def cascade_policy_units(u1: int, u2: int, menu: Menu = PAPER_MENU):
    def policy(b, w, h):
        e = b - menu.sleep_u
        return np.where(e >= u2, RUN_L, np.where(e >= u1, RUN_M, RUN_S)).astype(np.int64)
    return policy


def cascade_policy(k1, k2, cap: int, menu: Menu = PAPER_MENU):
    return cascade_policy_units(*cascade_thresholds(k1, k2, cap, menu), menu=menu)


def table_policy(act: np.ndarray):
    """Policy from an action table [weather, hour, level]."""
    def policy(b, w, h):
        return act[w, h, b]
    return policy


# ---------------------------------------------------------------------------------------------
# exact evaluation of a deterministic policy on the two-state weather model
# ---------------------------------------------------------------------------------------------

def day_tables(setting: Setting, policy):
    """Run one day from every start level, for each weather, under a state-feedback policy.

    Per weather: dict with end level, detected share of the day's events, spilled units, harvested
    units (scalar), hours per action [n, 4] (column 0: asleep or browned out), model switches inside
    the day, first and last action of the day."""
    cap = setting.cap
    menu = setting.menu
    prof = setting.profiles()
    wts = np.asarray(setting.event_weights)
    out = []
    for w in (SUN, CLOUD):
        b = np.arange(cap + 1, dtype=np.int64)
        det = np.zeros(cap + 1)
        spill = np.zeros(cap + 1)
        used = np.zeros((cap + 1, 4))
        switches = np.zeros(cap + 1)
        prev = first = None
        for h in range(24):
            a = policy(b, w, h)
            b, ran, sp, _ = step(b, int(prof[w, h]), a, cap, menu)
            det += np.where(ran, menu.reward[a], 0.0) * wts[h]
            spill += sp
            for act in (RUN_S, RUN_M, RUN_L):
                used[:, act] += ran & (a == act)
            used[:, SLEEP] += ~ran
            if prev is not None:
                switches += (a != prev)
            else:
                first = a.copy()
            prev = a
        out.append({"end": b, "det": det, "spill": spill, "harvest": float(prof[w].sum()), "hours": used,
                    "switches": switches, "first": first, "last": prev})
    return out


def stationary_midnight(setting: Setting, end_level, tol: float = 1e-12, max_iter: int = 2000000) -> np.ndarray:
    """Stationary law over (weather of the coming day, level at midnight), array [2, n].

    Power iteration of the lazy chain (half a step per iteration), which has the same stationary law
    and cannot oscillate; started from a full battery. The chain has one recurrent class: a long
    sunny or a long cloudy spell leads every level to the same boundary."""
    n = setting.cap + 1
    W = setting.weather_matrix()
    pi = np.zeros((2, n))
    pi[:, n - 1] = 0.5
    for _ in range(max_iter):
        moved = np.stack([np.bincount(end_level[w], weights=pi[w], minlength=n) for w in (SUN, CLOUD)])
        new = 0.5 * pi + 0.5 * (W.T @ moved)
        if np.abs(new - pi).sum() < tol:
            return new / new.sum()
        pi = new
    raise RuntimeError("stationary law did not converge")


def stationary_midnight_direct(setting: Setting, end_level) -> np.ndarray:
    """The same law by a sparse linear solve (used as a check of the power iteration)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = setting.cap + 1
    W = setting.weather_matrix()
    rows, cols, vals = [], [], []
    for w in (SUN, CLOUD):
        src_idx = w * n + np.arange(n)
        for wn in (SUN, CLOUD):
            rows.append(src_idx)
            cols.append(wn * n + end_level[w])
            vals.append(np.full(n, W[w, wn]))
    T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    A = (sp.identity(2 * n, format="csr") - T).T.tolil()
    A[2 * n - 1, :] = 1.0
    rhs = np.zeros(2 * n)
    rhs[2 * n - 1] = 1.0
    pi = spla.spsolve(A.tocsc(), rhs)
    pi = np.clip(pi, 0.0, None)
    return (pi / pi.sum()).reshape(2, n)


def evaluate(setting: Setting, policy, full: bool = True) -> dict:
    """Exact long-run performance of a deterministic state-feedback policy."""
    tabs = day_tables(setting, policy)
    pi = stationary_midnight(setting, [t["end"] for t in tabs])
    det = sum(float(pi[w] @ tabs[w]["det"]) for w in (SUN, CLOUD))
    if not full:
        return {"rate": det}
    spill = sum(float(pi[w] @ tabs[w]["spill"]) for w in (SUN, CLOUD))
    harvest = sum(float(pi[w].sum()) * tabs[w]["harvest"] for w in (SUN, CLOUD))
    hours = sum(pi[w] @ tabs[w]["hours"] for w in (SUN, CLOUD))
    switches = sum(float(pi[w] @ tabs[w]["switches"]) for w in (SUN, CLOUD))
    return {"rate": det, "spill_frac": spill / harvest if harvest > 0 else 0.0, "hours": hours,
            "switches_in_day": switches, "pi": pi, "tabs": tabs}


def rate_only(setting: Setting, policy) -> float:
    """Exact detected share of events, without the other statistics (used inside searches)."""
    cap = setting.cap
    menu = setting.menu
    prof = setting.profiles()
    wts = np.asarray(setting.event_weights)
    ends, dets = [], []
    for w in (SUN, CLOUD):
        b = np.arange(cap + 1, dtype=np.int64)
        det = np.zeros(cap + 1)
        for h in range(24):
            a = policy(b, w, h)
            b, ran, _, _ = step(b, int(prof[w, h]), a, cap, menu)
            det += np.where(ran, menu.reward[a], 0.0) * wts[h]
        ends.append(b)
        dets.append(det)
    pi = stationary_midnight(setting, ends)
    return sum(float(pi[w] @ dets[w]) for w in (SUN, CLOUD))


# ---------------------------------------------------------------------------------------------
# cascade search, optimal policy
# ---------------------------------------------------------------------------------------------

def cascade_grid(cap: int, menu: Menu = PAPER_MENU):
    """All (k1, k2) with whole hours that the battery can hold, plus 'never', with u2 >= u1."""
    k1s = list(range(0, cap // menu.infer_u[RUN_M] + 1)) + [NEVER]
    k2s = list(range(0, cap // menu.infer_u[RUN_L] + 1)) + [NEVER]
    for k1 in k1s:
        for k2 in k2s:
            u1, u2 = cascade_thresholds(k1, k2, cap, menu)
            if u2 >= u1:
                yield k1, k2, u1, u2


def best_cascade(setting: Setting, score=None):
    """Grid search of the cascade thresholds. score(policy) defaults to the exact detection rate."""
    if score is None:
        def score(pol):
            return rate_only(setting, pol)
    top = (-1.0, NEVER, NEVER)
    n = 0
    for k1, k2, u1, u2 in cascade_grid(setting.cap, setting.menu):
        n += 1
        r = score(cascade_policy_units(u1, u2, setting.menu))
        if r > top[0] + 1e-13:
            top = (r, k1, k2)
    return {"rate": top[0], "k1": top[1], "k2": top[2], "pairs": n}


def optimal_policy(setting: Setting, tol: float = 1e-7, damping: float = 0.5, max_iter: int = 200000):
    """Average-reward optimal policy over (level, weather, hour) by damped relative value iteration on
    the one-day operator. Returns (lower and upper bound of the gain per day, action table [w, h, level])."""
    cap = setting.cap
    menu = setting.menu
    n = cap + 1
    prof = setting.profiles()
    W = setting.weather_matrix()
    wts = np.asarray(setting.event_weights)
    levels = np.arange(n, dtype=np.int64)
    nxt = np.zeros((2, 24, 4, n), dtype=np.int64)
    rew = np.zeros((2, 24, 4, n))
    for w in (SUN, CLOUD):
        for h in range(24):
            for a in ACTIONS:
                nb, ran, _, _ = step(levels, int(prof[w, h]), np.full(n, a, dtype=np.int64), cap, menu)
                nxt[w, h, a] = nb
                if a == SLEEP:
                    rew[w, h, a] = 0.0
                else:
                    # a model that the hour cannot pay for is not offered (it would only brown out)
                    rew[w, h, a] = np.where(ran, menu.reward[a] * wts[h], -1.0)
    v = np.zeros((2, n))                    # value at midnight, by weather of the coming day
    act = np.zeros((2, 24, n), dtype=np.int64)
    lo = hi = 0.0
    for it in range(max_iter):
        cont = W @ v                        # expected value at the next midnight, given today's weather
        new = np.zeros((2, n))
        for w in (SUN, CLOUD):
            u = cont[w].copy()
            for h in range(23, -1, -1):
                q = rew[w, h] + u[nxt[w, h]]          # [action, level]
                act[w, h] = np.argmax(q, axis=0)
                u = np.max(q, axis=0)
            new[w] = u
        diff = new - v
        lo, hi = float(diff.min()), float(diff.max())
        if hi - lo < tol:
            break
        new = new - new[0, n - 1]            # relative values
        v = damping * v + (1.0 - damping) * new if it > 0 else new
    else:
        raise RuntimeError("relative value iteration did not converge")
    return lo, hi, act.copy()


# ---------------------------------------------------------------------------------------------
# stochastic policies (battery level read with an error), exact
# ---------------------------------------------------------------------------------------------

def evaluate_noisy_cascade(setting: Setting, u1: int, u2: int, err_units: int, tol: float = 1e-11,
                           max_days: int = 200000) -> dict:
    """Cascade whose level reading is off by an error uniform on {-err_units, ..., +err_units} units,
    drawn independently every hour. Exact: the law over levels is pushed hour by hour."""
    cap = setting.cap
    menu = setting.menu
    n = cap + 1
    prof = setting.profiles()
    W = setting.weather_matrix()
    wts = np.asarray(setting.event_weights)
    levels = np.arange(n, dtype=np.int64)
    e = levels - menu.sleep_u
    width = 2 * err_units + 1

    def p_geq(u):                            # P(reading >= u) for a reading e + d, d uniform on -err..err
        return np.clip((e + err_units - u + 1) / width, 0.0, 1.0)
    pL = p_geq(u2)
    pM = np.clip(p_geq(u1) - pL, 0.0, 1.0)
    pS = 1.0 - pL - pM
    probs = {RUN_S: pS, RUN_M: pM, RUN_L: pL}
    nxt = {}
    ran = {}
    for w in (SUN, CLOUD):
        for h in range(24):
            for a in (RUN_S, RUN_M, RUN_L):
                nb, rn, _, _ = step(levels, int(prof[w, h]), np.full(n, a, dtype=np.int64), cap, menu)
                nxt[w, h, a] = nb
                ran[w, h, a] = rn

    def one_day(pi):
        det = 0.0
        out = np.zeros((2, n))
        for w in (SUN, CLOUD):
            d = pi[w]
            for h in range(24):
                new = np.zeros(n)
                for a in (RUN_S, RUN_M, RUN_L):
                    m = d * probs[a]
                    det += float((m * ran[w, h, a]).sum()) * menu.reward[a] * wts[h]
                    new += np.bincount(nxt[w, h, a], weights=m, minlength=n)
                d = new
            out[w] = d
        return W.T @ out, det

    pi = np.zeros((2, n))
    pi[:, n - 1] = 0.5
    for _ in range(max_days):
        new, _ = one_day(pi)
        new = 0.5 * pi + 0.5 * new
        if np.abs(new - pi).sum() < tol:
            pi = new / new.sum()
            _, det = one_day(pi)
            return {"rate": det}
        pi = new
    raise RuntimeError("noisy cascade did not converge")


# ---------------------------------------------------------------------------------------------
# replay of a recorded sequence of days
# ---------------------------------------------------------------------------------------------

def replay(day_joules, policy, cap: int, start_level, weather=None, menu: Menu = PAPER_MENU,
           event_weights=UNIFORM_EVENTS):
    """Run a policy through recorded days.

    day_joules: array [..., days] of daily harvest totals in joules (leading axes are independent
    replicas: shuffled orders, or one replica per candidate policy). weather: optional array
    [..., days] of the day's class (SUN or CLOUD) handed to the policy.
    Returns per-day arrays [..., days]: detected share of the day's events, spilled units, harvested
    units; and the final level."""
    dj = np.asarray(day_joules, dtype=float)
    prof = harvest_profile_units(dj)                  # [..., days, 24]
    days = dj.shape[-1]
    wts = np.asarray(event_weights)
    b = np.broadcast_to(np.asarray(start_level, dtype=np.int64), np.broadcast_shapes(dj.shape[:-1], np.shape(start_level))).copy()
    lead = b.shape
    det = np.zeros(lead + (days,))
    spilled = np.zeros(lead + (days,))
    harvested = np.zeros(lead + (days,))
    for d in range(days):
        w = None if weather is None else weather[..., d]
        for h in range(24):
            a = policy(b, w, h)
            g = prof[..., d, h]
            b, ran, sp, _ = step(b, g, a, cap, menu)
            det[..., d] += np.where(ran, menu.reward[a], 0.0) * wts[h]
            spilled[..., d] += sp
            harvested[..., d] += g
    return {"daily": det, "spilled": spilled, "harvested": harvested, "end_level": b}


def clairvoyant(day_joules, cap: int, start_level: int, menu: Menu = PAPER_MENU, event_weights=UNIFORM_EVENTS):
    """Best detected share of events for a recorded sequence of days when the whole future is known
    (dynamic programme backwards over the record). An upper bound for every policy on that record."""
    dj = np.asarray(day_joules, dtype=float)
    prof = harvest_profile_units(dj)                  # [days, 24]
    days = dj.shape[0]
    n = cap + 1
    wts = np.asarray(event_weights)
    levels = np.arange(n, dtype=np.int64)
    v = np.zeros(n)
    for d in range(days - 1, -1, -1):
        for h in range(23, -1, -1):
            g = int(prof[d, h])
            best = None
            for a in ACTIONS:
                nb, ran, _, _ = step(levels, g, np.full(n, a, dtype=np.int64), cap, menu)
                q = v[nb] + np.where(ran, menu.reward[a] * wts[h], 0.0)
                if a != SLEEP:
                    q = np.where(ran, q, -np.inf)     # a model that cannot be paid is not offered
                best = q if best is None else np.maximum(best, q)
            v = best
    return float(v[start_level]) / days


# ---------------------------------------------------------------------------------------------
# Monte Carlo check of the exact rates
# ---------------------------------------------------------------------------------------------

def simulate(setting: Setting, policy, days: int, replicas: int, rng: np.random.Generator, burn_in: int = 400):
    """Independent replicas of the node under the two-state weather; returns per-replica detected shares
    and the largest violation of the energy balance (start + harvest - paid - spilled - end, must be 0)."""
    cap = setting.cap
    menu = setting.menu
    prof = setting.profiles()
    wts = np.asarray(setting.event_weights)
    b = np.full(replicas, cap, dtype=np.int64)
    w = (rng.random(replicas) < 0.5).astype(np.int64)
    det = np.zeros(replicas)
    harvested = np.zeros(replicas, dtype=np.int64)
    paid = np.zeros(replicas, dtype=np.int64)
    spilled = np.zeros(replicas, dtype=np.int64)
    start = None
    for d in range(-burn_in, days):
        if d == 0:
            start = b.copy()
        for h in range(24):
            a = policy(b, w, h)
            g = prof[w, h]
            b, ran, sp, pay = step(b, g, a, cap, menu)
            if d >= 0:
                det += np.where(ran, menu.reward[a], 0.0) * wts[h]
                harvested += g
                paid += pay
                spilled += sp
        flip = rng.random(replicas) >= setting.p
        w = np.where(flip, 1 - w, w)
    balance = start + harvested - paid - spilled - b
    return det / days, int(np.abs(balance).max())
