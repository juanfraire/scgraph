"""BRUF and RUCoP on the slotted Bernoulli projection (revision, T1).

The uncertainty-aware DTN line computes optimal *contingent* routing policies
on a slotted uncertain contact plan: each (contact, slot) pair succeeds
independently with a known probability, a failed transmission leaves the copy
at the sender (acknowledged communication), and the policy acts again in the
next slot. BRUF [Raverta et al., WiSEE 2018] does this for one copy; RUCoP
[Raverta et al., Ad Hoc Networks 2021] for N copies, with the MDP state being
the distribution of copies over nodes at a slot and the objective
P[at least one copy at the destination by the last slot]. Both are solved by
backward induction, since time only moves forward and the MDP is acyclic.

This module reimplements both on the projection q_c[k] of eq. (14)
(`routing/bernoulli.py`), so a stochastic contact graph can be handed to them
exactly as the paper hands it to the static Bernoulli router:

  project(...)          the slotted plan {cid -> q[k]} plus slot bookkeeping
  solve(model, N)       exact backward induction, states pruned by forward
                        reachability from the source state; returns the policy
                        table, the model's predicted delivery probability, the
                        number of (state, slot) pairs, transitions and runtime
  bruf(model)           N = 1 with a complete (node, slot) table
  rucop(model, N)       N copies, joint actions over all copy holders
  execute_policy(...)   online execution of a policy on ONE realization of the
                        full stochastic contact graph

Modelling choices, each matching the published algorithms:

  * One hop per slot. A copy sent in slot k is at the receiver in slot k+1.
  * Per-slot independence. q_c[k] and q_c[k+1] are treated as independent
    trials, which is how a contact spanning several slots enters the BRUF and
    RUCoP plans (one uncertain contact per slot).
  * Joint actions. In a slot, every node holding copies splits them among
    `stay` and its usable contacts; contacts succeed independently, and the
    copies sent over one contact move or stay together.
  * Contacts towards nodes that can no longer reach the destination are left
    out of the action set (the rc_src / rc_dest pruning of the LSS/RUCoP
    tooling). This never changes the optimal value.
  * Ties on the delivery probability are broken by the earliest expected
    delivery slot, a lexicographic objective that backward induction solves
    exactly. Remaining ties keep the first action in a fixed order, with
    `wait` first.

`joint=False` restricts the action space to a single transmission per slot
across the whole network. That is the action space of the dependency-free toy
in the uncertain-DTN artefact (examples/toy_rucop.py), and it is kept only so
the test suite can reproduce the toy's numbers exactly.

Execution (`execute_policy`) follows the policy table on a realized plan of
`DetContact`s. A copy at node u at continuous time a sits in slot
k = floor((a - t0) / Delta). When the policy assigns it a contact, the copy
takes it if the contact exists in the realization and can carry the bundle
starting at max(a, t_s) and finishing by t_e (the `_hop` rule of
routing/placement.py). On a failure the copy waits for the boundary after the
attempt's slot and the policy is queried again. A copy that lands at a node
mid-slot is offered a decision straight away, using the slot it landed in.
By default (mode "asap") a `wait` entry does not hold the copy at a slot
boundary: the copy is queued on the next contact the policy plans for it and
leaves as soon as that contact can carry it, the same continuous-time rule
the static routes are executed with. Mode "slot" holds it to the grid instead.
Copies of one bundle share the realization, and the joint state the policy
sees is the global copy distribution (RUCoP's global-information policy).
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from scgraph.arrival import delivery_time
from scgraph.contact import DetContact, StochasticContact
from scgraph.graph import Realization, StochasticContactGraph
from scgraph.routing.bernoulli import bernoulli_projection

GOAL: tuple[int, ...] = (-1,)      # any state with a copy at the destination
TOL = 1e-12                        # value tie tolerance


# --------------------------------------------------------------------------
# the slotted uncertain contact plan
# --------------------------------------------------------------------------


@dataclass
class SlotModel:
    """A slotted uncertain contact plan: q[cid][k] = P[contact cid carries a
    bundle offered at slot boundary k], with contact endpoints."""

    src: int
    dst: int
    n_slots: int
    q: dict[int, np.ndarray]
    ends: dict[int, tuple[int, int]]          # cid -> (frm, to)
    t0: float = 0.0
    slot_width: float = 1.0
    project_seconds: float = 0.0

    def usable(self, k: int) -> list[tuple[int, int, int, float]]:
        """(cid, frm, to, q) for every contact with q > 0 in slot k."""
        out = []
        for cid, qc in self.q.items():
            p = float(qc[k])
            if p > 0.0:
                u, v = self.ends[cid]
                if u != v:
                    out.append((cid, u, v, min(1.0, p)))
        out.sort()
        return out

    @property
    def nodes(self) -> set[int]:
        ns = {self.src, self.dst}
        for u, v in self.ends.values():
            ns.add(u)
            ns.add(v)
        return ns

    @classmethod
    def from_table(cls, contacts, n_slots: int, src: int, dst: int,
                   t0: float = 0.0, slot_width: float = 1.0) -> "SlotModel":
        """Build from explicit (cid, frm, to, slot, p) rows (one slot each)."""
        q: dict[int, np.ndarray] = {}
        ends: dict[int, tuple[int, int]] = {}
        for cid, u, v, k, p in contacts:
            if cid not in q:
                q[cid] = np.zeros(n_slots)
                ends[cid] = (u, v)
            q[cid][k] = p
        return cls(src, dst, n_slots, q, ends, t0, slot_width)


def n_slots_for(t0: float, deadline: float, slot_width: float) -> int:
    """The grid `bernoulli_route` uses: ceil((D - t0) / Delta) slots."""
    return max(1, int(math.ceil((deadline - t0) / slot_width)))


def project(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    slot_width: float,
    rng: np.random.Generator,
    m: int = 4000,
    window: bool = True,
) -> SlotModel:
    """Eq. (14) projection of `graph` onto the slot grid anchored at t0.

    With `window` only contacts whose support overlaps [t0, D] are projected;
    every other contact has q = 0 on the whole grid anyway. A latent state, if
    any, enters through its marginal (bernoulli.py), so the model sees the
    independent marginals and none of the correlation.
    """
    t_start = time.perf_counter()
    g = graph.window(t0, deadline) if window else graph
    K = n_slots_for(t0, deadline, slot_width)
    q = bernoulli_projection(g, K, slot_width, size, m, rng, t0=t0)
    ends = {c.cid: (c.frm, c.to) for c in g.contacts}
    return SlotModel(src, dst, K, q, ends, t0, slot_width,
                     project_seconds=time.perf_counter() - t_start)


# --------------------------------------------------------------------------
# backward induction
# --------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised when a solve passes its transition or time budget."""

    def __init__(self, msg, stats):
        super().__init__(msg)
        self.stats = stats


@dataclass
class Policy:
    """An optimal BRUF / RUCoP policy and what it cost to compute."""

    n_copies: int
    src: int
    dst: int
    t0: float
    slot_width: float
    n_slots: int
    ends: dict[int, tuple[int, int]]
    table: list[dict] = field(default_factory=list)   # per slot: state -> action
    value: float = 0.0          # the model's P[some copy at dst by the last slot]
    exp_slot: float = math.inf  # the model's E[delivery slot | delivered]
    n_states: int = 0           # (state, slot) pairs evaluated
    n_transitions: int = 0      # (state, action, outcome) triples evaluated
    n_actions: int = 0          # (state, action) pairs evaluated
    seconds: float = 0.0
    complete: bool = False
    joint: bool = True

    def action(self, k: int, state: tuple[int, ...]):
        """Action stored for (slot k, state), or None if the pair was pruned."""
        if not self.table or k < 0 or k >= len(self.table):
            return None
        return self.table[k].get(state)


def _canon(nodes, dst: int) -> tuple[int, ...]:
    if dst in nodes:
        return GOAL
    return tuple(sorted(nodes))


def _node_allocations(n: int, contacts: list[tuple[int, int, float]], joint: bool):
    """Every way to send up to n copies over `contacts` from one node.

    Yields (key, used) where used = [(cid, to, q, count), ...] with count > 0
    and key = ((cid, count), ...). The empty allocation (`wait`) comes first.
    With joint=False at most one contact is used.
    """
    m = len(contacts)
    yield (), []
    if m == 0 or n == 0:
        return
    if not joint:
        for i, (cid, v, p) in enumerate(contacts):
            for x in range(1, n + 1):
                yield ((cid, x),), [(cid, v, p, x)]
        return

    def rec(i, left):
        if i == m:
            yield []
            return
        for x in range(0, left + 1):
            for rest in rec(i + 1, left - x):
                yield [x] + rest

    for xs in rec(0, n):
        if sum(xs) == 0:
            continue
        used = [(contacts[i][0], contacts[i][1], contacts[i][2], xs[i])
                for i in range(m) if xs[i] > 0]
        key = tuple((cid, x) for cid, _, _, x in used)
        yield key, used


def _allocation_outcomes(u: int, n: int, used):
    """Outcome distribution of one node's allocation: [(copy nodes, prob)].

    Each used contact succeeds independently; on success its copies move to
    the receiver, on failure they stay at u (acknowledged communication).
    """
    if not used:
        return [((u,) * n, 1.0)]
    out: dict[tuple[int, ...], float] = {}
    sent = sum(x for _, _, _, x in used)
    for mask in itertools.product((True, False), repeat=len(used)):
        p = 1.0
        nodes = [u] * (n - sent)
        for ok, (_cid, v, q, x) in zip(mask, used):
            if ok:
                p *= q
                nodes.extend([v] * x)
            else:
                p *= 1.0 - q
                nodes.extend([u] * x)
        if p <= 0.0:
            continue
        key = tuple(sorted(nodes))
        out[key] = out.get(key, 0.0) + p
    return list(out.items())


def solve(
    model: SlotModel,
    n_copies: int = 1,
    *,
    joint: bool = True,
    complete: bool = False,
    keep_table: bool = True,
    max_transitions: int | None = None,
    max_seconds: float | None = None,
    max_states: int | None = None,
) -> Policy:
    """Optimal policy by backward induction over (copy distribution, slot).

    `complete=True` (single copy only) evaluates every (node, slot) pair
    instead of the forward-reachable ones, so the table answers any query the
    online executor can make. Budgets raise `BudgetExceeded` carrying the
    partial counts, which the tractability study records as censored;
    `max_states` caps the forward pass, which is where memory goes.
    """
    if complete and n_copies != 1:
        raise ValueError("complete tables are only built for a single copy")
    t_start = time.perf_counter()
    K, src, dst = model.n_slots, model.src, model.dst
    usable = [model.usable(k) for k in range(K)]

    # nodes that can still reach dst from slot k (rc_dest), and nodes a copy
    # can occupy at slot k (rc_src)
    R: list[set[int]] = [set() for _ in range(K + 1)]
    R[K] = {dst}
    for k in reversed(range(K)):
        r = set(R[k + 1])
        for _cid, u, v, _p in usable[k]:
            if v in R[k + 1]:
                r.add(u)
        R[k] = r
    S: list[set[int]] = [set() for _ in range(K + 1)]
    S[0] = {src}
    for k in range(K):
        s = set(S[k])
        for _cid, u, v, _p in usable[k]:
            if u in S[k]:
                s.add(v)
        S[k + 1] = s

    # useful contacts per slot and sender
    out: list[dict[int, list[tuple[int, int, float]]]] = []
    for k in range(K):
        d: dict[int, list[tuple[int, int, float]]] = {}
        for cid, u, v, p in usable[k]:
            if u == dst or v not in R[k + 1]:
                continue
            if not complete and u not in S[k]:
                continue
            d.setdefault(u, []).append((cid, v, p))
        out.append(d)

    init = _canon((src,) * n_copies, dst)
    stats = {"n_states": 0, "n_transitions": 0, "n_actions": 0}

    def check_budget():
        if max_transitions is not None and stats["n_transitions"] > max_transitions:
            raise BudgetExceeded("transition budget", {**stats, "seconds": time.perf_counter() - t_start})
        if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
            raise BudgetExceeded("time budget", {**stats, "seconds": time.perf_counter() - t_start})

    # ---- forward reachability over copy distributions ----
    if complete:
        nodes = sorted(model.nodes - {dst})
        states = [{(u,) for u in nodes} for _ in range(K + 1)]
        for st in states:
            st.add(GOAL)
    else:
        states = [set() for _ in range(K + 1)]
        states[0].add(init)
        for k in range(K):
            nxt = states[k + 1]
            for s in states[k]:
                if s == GOAL:
                    nxt.add(GOAL)
                    continue
                per_node = []
                for u, n_u in sorted(Counter(s).items()):
                    targets = sorted({u} | {v for _c, v, _p in out[k].get(u, ())})
                    per_node.append(list(itertools.combinations_with_replacement(targets, n_u)))
                for combo in itertools.product(*per_node):
                    merged = [x for part in combo for x in part]
                    nxt.add(_canon(merged, dst))
            over_t = max_seconds is not None and time.perf_counter() - t_start > max_seconds
            n_fwd = sum(len(x) for x in states[: k + 2])
            if over_t or (max_states is not None and n_fwd > max_states):
                raise BudgetExceeded("budget (forward pass)", {
                    "n_states": n_fwd, "n_transitions": 0,
                    "n_actions": 0, "seconds": time.perf_counter() - t_start})

    # ---- backward induction (lexicographic: max V, then min E[slot; delivered]) ----
    V_next = {s: (1.0 if s == GOAL else 0.0) for s in states[K]}
    W_next = {s: 0.0 for s in states[K]}
    table: list[dict] = [dict() for _ in range(K)] if keep_table else []

    for k in reversed(range(K)):
        alloc_cache: dict = {}
        V_cur: dict = {}
        W_cur: dict = {}
        tab_k = table[k] if keep_table else None
        out_k = out[k]
        for s in states[k]:
            stats["n_states"] += 1
            if s == GOAL:
                V_cur[s], W_cur[s] = 1.0, 0.0
                continue
            groups = sorted(Counter(s).items())
            per_node = []
            for u, n_u in groups:
                key = (u, n_u)
                opts = alloc_cache.get(key)
                if opts is None:
                    opts = [
                        (akey, _allocation_outcomes(u, n_u, used))
                        for akey, used in _node_allocations(n_u, out_k.get(u, []), joint)
                    ]
                    alloc_cache[key] = opts
                per_node.append(opts)

            best_v, best_w, best_a = -1.0, math.inf, ()
            for combo in itertools.product(*per_node):
                akey = tuple(a for part in combo for a in part[0])
                if not joint and len(akey) > 1:
                    continue
                v = 0.0
                w = 0.0
                stats["n_actions"] += 1
                for outs in itertools.product(*(part[1] for part in combo)):
                    p = 1.0
                    merged = []
                    for nodes_part, pp in outs:
                        p *= pp
                        merged.extend(nodes_part)
                    stats["n_transitions"] += 1
                    nxt = _canon(merged, dst)
                    if nxt == GOAL:
                        v += p
                        w += p * (k + 1)
                    else:
                        v += p * V_next[nxt]
                        w += p * W_next[nxt]
                if v > best_v + TOL or (abs(v - best_v) <= TOL and w < best_w - TOL):
                    best_v, best_w, best_a = v, w, akey
            V_cur[s], W_cur[s] = best_v, best_w
            if keep_table:
                tab_k[s] = tuple(sorted(best_a))
            if (stats["n_states"] & 255) == 0:
                check_budget()
        check_budget()
        V_next, W_next = V_cur, W_cur

    if init == GOAL:
        value, exp_slot = 1.0, 0.0
    else:
        value = V_next.get(init, 0.0)
        exp_slot = W_next.get(init, 0.0) / value if value > 0 else math.inf
    return Policy(
        n_copies=n_copies, src=src, dst=dst, t0=model.t0,
        slot_width=model.slot_width, n_slots=K, ends=dict(model.ends),
        table=table, value=float(value), exp_slot=float(exp_slot),
        n_states=stats["n_states"], n_transitions=stats["n_transitions"],
        n_actions=stats["n_actions"], seconds=time.perf_counter() - t_start,
        complete=complete, joint=joint,
    )


def bruf(model: SlotModel, **kw) -> Policy:
    """BRUF: the optimal single-copy policy, with a complete (node, slot) table."""
    return solve(model, 1, complete=True, **kw)


def rucop(model: SlotModel, n_copies: int, **kw) -> Policy:
    """RUCoP-N: the optimal N-copy policy over forward-reachable states."""
    return solve(model, n_copies, **kw)


def simulate_in_model(policy: Policy, model: SlotModel, n: int,
                      rng: np.random.Generator) -> float:
    """Delivery ratio of `policy` when the slotted model itself is the world.

    Every attempt of contact c in slot k succeeds with probability q_c[k],
    independently. Its expectation is policy.value; the tests use it to check
    the backward induction.
    """
    ok = 0
    for _ in range(n):
        nodes = [model.src] * policy.n_copies
        for k in range(model.n_slots):
            s = _canon(nodes, model.dst)
            if s == GOAL:
                break
            act = policy.action(k, s) or ()
            for cid, cnt in act:
                u, v = model.ends[cid]
                if rng.random() < model.q[cid][k]:
                    moved = 0
                    for i in range(len(nodes)):
                        if nodes[i] == u and moved < cnt:
                            nodes[i] = v
                            moved += 1
        if model.dst in nodes:
            ok += 1
    return ok / n


# --------------------------------------------------------------------------
# online execution against a realization of the full laws
# --------------------------------------------------------------------------


def _hop(c: DetContact, size: float, a: float) -> float:
    """Arrival after taking c at time a; inf if the bundle does not fit.

    Same rule as routing/placement.py `_hop`: start at max(a, t_s), finish
    by t_e, then add the one-way light time.
    """
    tau = max(a, c.t_start)
    e = tau + size / c.rate
    if e > c.t_end:
        return math.inf
    return e + c.owlt


@dataclass
class ExecStats:
    """Diagnostics of one execution."""

    attempts: int = 0
    failures: int = 0
    misses: int = 0        # (slot, state) pairs with no stored action


def _next_action(policy: Policy, k: int, state, node: int):
    """First planned transmission from `node` at or after slot k, assuming the
    joint state stays as it is: (slot, [(cid, count), ...]) or None.

    States pruned at slot k reappear at later slots (waiting is always
    allowed, so a state reachable at k is reachable at every later slot);
    slots where the state is absent are skipped.
    """
    for k2 in range(k, policy.n_slots):
        act = policy.action(k2, state)
        if act is None:
            continue
        tx = [(cid, cnt) for cid, cnt in act if policy.ends[cid][0] == node]
        if tx:
            return k2, tx
    return None


def execute_policy(
    policy: Policy,
    real: Realization | dict[int, DetContact],
    size: float,
    deadline: float | None = None,
    mode: str = "asap",
    requery: bool = True,
    stats: ExecStats | None = None,
) -> float:
    """Run a BRUF / RUCoP policy on one realized plan.

    Returns the earliest arrival of any copy at the destination (inf if no
    copy gets there inside the policy's horizon). `deadline` is not used to
    steer the policy, which only knows its slot grid; compare the return value
    with the deadline to score it.

    `mode` sets how a table entry becomes a transmission:

      "asap"  the copy is queued on the next contact its policy plans for the
              current joint state (looking ahead over `wait` slots) and is
              sent as soon as that contact can carry it, which is how a DTN
              node forwards along a chosen contact and how the static routes
              in this study are executed. If the attempt fails, the copy is
              back in play at the boundary after the slot the attempt was
              planned for, as in the model.
      "slot"  decisions only take effect at the slot they are stored for: a
              `wait` holds the copy until the next boundary. Kept as a
              sensitivity check; on tight deadlines the waiting at slot
              boundaries costs deliveries the policy's own contacts would
              have made.
    """
    if mode not in ("asap", "slot"):
        raise ValueError(f"unknown execution mode {mode!r}")
    by_cid = real if isinstance(real, dict) else {c.cid: c for c in real.contacts}
    N, dst = policy.n_copies, policy.dst
    t0, dt, K = policy.t0, policy.slot_width, policy.n_slots
    nodes = [policy.src] * N
    ready = [t0] * N
    st = stats if stats is not None else ExecStats()

    for k in range(K):
        tk = t0 + k * dt
        tk1 = tk + dt
        decided = [False] * N
        for _round in range(8 * N + 8):
            if dst in nodes:
                break
            elig = [i for i in range(N) if not decided[i] and ready[i] < tk1]
            if not elig:
                break
            t_dec = max(tk, min(ready[i] for i in elig))
            group = [i for i in elig if ready[i] <= t_dec]
            state = _canon(nodes, dst)
            free: dict[int, list[int]] = {}
            for i in group:
                free.setdefault(nodes[i], []).append(i)

            plans: list[tuple[int, list[int], int]] = []     # (cid, copies, slot)
            if mode == "slot":
                act = policy.action(k, state)
                if act is None and t_dec > tk:
                    # a mid-slot arrival can land in a state the model reaches
                    # only from the next slot on: use that slot's decision
                    act = policy.action(k + 1, state)
                if act is None:
                    st.misses += 1
                    act = ()
                for cid, cnt in act:
                    u = policy.ends[cid][0]
                    pool = free.get(u, [])
                    copies, free[u] = pool[:cnt], pool[cnt:]
                    if copies:
                        plans.append((cid, copies, k))
            else:
                for u in list(free):
                    nxt = _next_action(policy, k, state, u)
                    if nxt is None:
                        continue
                    k2, tx = nxt
                    for cid, cnt in tx:
                        pool = free[u]
                        copies, free[u] = pool[:cnt], pool[cnt:]
                        if copies:
                            plans.append((cid, copies, k2))

            for cid, copies, k2 in plans:
                st.attempts += 1
                c = by_cid.get(cid)
                arr = _hop(c, size, t_dec) if c is not None else math.inf
                if math.isfinite(arr):
                    for i in copies:
                        nodes[i] = c.to
                        ready[i] = arr
                        # with requery, a copy landing before the next
                        # boundary gets a fresh decision in this slot
                        decided[i] = not requery
                else:
                    st.failures += 1
                    for i in copies:
                        ready[i] = t0 + (k2 + 1) * dt
                        decided[i] = True
            for pool in free.values():          # nothing planned: wait
                for i in pool:
                    decided[i] = True
        if dst in nodes:
            break

    arrivals = [ready[i] for i in range(N) if nodes[i] == dst]
    return min(arrivals) if arrivals else math.inf


# --------------------------------------------------------------------------
# static baselines executed on the same realizations
# --------------------------------------------------------------------------


def execute_route(by_cid: dict[int, DetContact], cids, size: float, t0: float) -> float:
    """A committed route followed blindly; inf as soon as a hop is missing or
    does not fit."""
    if not cids:
        return math.inf
    a = t0
    for cid in cids:
        c = by_cid.get(cid)
        if c is None:
            return math.inf
        a = _hop(c, size, a)
        if not math.isfinite(a):
            return math.inf
    return a


def execute_route_set(by_cid, route_cids, size: float, t0: float) -> float:
    """Earliest arrival over a replicated static route set."""
    return min((execute_route(by_cid, r, size, t0) for r in route_cids), default=math.inf)


def realize_with_state(graph: StochasticContactGraph, rng: np.random.Generator):
    """`graph.realize(rng)` that also returns the index of the latent state.

    Mirrors StochasticContactGraph.realize draw for draw (same RNG
    consumption, same realization); the index is what an oracle that knows
    the latent state would condition on. Returns (Realization, index | None).
    """
    idx = None
    scales: dict = {}
    if graph.latent is not None:
        probs = [p for p, _ in graph.latent.states]
        idx = int(rng.choice(len(graph.latent.states), p=probs))
        scales = graph.latent.states[idx][1]
    names = sorted({c.factor for c in graph.contacts
                    if c.factor is not None and c.factor_rho > 0.0})
    zf = {n: rng.standard_normal(1) for n in names} if names else None
    det = []
    for c in graph.contacts:
        s = scales.get(c.group, 1.0) if c.group is not None else 1.0
        d = c.sample(rng, p_scale=s, zf=zf)
        if d is not None:
            det.append(d)
    return Realization(graph.num_nodes, det), idx


def conditioned_graph(graph: StochasticContactGraph, state_index: int) -> StochasticContactGraph:
    """The graph with its latent state fixed to `state_index` (no latent left).

    Existence probabilities are scaled by that state's group factors. Used to
    build the latent-oracle RUCoP reference, which knows the state that an
    honest policy would have to infer.
    """
    _, scales = graph.latent.states[state_index]
    contacts = [
        c.with_laws(p_exist=c.p_exist * scales.get(c.group, 1.0), group=None)
        if c.group is not None else c
        for c in graph.contacts
    ]
    return StochasticContactGraph(num_nodes=graph.num_nodes, contacts=contacts,
                                  horizon=graph.horizon, latent=None)


# --------------------------------------------------------------------------
# probabilistic CGR (Walter and Feldmann, GLOBECOM 2019), simplified
# --------------------------------------------------------------------------


def contact_confidence(c: StochasticContact, latent=None) -> float:
    """Marginal existence probability: the per-contact confidence of PCGR."""
    p = c.p_exist
    if latent is not None and c.group is not None:
        p *= sum(pz * sc.get(c.group, 1.0) for pz, sc in latent.states)
    return float(p)


def pcgr_select(
    graph: StochasticContactGraph,
    candidates: list[list[StochasticContact]],
    size: float,
    t0: float,
    budget: int = 1,
) -> list[list[StochasticContact]]:
    """Rank candidate routes by the product of contact confidences, tie-break
    by nominal (mean-plan) arrival, and keep the top `budget`.

    This keeps the part of probabilistic CGR that matters for a comparison:
    a route's delivery probability is the product of independent per-contact
    confidences, and timing enters only through the nominal plan.
    """
    scored = []
    for r in candidates:
        nominal = delivery_time([c.mean_det() for c in r], size, t0)
        if not math.isfinite(nominal):
            continue
        p = 1.0
        for c in r:
            p *= contact_confidence(c, graph.latent)
        scored.append((-p, nominal, len(r), r))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return [r for *_, r in scored[:budget]]
