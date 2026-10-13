"""BRUF / RUCoP backward induction and online execution (routing/rucop.py).

What is pinned: the solver reproduces the uncertain-DTN artefact's toy RUCoP
exactly, it is optimal against the obvious bounds (more copies never hurt; a
contingent policy is never worse than the static max-product path in its own
model), its value is what the policy achieves when the slotted model is the
world, and the executor turns a policy into the delivery the realization
allows.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from scgraph.contact import DetContact, StochasticContact
from scgraph.distributions import Dirac
from scgraph.graph import Realization, StochasticContactGraph
from scgraph.routing.bernoulli import bernoulli_route
from scgraph.routing.rucop import (
    BudgetExceeded,
    SlotModel,
    bruf,
    execute_policy,
    execute_route,
    pcgr_select,
    project,
    realize_with_state,
    rucop,
    simulate_in_model,
    solve,
)
from scgraph.scenarios.uc1_event import make_instance
from scgraph.scenarios.uc3_emergency import make_instance as make_uc3

# examples/toy_rucop.py of the uncertain-DTN artefact: (cid, from, to, slot, p)
TOY = [(0, 0, 1, 0, 0.8), (1, 0, 2, 0, 0.7), (2, 1, 3, 1, 0.9), (3, 2, 3, 1, 0.85)]
TOY_PATH = Path("/Users/jfraire/Documents/Code/uncertain-dtn/artifact/examples/toy_rucop.py")


def toy_model() -> SlotModel:
    return SlotModel.from_table(TOY, n_slots=2, src=0, dst=3)


# --------------------------------------------------------------------------
# the toy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_toy_values_reproduced_exactly(n):
    """The toy's action space (one transmission per slot) gives 0.72 for every N."""
    pol = solve(toy_model(), n, joint=False)
    assert pol.value == pytest.approx(0.72, abs=1e-12)
    assert pol.action(0, (0,) * n) == ((0, 1),)      # one copy over 0->1


@pytest.mark.skipif(not TOY_PATH.exists(), reason="uncertain-DTN artefact not on this machine")
@pytest.mark.parametrize("n", [1, 2])
def test_toy_module_agrees(n):
    spec = importlib.util.spec_from_file_location("toy_rucop", TOY_PATH)
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    _, sdp = toy.compute_optimal_policy(n)
    assert solve(toy_model(), n, joint=False).value == pytest.approx(sdp, abs=1e-12)


def test_joint_actions_split_copies_over_both_relays():
    """With RUCoP's joint actions, two copies leave together over both relays."""
    m = toy_model()
    assert rucop(m, 1).value == pytest.approx(0.72, abs=1e-12)
    two = rucop(m, 2)
    assert two.value == pytest.approx(1 - (1 - 0.8 * 0.9) * (1 - 0.7 * 0.85), abs=1e-12)
    assert two.action(0, (0, 0)) == ((0, 1), (1, 1))
    assert bruf(m).value == pytest.approx(0.72, abs=1e-12)


def test_budget_is_enforced():
    with pytest.raises(BudgetExceeded):
        rucop(toy_model(), 3, max_transitions=5)


# --------------------------------------------------------------------------
# optimality bounds on projected scenarios
# --------------------------------------------------------------------------


def _uc3_model(slot=30.0, seed=3, rho=0.5):
    g, p = make_uc3(np.random.default_rng(seed), rho)
    m = project(g, p["src"], p["dst"], p["size"], p["t0"], p["deadline"], slot,
                np.random.default_rng(0), m=2000)
    return g, p, m


def test_bruf_equals_rucop_one():
    _, _, m = _uc3_model()
    assert bruf(m).value == pytest.approx(rucop(m, 1).value, abs=1e-12)


def test_more_copies_never_hurt():
    _, _, m = _uc3_model()
    v = [rucop(m, n).value for n in (1, 2, 3)]
    assert v[0] <= v[1] + 1e-12 <= v[2] + 2e-12


@pytest.mark.parametrize("which", ["uc1", "uc3"])
def test_contingent_dominates_static_path_in_its_own_model(which):
    """The static max-product path is one policy of the MDP, so BRUF >= it."""
    if which == "uc3":
        g, p, m = _uc3_model()
    else:
        g, p = make_instance(np.random.default_rng(1), 40.0)
        m = project(g, p["src"], p["dst"], p["size"], p["t0"], p["deadline"], 30.0,
                    np.random.default_rng(0), m=2000)
    _, p_static = bernoulli_route(g, p["src"], p["dst"], p["size"], p["t0"],
                                  p["deadline"], 30.0, np.random.default_rng(0), q=m.q)
    assert bruf(m).value >= p_static - 1e-12


def test_value_is_what_the_policy_achieves_in_the_model():
    _, _, m = _uc3_model()
    pol = rucop(m, 2)
    n = 6000
    sim = simulate_in_model(pol, m, n, np.random.default_rng(11))
    se = math.sqrt(pol.value * (1 - pol.value) / n) + 1e-3
    assert sim == pytest.approx(pol.value, abs=4 * se)


# --------------------------------------------------------------------------
# execution against realizations
# --------------------------------------------------------------------------


def _toy_graph():
    """The toy as a stochastic contact graph: each contact fills one 10 s slot."""
    g = StochasticContactGraph(num_nodes=4)
    for cid, u, v, k, p in TOY:
        g.add(StochasticContact(cid, u, v, Dirac(10.0 * k), Dirac(10.0), Dirac(1.0),
                                p_exist=p))
    return g


@pytest.mark.parametrize("n,expected", [(1, 0.72), (2, 1 - 0.28 * 0.405)])
def test_execution_matches_model_when_the_model_is_exact(n, expected):
    g = _toy_graph()
    m = SlotModel.from_table(TOY, 2, 0, 3, t0=0.0, slot_width=10.0)
    pol = rucop(m, n)
    rng = np.random.default_rng(5)
    trials = 20000
    ok = sum(execute_policy(pol, g.realize(rng), size=1.0) <= 20.0 for _ in range(trials))
    assert ok / trials == pytest.approx(expected, abs=0.015)


def test_projection_of_the_toy_graph_recovers_the_toy():
    g = _toy_graph()
    m = project(g, 0, 3, 1.0, 0.0, 20.0, 10.0, np.random.default_rng(0), m=40000)
    assert m.n_slots == 2
    assert m.q[0][0] == pytest.approx(0.8, abs=0.01)
    assert m.q[0][1] == 0.0
    assert m.q[2][1] == pytest.approx(0.9, abs=0.01)
    assert rucop(m, 2).value == pytest.approx(0.8866, abs=0.01)


def test_point_mass_chain_delivers_at_the_earliest_slot_boundary():
    g = StochasticContactGraph(num_nodes=3)
    g.add(StochasticContact.deterministic(0, 0, 1, 0.0, 100.0, 10.0))
    g.add(StochasticContact.deterministic(1, 1, 2, 20.0, 120.0, 10.0))
    m = project(g, 0, 2, 10.0, 0.0, 100.0, 10.0, np.random.default_rng(0), m=200)
    pol = bruf(m)
    assert pol.value == 1.0
    # 0->1 at t=0 lands at t=1; 1->2 opens at 20, a slot boundary: arrival 21
    assert execute_policy(pol, g.realize(np.random.default_rng(1)), 10.0) == pytest.approx(21.0)


def test_failed_attempt_leaves_the_copy_and_the_policy_acts_again():
    """Contact 0 (slot 0) is planned first; it is missing, so contact 1 is used."""
    m = SlotModel.from_table([(0, 0, 1, 0, 0.6), (1, 0, 1, 1, 0.5)], 2, 0, 1,
                             t0=0.0, slot_width=10.0)
    pol = bruf(m)
    assert pol.value == pytest.approx(0.6 + 0.4 * 0.5)
    assert pol.action(0, (0,)) == ((0, 1),)
    real = Realization(2, [DetContact(1, 0, 1, 10.0, 20.0, 1.0)])     # contact 0 absent
    assert execute_policy(pol, real, size=1.0) == pytest.approx(11.0)
    real_none = Realization(2, [])
    assert math.isinf(execute_policy(pol, real_none, size=1.0))


def test_copies_sent_over_one_contact_move_together():
    m = SlotModel.from_table([(0, 0, 1, 0, 0.5), (1, 1, 2, 1, 0.5)], 2, 0, 2,
                             t0=0.0, slot_width=10.0)
    pol = rucop(m, 2)
    real = Realization(3, [DetContact(0, 0, 1, 0.0, 10.0, 1.0),
                           DetContact(1, 1, 2, 10.0, 20.0, 1.0)])
    assert execute_policy(pol, real, size=1.0) == pytest.approx(11.0)


def test_realize_with_state_mirrors_graph_realize():
    g, _ = make_uc3(np.random.default_rng(2), 0.6)
    a = g.realize(np.random.default_rng(9))
    b, idx = realize_with_state(g, np.random.default_rng(9))
    assert a.contacts == b.contacts
    assert idx in range(len(g.latent.states))


def test_static_route_execution():
    by = {0: DetContact(0, 0, 1, 0.0, 10.0, 1.0), 1: DetContact(1, 1, 2, 5.0, 6.0, 1.0)}
    assert execute_route(by, [0, 1], 1.0, 0.0) == pytest.approx(6.0)
    assert math.isinf(execute_route(by, [0, 1], 2.0, 0.0))     # second hop too short


def test_pcgr_ranks_by_confidence_then_nominal_arrival():
    g = StochasticContactGraph(num_nodes=3)
    fast = StochasticContact(0, 0, 1, Dirac(0.0), Dirac(50.0), Dirac(10.0), p_exist=0.5)
    slow = StochasticContact(1, 0, 1, Dirac(20.0), Dirac(50.0), Dirac(10.0), p_exist=0.9)
    slower = StochasticContact(2, 0, 1, Dirac(30.0), Dirac(50.0), Dirac(10.0), p_exist=0.9)
    for c in (fast, slow, slower):
        g.add(c)
    picked = pcgr_select(g, [[fast], [slower], [slow]], 10.0, 0.0, budget=2)
    assert [r[0].cid for r in picked] == [1, 2]


def test_asap_mode_leaves_when_the_planned_contact_opens():
    """1->2 opens at 15 s, between boundaries: "asap" leaves at 15, "slot" at 20."""
    g = StochasticContactGraph(num_nodes=3)
    g.add(StochasticContact.deterministic(0, 0, 1, 0.0, 100.0, 10.0))
    g.add(StochasticContact.deterministic(1, 1, 2, 15.0, 100.0, 10.0))
    m = project(g, 0, 2, 10.0, 0.0, 100.0, 10.0, np.random.default_rng(0), m=200)
    pol = bruf(m)
    real = g.realize(np.random.default_rng(1))
    assert execute_policy(pol, real, 10.0, mode="asap") == pytest.approx(16.0)
    assert execute_policy(pol, real, 10.0, mode="slot") == pytest.approx(21.0)


def test_asap_failure_resumes_after_the_planned_slot():
    """Planned contact (slot 2) missing: the copy is back at t_3 and uses slot 3."""
    m = SlotModel.from_table([(0, 0, 1, 2, 0.5), (1, 0, 1, 3, 0.5)], 4, 0, 1,
                             t0=0.0, slot_width=10.0)
    pol = bruf(m)
    real = Realization(2, [DetContact(1, 0, 1, 30.0, 40.0, 1.0)])
    assert execute_policy(pol, real, size=1.0, mode="asap") == pytest.approx(31.0)
