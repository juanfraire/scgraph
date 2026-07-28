# The pipeline, stage by stage

This walks through Section IV of the paper (*Populating and Learning the Graph*)
and Section V (*SLA-Aware Routing*) as executable steps, in the order Fig. 2
draws them:

```
scenario ──CPD──▶ orbital skeleton ──inject/annotate──▶ ensemble ──estimate──▶ 𝒢 ──▶ routing ──▶ verification
          Stage 1      (A1, point mass)     Stage 2         Stage 3              §V           §V-F
```

Every snippet below is real, was run against the committed skeleton, and its
printed output is reproduced verbatim. Run them from the repository root with
`.venv/bin/python`. UC1 is the vehicle because it is the only use case that
exercises all three stages; where UC2 and UC3 differ is noted at each step.

The one property that makes the whole loop work, and that the paper leans on
repeatedly: **a realisation drawn from 𝒢 is itself a valid classical contact
plan**, so an unmodified deterministic router runs inside the verification loop.

---

## Stage 0 — the scenario

`scenarios/uc1_skeleton.yaml` is the input to the Contact Plan Designer: a
Walker-Delta shell of 12 satellites in 3 planes at 600 km and 53°, an event site
(Lyon) and a gateway (Paris) 390 km apart, a 10° elevation mask, and a six-hour
horizon at 10 s propagation steps.

```yaml
constellations:
  - code: "D:600:53:12/3/1"       # 12 sats, 3 planes, 600 km, 53 deg
    base-id: 200
nodes:
  - id: 101   # EventSite, 45.76 N  4.83 E
  - id: 102   # Gateway,   48.85 N  2.35 E
```

The generated plan is **committed** under `scenarios/uc1_cpd/`, so nothing
downstream needs CPD installed. To regenerate it (or the larger constellations
the runtime study uses):

```bash
scripts/generate_skeletons.sh          # needs `cpd` on PATH
```

---

## Stage 1 — the orbital skeleton (assumption A1)

The space layer is *computed*, not invented. CPD propagates the constellation,
computes accesses, resolves interface conflicts, and exports a classical contact
plan; `scgraph.io.cpd` reads it back as contacts whose every law is a point
mass. That is assumption A1 expressed in code.

```python
from scgraph.io import load_cpd_plan

sk = load_cpd_plan("scenarios/uc1_cpd", layer="space")
print(len(sk.contacts), sk.num_nodes, sk.horizon)
print(sk.contacts[0])
print(sk.contacts[0].is_point_mass)
```

```
298 14 21600.0
StochasticContact(cid=0, frm=7, to=10, start=Dirac(0.0),
                  duration=Dirac(101.31805419921875), rate=Dirac(10.0),
                  owlt=0.009586415737473585, p_exist=1.0, layer='space', group=None)
True
```

Three things to notice:

- **Everything is a `Dirac`.** The skeleton is a `StochasticContactGraph` whose
  laws happen to be degenerate, which is exactly the deterministic corner of
  eq. (1). Nothing special-cases it.
- **Node ids are remapped.** CPD ids are sparse and meaningful (101 = event
  site, 102 = gateway, 2xx = the constellation); scgraph indexes densely from
  zero. `sk.index` maps CPD → scgraph and `sk.label` inverts it, so a route can
  always be reported back in the operator's numbering. Use
  `sk.node(101)` rather than hard-coding an index.
- **OWLT is computed when CPD omits it**, from the reported range: near-Earth
  one-way light time is milliseconds, but it is not zero.

`load_cpd_plan` accepts either `plan.json` (CPD's native export, float window
edges) or `ionFormatContactPlan.txt` (the ION `a contact +start +end tx rx rate`
dialect, integer-valued). Pass the directory and it picks whichever it finds.

---

## Stage 2 — stochastic annotation

The air and ground layers are *added* on top of the skeleton, and the ensemble
is annotated with laws. Annotation is deliberately non-invasive: the underlying
plan stays readable by any deterministic tool.

For UC1 this is `make_instance_cpd`, which injects three air relays whose
hand-offs jitter with standard deviation σ:

```python
import numpy as np
from scgraph.scenarios.uc1_event import make_instance_cpd

rng = np.random.default_rng(20260709)
g, p = make_instance_cpd(rng, sigma=60.0)

print(len(g.contacts), "contacts")          # 298 space + 6 air
print({k: round(v, 2) if isinstance(v, float) else v for k, v in p.items()})
print([c for c in g.contacts if c.layer == "air"][0].start)
print(g.is_point_mass)
```

```
304 contacts
{'src': 0, 'dst': 1, 'size': 100.0, 't0': 5671.38, 'deadline': 6534.52,
 'space_arrival': 6494.52, 'pass_wait': 823.14}
TruncNormal(mu=5720.453028431042, sigma=60.0, lo=5510.453028431042, hi=5930.453028431042)
False
```

**An instance is a release time, not a random topology.** `t0 = 5671.38` s is a
moment drawn from inside the six-hour plan; the geometry a router faces differs
from instance to instance while every space contact stays physically
realisable. Release times are restricted to a 200–900 s pass-wait band
(`PASS_BAND`), because below it the backbone is already prompt and a local relay
buys nothing, and above it the deadline is so loose every policy wins. Here
`pass_wait = 823.14` s.

**Where the laws go, and why.** A2 places the law on the start and on the
**duration**, never on the end directly, so that `T_e = T_s + D` makes A5's
ordering `T_s ≤ T_e` true by construction rather than a constraint to enforce
between two marginals. The window shifts rigidly with its start:

```python
StochasticContact(
    cid, src, relay,
    start=TruncNormal(nominal, sigma, lo=nominal - 3.5*sigma, hi=nominal + 3.5*sigma),
    duration=Dirac(dur),      # rigid window
    rate=Dirac(rate),
    owlt=0.5, layer="air",
)
```

Existence is carried separately as `p_exist`, which is A3's factored form
`p_c · f(T_s, T_e, V | exists)`. Section III-A shows the two are formally one
(possibly defective) joint law; they are kept apart in code because existence
and timing are estimated from different signals.

The other two use cases annotate differently:

| | what is random | where |
|---|---|---|
| UC1 | timing (`TruncNormal` start, rigid window) | `scenarios/uc1_event.py` |
| UC2 | capacity (`LogNormal` fade on real pass volumes) | `scenarios/uc2_logistics.py` |
| UC3 | existence + **correlation** via a shared latent state | `scenarios/uc3_emergency.py` |

UC3 carries no orbital layer at all: it is a ground/D2D collapse scenario by
construction, and the paper says so rather than leaving it as an apparent
omission.

**Correlation** is the one thing that cannot be expressed per contact. It lives
in a `LatentModel` on the graph, and contacts opt in by naming a `group`:

```python
from scgraph.graph import LatentModel

g.latent = LatentModel(states=[
    ((1-rho)*(1-rho), {"A": 1.0,  "B": 1.0}),    # neither region collapses
    (rho*(1-rho),     {"A": 0.1,  "B": 1.0}),    # A collapses
    ((1-rho)*rho,     {"A": 1.0,  "B": 0.1}),    # B collapses
    (rho*rho,         {"A": 0.1,  "B": 0.1}),    # both
])
```

Each state multiplies the existence probability of every contact in the named
group. Drawing one state per Monte-Carlo sample and sharing it across contacts
is what turns eq. (12)'s optimistic independence into eq. (13).

---

## Stage 3 — estimating the laws

In v1 the parameters come from domain models rather than being fitted
end-to-end: ITU-R P.618 rain fade for feeder capacity, 3GPP NTN mobility and
handover dynamics for air/ground timing jitter, and channel-access statistics
for D2D existence. Because the evaluation **sweeps these parameters as its
independent variables** (σ in UC1, the fade coefficient of variation in UC2, ρ
in UC3), the conclusions are sensitivity results over the laws rather than
artefacts of one calibration.

This is the seam where learning drops in. Contact prediction, in this framing,
is the regression from observed history to `(p_c, f(T_s,T_e), f(V))` — a
replacement for Stage 3, not a new pipeline. Nothing downstream would change.

---

## 𝒢 — what you now have

`StochasticContactGraph` is just the collection of stochastic contacts plus the
optional latent model, and it offers three views:

```python
real = g.realize(rng)        # one draw: a classical contact plan
mean = g.mean_plan()         # law means: the nominal plan an operator would build
g.is_point_mass              # True only for a fully degenerate graph
```

```python
print(type(real).__name__, len(real.contacts))
print(real.contacts[0])
```

```
Realization 304
DetContact(cid=0, frm=7, to=10, t_start=0.0, t_end=101.31805419921875,
           rate=10.0, owlt=0.009586415737473585)
```

`DetContact` keeps `cid`, so a route found on a realisation can be mapped back
to the stochastic contacts it came from (`g.by_id(cid)`) and re-scored under the
laws. That round trip is used by every experiment.

`Realization.write_asabr(path)` emits an A-SABR-format plan for the correctness
anchor. It **raises rather than rounding** on non-integral attributes, because
A-SABR's lexer parses numbers as `i64`.

---

## Routing (Section V)

### The baseline: deterministic CGR on the nominal plan

```python
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import route_stats

src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
det, arrival = cgr_route(mean, src, dst, size, t0)
print([(c.frm, c.to) for c in det], round(arrival, 2))

det_sc = [g.by_id(c.cid) for c in det]                      # back to the laws
print(route_stats(det_sc, size, t0, D, 2000, rng)["p_success"])
```

```
[(0, 15), (15, 1)] 5746.58
0.762
```

This is the H1 baseline in two lines: route on the mean plan, then **live with
the real laws**. The deterministic router takes the nominally fast air relay
pair and, at σ = 60 s, misses its deadline about a quarter of the time.

### Candidate generation

`Q_{1-ε}` has no optimal substructure (the quantile of a sum is not the sum of
quantiles), so it cannot be a Dijkstra label. It is used to **rank an enumerated
candidate set** instead:

```python
from scgraph.routing.quantile import candidate_routes

cands = candidate_routes(g, src, dst, size, t0, rng, n_realizations=40)
print(len(cands))
for r in cands[:4]:
    print([(c.frm, c.to, c.layer) for c in r])
```

```
22
[(0, 15, 'air'), (15, 1, 'air')]
[(0, 14, 'air'), (14, 1, 'air')]
[(0, 16, 'air'), (16, 1, 'air')]
[(0, 2, 'space'), (2, 1, 'space')]
```

Three sources feed the set, and the third is load-bearing:

1. the mean-plan route;
2. CGR-style single-contact exclusions of it (SABR's alternative route search);
3. a **disjoint suppression sweep** — repeatedly exclude every contact used so
   far and re-route.

Without (3) the timing-certain space backbone never enters the candidate set,
because it is always slower than the air relays on the nominal plan, and H1
does not reproduce. That is the fourth entry above.

### The SLA-aware router

```python
from scgraph.routing.quantile import pick_route, pick_route_chance

route, stats = pick_route(g, src, dst, size, t0, D, rng, m=2000)
print(stats["p_success"], round(stats["quantile"], 2), [c.layer for c in route])
```

```
1.0 6494.52 ['space', 'space']
```

`pick_route` is eq. (6), argmax `P[T ≤ D]` over the candidate set, ties broken
by mean delivery time. It abandons the air relays for the backbone and drives
the violation rate to zero. The price is visible in the next call.

`pick_route_chance` is eq. (7), the chance-constrained form: among candidates
meeting `P[T ≤ D] ≥ 1-ε`, take the fastest. Eq. (9) is what licenses testing
admissibility on `P` rather than on the quantile.

```python
r, s = pick_route_chance(g, src, dst, size, t0, D, rng, m=2000, eps=0.05)
print(s["feasible"], s["p_success"], round(s["mean_delivery"] - t0, 1))
```

```
True 1.0 823.1
```

**823 s of end-to-end latency** against the deterministic route's ~75 s: that is
the reliability price H1 reports, and `run_eps.py` is what turns ε into the knob
that trades between them. Note `mean_delivery - t0`: with a CPD skeleton the
release time sits hours into the plan, so absolute clock time is not latency.

When no candidate meets the constraint, `pick_route_chance` falls back to argmax
`P` and reports `feasible: False`, so the caller can tell a best effort from a
certificate.

---

## Verification (Section V-F)

Two independent evaluators, plus a reference bound.

**(a) Monte Carlo**, the default. Draw `M` realisations of the attributes along
a route and apply the recursion. `route_stats` above is this.

**(b) Analytic convolution**, propagating the arrival CDF on a grid:

```python
from scgraph.analytic import route_p_success
print(route_p_success(det_sc, size, t0, D))       # 0.760
```

against Monte Carlo's `0.762` on the same route — agreement within the CI, which
is what `tests/test_quantile.py` asserts. **The convolution is the slower path
here**, roughly 200× slower at the ε values the paper uses (a fixed ~170 ms grid
cost against 0.75 ms for M = 2000; they cross near M ≈ 5·10⁵). The paper says so
explicitly, having first assumed the opposite. It is kept for extreme tails.

**The rerouting reference.** `verify_fullgraph` realises the *whole* graph and
lets a deterministic router re-plan on each draw, which is an omniscient upper
bound no committed policy can beat:

```python
from scgraph.verify.montecarlo import verify_fullgraph
v = verify_fullgraph(g, src, dst, size, t0, D, 400, rng)
print(v["p_success"], v["ci"])          # 1.0 (0.9905, 1.0)
```

Wilson score intervals, not normal approximations, because the proportions here
are routinely at or near 1.

---

## The two corners

The subsumption claim is checked in both directions, and the two checks are of
different kinds.

**Deterministic corner — an equality, enforced in code.** On a point-mass graph,
`cgr_route` must agree with A-SABR exactly:

```bash
make anchor      # builds the Rust CLI, then runs 20 randomised cross-checks
```

**Bernoulli corner — a lossy projection, measured.** `routing/bernoulli.py`
builds the discrete on/off TEG of the uncertainty-aware DTN line explicitly,
per eq. (14), then routes on it and scores the result under the *full* laws:

```python
from scgraph.routing.bernoulli import bernoulli_route

br, belief = bernoulli_route(g, src, dst, size, t0, D, 30.0, rng, m=2000)
print([c.layer for c in br], belief)
print(route_stats(br, size, t0, D, 2000, rng)["p_success"])
```

```
['space', 'space'] 1.0
1.0
```

The second number is the honest one: `belief` is the coarse model's *own*
prediction, which it is entitled to get wrong. Scoring under eq. (1) is the
caller's job, and it is what makes the ablation a fair comparison of routing
decisions rather than of evaluators. Sweeping the slot width Δ traces the path
between the two corners, which is Fig. 3.

---

## Where each experiment enters

| Experiment | Enters at | Varies |
|---|---|---|
| `run_uc1.py` | Stage 2 | σ, the timing jitter |
| `run_ablation.py` | the Bernoulli projection | Δ, the slot width |
| `run_eps.py` | routing | ε, the violation budget |
| `run_uc2.py` | Stage 2 (capacity only) | fade severity; uses the plan's real pass volumes |
| `run_uc3.py` | Stage 2 (no orbital layer) | ρ, the collapse probability |
| `run_placement.py` | execution (§VI) | which placement hosts the planning plane |
| `run_runtime.py` | Stage 1 | plan size: 98 → 14 640 contacts |

---

## A note on equation numbers

Docstrings throughout the code cite equation numbers from `main.tex`, checked
against the current draft (eqs. 1–14, verified via `main.aux`). If the
manuscript gains or loses a numbered equation, these references drift silently.
The mapping is tabulated in the [README](../README.md#equations); re-check it
against `main.aux` after any change to the paper's equation set.
