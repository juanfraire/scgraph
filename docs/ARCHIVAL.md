# Release, archival and DOI plan

What to do, and when, so that the paper can cite a permanent, citable version of
this code.

---

## Recommendation in one line

**Software Heritage for the GitLab Inria repository, plus a Zenodo DOI via a
read-only GitHub mirror.** Software Heritage gives permanence with no extra
infrastructure; Zenodo gives the DOI that IEEE and readers expect. They are
complementary, and doing both costs about twenty minutes once.

---

## Why not just one of them

| | Software Heritage | Zenodo |
|---|---|---|
| Works directly with gitlab.inria.fr | **yes**, "Save code now" on any public repo | no, the GitHub integration is what is automated |
| Issues a DOI | SWHID, and a DOI for deposits | **yes**, a versioned DOI per release |
| Expected by IEEE / reviewers | less familiar | **the usual citation target** |
| Captures full git history | **yes**, the whole VCS | no, a snapshot of the tag |
| Effort | one URL submission | one mirror + one release |

Software Heritage is the natural fit for an Inria-hosted project (Inria founded
it), and it archives the *history*, not just a tarball. But a reviewer asked to
"cite the software" wants a DOI, and Zenodo is where that lives. Hence both.

---

## The plan

### 1. Before submission (by 31 August 2026)

- [ ] Create the GitLab Inria project (`gitlab.inria.fr/jfraire/scgraph`) and
      push. **Make it public** — an artefact behind a login is not an artefact.
- [ ] Confirm the CI passes: `test:pinned`, then run the `anchor` and
      `reproduce` jobs manually at least once.
- [ ] Check `make reproduce && make verify` from a **fresh clone** into a clean
      virtualenv, not from the working tree.
- [ ] Tag `v0.1.0-submission` at the exact commit the manuscript's numbers came
      from, and note the short SHA in the submission cover letter.
- [ ] Archive that tag on Software Heritage:
      https://archive.softwareheritage.org/save/ → paste the repository URL.
      Record the resulting SWHID in this file.
- [ ] Paste the code-availability statement
      ([`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md)) into `main.tex`.

At submission the DOI does not exist yet, and that is fine: cite the repository
URL and the tag. Reviewers can clone it.

### 2. At acceptance

- [ ] Mirror to GitHub (read-only, e.g. `github.com/jfraire/scgraph`), with the
      README stating that GitLab Inria is canonical.
- [ ] Enable the Zenodo–GitHub integration for the mirror, **before** cutting
      the release: Zenodo only mints DOIs for releases published after the hook
      is switched on.
- [ ] Cut `v1.0.0` on the mirror. Zenodo mints a concept DOI (all versions) and
      a version DOI (this one).
- [ ] **Cite the concept DOI** in the paper, so the citation keeps resolving
      when the code is updated.
- [ ] Update in this repository:
      - `CITATION.cff`: `doi`, `volume`, `start`, `end`, `year`,
        `date-released`, and the software `identifiers`
      - `README.md`: the "How to cite" BibTeX (both entries)
      - this file: record the SWHID and both DOIs below

### 3. Recorded identifiers

Fill in as they are issued.

```
Submission tag   : v0.1.0-submission   commit ________
SWHID            : swh:1:rev:__________________________________
Zenodo concept   : 10.5281/zenodo.__________   (cite this one)
Zenodo v1.0.0    : 10.5281/zenodo.__________
Paper DOI        : 10.1109/OJCOMS.2026.__________
```

---

## What the archived snapshot must contain

The point of the tag is that someone can reproduce the paper from it alone.
Before tagging, confirm from a fresh clone:

- [ ] `scenarios/uc1_cpd/` and `scenarios/scale_{6,12,24,48}/` are **committed**
      (~190 kB packed). `run_runtime.py` skips missing sizes silently, so a
      missing skeleton degrades into a quietly shorter figure rather than an
      error.
- [ ] `results/reference/` is committed — this is what `make verify` compares
      against, and therefore what pins the manuscript's numbers.
- [ ] `results/*.csv` and `figures/*.{pdf,png}` are committed, so the artefact
      is inspectable without running anything.
- [ ] `LICENSE`, `CITATION.cff`, `requirements-repro.txt` are present.
- [ ] `rust/asabr-cli/patches/` is present. Without it the anchor cannot be
      rebuilt by anyone.

Deliberately **not** archived: `rust/vendor/` (A-SABR is fetched at a pinned
revision), `.venv/`, build output.

---

## The A-SABR dependency, and the risk it carries

The anchor depends on an external repository at a pinned revision, plus a patch
carried here. Two failure modes are worth naming, because an artefact is
supposed to outlive its dependencies:

1. **Upstream disappears or rewrites history.** The pin
   (`1277e360f2d61a593c336c6f6eb2e583758d7580`) would stop resolving and the
   anchor could not be rebuilt. Everything else still works: no result in the
   paper depends on the anchor.
2. **The patch stops applying**, if the pin is ever bumped without updating it.

The clean fix is to get the patch upstream. It is a small, uncontroversial
change — three call sites moved to the stable uninit-array idiom, needed because
`MaybeUninit::uninit().into()` has never compiled on stable Rust and rustc 1.90
rejects it outright. **Worth opening as a PR against DTN-MTP/A-SABR.** Once
merged, replace the clone-and-patch in `scripts/build_anchor.sh` with a plain
git dependency on the merged revision and delete the patch.

If that stalls, the fallback is to fork A-SABR under an account you control
(Inria GitLab or personal GitHub), apply the patch on a branch, and pin that.
Vendoring the crate into this repository is the last resort: it is
MIT-licensed so it is permitted, but it forks someone else's code into a paper
artefact, which is worse for both parties than a pinned fork.

---

## Reproducibility badges

IEEE OJ-COMS does not run an artefact-evaluation track, so there is no badge to
apply for. If a reviewer asks for evidence, point at:

- `make reproduce && make verify` — regenerates every number and diffs it
- the `reproduce` CI job, which does exactly that on every tag
- `make anchor` — the cross-check against an independent implementation

That is a stronger claim than most badges certify.
