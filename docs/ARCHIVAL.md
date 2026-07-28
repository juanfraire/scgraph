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
| Works directly with gitlab.inria.fr | **yes**, "Save code now" on any public repo | no, only the GitHub integration is automated |
| Issues a DOI | SWHID, and a DOI for deposits | **yes**, a versioned DOI per release |
| Expected by IEEE / reviewers | less familiar | **the usual citation target** |
| Captures full git history | **yes**, the whole VCS | no, a snapshot of the tag |
| Effort | one URL submission | one mirror + one release |

Software Heritage is the natural fit for an Inria-hosted project (Inria founded
it), and it archives the whole *history* instead of a tarball. But a reviewer asked to
"cite the software" wants a DOI, and Zenodo is where that lives. Hence both.

---

## The plan

### 1. Before submission (by 31 August 2026)

- [ ] Create the GitLab Inria project (`gitlab.inria.fr/jfraire/scgraph`) and
      push. **Make it public**: an artefact behind a login is no artefact.
- [ ] Confirm the CI passes: `test:pinned`, then run the `anchor` and
      `reproduce` jobs manually at least once.
- [ ] Check `make reproduce && make verify` from a **fresh clone** into a clean
      virtualenv, and not from the working tree.
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
A-SABR pin       : 1277e360f2d61a593c336c6f6eb2e583758d7580  (2026-06-30, fixed)
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
      missing skeleton quietly shortens the figure instead of raising an
      error.
- [ ] `results/reference/` is committed. `make verify` compares against it, so
      it is what pins the manuscript's numbers.
- [ ] `results/*.csv` and `figures/*.{pdf,png}` are committed, so the artefact
      is inspectable without running anything.
- [ ] `LICENSE`, `CITATION.cff`, `requirements-repro.txt` are present.
- [ ] `rust/asabr-cli/Cargo.toml` and `rust-toolchain.toml` are present: the
      A-SABR pin and the minimum Rust version are the only things needed to
      rebuild the anchor.

Deliberately **not** archived: `.venv/` and build output. A-SABR is not
vendored; Cargo fetches it at the pinned revision.

---

## The A-SABR dependency, and the risk it carries

The anchor depends on an external repository at a pinned revision, fetched by
Cargo. Nothing is vendored or patched.

```
repo : https://github.com/DTN-MTP/A-SABR.git
rev  : 1277e360f2d61a593c336c6f6eb2e583758d7580
date : 2026-06-30  ("ops(ci): add manual workflow")
```

Recorded in two places, both committed: `rust/asabr-cli/Cargo.toml` (`rev =`)
and `rust/asabr-cli/Cargo.lock`. Quote this revision in the paper or the cover
letter if a reviewer asks what "anchored against A-SABR" means concretely.

The pin is already behind upstream (73 commits as of 2026-07-28) and that is
fine. A-SABR restructured the API the CLI imports, so re-anchoring is a port
and not a version bump. No result in the paper depends on the anchor either
way. See the README's "Which A-SABR, exactly" for the detail.

One failure mode is worth naming, because an artefact is supposed to outlive
its dependencies: **upstream could disappear or rewrite history**, and the pin
would stop resolving. Everything else still works, since no result in the
paper depends on the anchor and `make reproduce` never touches it.

If that ever happens, fork A-SABR under an account you control (Inria GitLab or
personal GitHub) and repoint the `git =` URL. Vendoring the crate into this
repository is the last resort: it is MIT-licensed so it is permitted, but it
copies someone else's code into a paper artefact, which is worse for both
parties than a pinned fork.

**A note on toolchain versions**, because it cost us a wrong diagnosis once.
A-SABR uses standard-library features stabilised up to Rust 1.97. On an older
toolchain the build fails with a long list of E0658 "use of unstable library
feature" errors, which looks exactly like a crate that requires nightly. It is
not: A-SABR targets stable and its CI runs on stable. `rust-toolchain.toml`
pins the minimum so rustup reports the real problem, and `rustup update` is the
fix. Do not patch A-SABR to work around this.

---

## Reproducibility badges

IEEE OJ-COMS does not run an artefact-evaluation track, so there is no badge to
apply for. If a reviewer asks for evidence, point at:

- `make reproduce && make verify`, which regenerates every number and diffs it
- the `reproduce` CI job, which does exactly that on every tag
- `make anchor`, the cross-check against an independent implementation

Between them those cover more than most badges certify.
