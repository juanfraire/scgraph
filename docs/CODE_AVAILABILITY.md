# Code-availability statement (draft for the paper)

For Juan to paste into `main.tex`. **Not inserted automatically.** IEEE OJ-COMS
has no mandated section for this; the two usual homes are a short subsection at
the end of Section VII (Evaluation), or immediately before the Acknowledgment.
The `\authornote` at the top is a third option if space is tight.

---

## Preferred version (3 sentences, for submission)

> \section*{Code Availability}
> The \texttt{scgraph} library implementing the stochastic contact graph, the
> SLA-aware routers, and every experiment reported here is openly available at
> \url{https://gitlab.inria.fr/jfraire/scgraph} under the MIT licence.
> All results are seeded: a single command regenerates every figure and table in
> this paper, and a verification target checks the regenerated values against the
> ones reported here.
> The repository also contains the orbital contact plans used as the space layer,
> so the evaluation reproduces without a contact-plan design tool installed, and
> the cross-check that anchors our deterministic router against the A-SABR
> framework.

At submission, replace the URL with the tagged revision once
`v0.1.0-submission` exists, or add: *"(tag \texttt{v0.1.0-submission}, commit
\texttt{abc1234})"*.

---

## Short version (2 sentences, if space is tight)

> \section*{Code Availability}
> The \texttt{scgraph} library and all experiments are openly available at
> \url{https://gitlab.inria.fr/jfraire/scgraph} under the MIT licence, including
> the orbital contact plans used as the space layer.
> All results are seeded and regenerate with a single command, with a
> verification target that checks them against the values reported here.

---

## Camera-ready version (after acceptance, with a DOI)

> \section*{Code Availability}
> The \texttt{scgraph} library implementing the stochastic contact graph, the
> SLA-aware routers, and every experiment reported here is openly available at
> \url{https://gitlab.inria.fr/jfraire/scgraph} under the MIT licence and
> archived at \mbox{doi:10.5281/zenodo.XXXXXXX}.
> All results are seeded: a single command regenerates every figure and table in
> this paper, and a verification target checks the regenerated values against the
> ones reported here.
> The repository also contains the orbital contact plans used as the space layer,
> so the evaluation reproduces without a contact-plan design tool installed, and
> the cross-check that anchors our deterministic router against the A-SABR
> framework.

---

## Notes on wording

- Say **"openly available"**. "Available on request" is treated as
  unavailable by most reproducibility policies, and would be false here anyway.
- The claim *"regenerates every figure and table"* is literally true and worth
  making, since it is stronger than what most papers can say. It is backed by
  `make reproduce` and `make verify`, and by the `reproduce` CI job on every tag.
- The **contact plans** are called out deliberately: the commonest reason a
  networking artefact fails to reproduce is a missing input dataset. Saying they
  are included pre-empts the question.
- The **A-SABR anchor** is mentioned because Section VII-A already claims the
  agreement; the statement tells a reviewer where to check it. Do not overclaim
  it as a dependency, since the anchor is optional for reproducing the results.
- Do not claim the library is "general-purpose" or "a framework". It is a paper
  companion, and §VIII already scopes what is future work.

## Authorship

Settled: the artefact follows `main.tex`'s **four** authors, in its order.

> Juan A. Fraire, Isabelle Guérin Lassous, Thomas Begin, Emmanuel Lavinal

Fernandez and Iova are commented out in the `\author` block and are therefore
not listed here, even though `materials/PLAN.md` still records the earlier
six-author plan. `CITATION.cff` and the README's two BibTeX entries all match
the list above.

If the author block changes before submission, three places move together:
`main.tex`, `CITATION.cff` (both the `authors:` list and the
`preferred-citation.authors:` list), and the README's "How to cite" section.
