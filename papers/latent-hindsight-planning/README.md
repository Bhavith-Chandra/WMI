# Paper 1: hindsight planning

- **`hindsight_planning.tex`**: the current paper (hierarchical planning with and without hindsight, relabeling,
  mechanisms, deployment realism, real time). Appendix: `theory_body.tex`.
- `main.tex`: the earlier version (before the hierarchical, relabeling and deployment experiments); kept for reference.
- `theory.tex`: theory section used by `main.tex`; `theory_body.tex` is the same text without the section header.
- `generated/`: every number, table and figure, produced from `results/` by `scripts/make_paper1.py` and
  `scripts/plot_v*.py`. Never edit by hand.
- `*_preregistration.md`: hypotheses and decision rules committed before each experiment, with outcomes appended.

Rebuild numbers and figures: `python scripts/make_paper1.py && for v in 3 6 7 8 9; do python scripts/plot_v$v.py; done`
(plot_v3 lives in scripts/plot_v3.py). Compile with any LaTeX distribution: `pdflatex hindsight_planning && bibtex
hindsight_planning && pdflatex hindsight_planning && pdflatex hindsight_planning`.
