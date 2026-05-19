# Branch A cohomological Higgs module paper 4

This repository is the public reproducibility package for:

**A Branch A cohomological Higgs module: stabilizer-derived quartic overlap and lightness-protection gates**

The manuscript is a cautious theoretical module paper motivated by the Tau Core
projection framework. It does not claim a completed Standard Model derivation,
a completed theory of everything, or empirical validation of Tau Core. It
isolates a Higgs-sector mechanism with explicit assumptions, sensitivity
audits, and falsification gates.

## Theory Context

The broader Tau Core / projection-theory background is maintained separately at:

```text
https://github.com/tau-core-research/tau-core-theory
```

This Paper 4 repository is a standalone reproducibility package. The source
material imported from the private theory hub is reduced to the minimum needed
to regenerate this manuscript.

## Repository Contents

```text
paper4_submission_source/                         LaTeX source, bibliography, figures, and compiled PDF
figures/                                          Regenerated SVG figures used by the manuscript
studies/tau_core_higgs_module_v01/                Paper 4 regeneration script and seed packet
tests/                                            Public reproducibility checks
arxiv_submission_source.zip                       arXiv-ready source package
```

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the paper source, diagnostic tables, figures, arXiv source ZIP, and
PDF:

```bash
python studies/tau_core_higgs_module_v01/make_paper4_submission_source_v01.py
```

Run the public reproducibility checks:

```bash
python -m pytest -q
```

The generator uses `tectonic` for PDF compilation when available. If `tectonic`
is missing, the TeX source, figures, diagnostic tables, and arXiv ZIP are still
regenerated, but the PDF readiness row records the compiler blocker.

## Main Outputs

- `paper4_submission_source/main.tex`
- `paper4_submission_source/main.pdf`
- `paper4_submission_source/references.bib`
- `paper4_submission_source/figures/*.pdf`
- `figures/*.svg`
- `arxiv_submission_source.zip`
- `studies/tau_core_higgs_module_v01/packet_v01_seed/*.csv`

The packet includes a quartic-overlap sensitivity audit and an explicit
`nu_rule` readiness blocker. The value `I4(3/10)` should therefore be read as
mathematical motivation, not standalone evidence.

## Scope

This repository is a reproducibility package for the Higgs-sector module only.
It should be read as a candidate mechanism and validation roadmap, not as a
claim that the full Tau Core parent theory has been proven.
