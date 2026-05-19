# Branch A cohomological Higgs module paper 4

This repository is the public reproducibility package for:

**A Branch A cohomological Higgs module: stabilizer-derived quartic overlap and lightness-protection gates**

Archived reproducibility package DOI: [10.5281/zenodo.20285909](https://doi.org/10.5281/zenodo.20285909)

The manuscript is a cautious Branch A projection/cohomology Higgs-module paper.
It does not claim a completed Standard Model derivation, a completed theory of
everything, or empirical validation of the broader parent framework. It isolates
a Higgs-sector mechanism with explicit assumptions, sensitivity audits, and
falsification gates.

## Author And Research Workflow

I am an independent researcher using an AI-assisted workflow to develop reproducible diagnostic tests around projection-sensitive residual hypotheses. I am not claiming expert-level validation. I would value criticism on whether the proposed gate/falsification structure is scientifically meaningful.

AI systems are used for drafting, mathematical organization, code generation, literature triage, and internal consistency checks. Numerical and symbolic audits can support reproducibility and error-finding, but they do not replace independent expert review or physical validation.

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
studies/tau_core_higgs_module_v01/                Paper 4 regeneration script
wolfram/                                         Optional Wolfram Language audit scripts
tests/                                            Public reproducibility checks
```

Generated manuscript outputs are intentionally not tracked. Running the
regeneration command creates `paper4_submission_source/`, regenerated SVG
figures in `figures/`, packet CSVs under
`studies/tau_core_higgs_module_v01/packet_v01_seed/`, and
`arxiv_submission_source.zip`.

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

Optional Wolfram Language checks can be run when Wolfram Engine is activated:

```bash
wolframscript -file wolfram/Higgs_Quartic_Overlap_Verification.wl
wolframscript -file wolfram/BranchA_Stabilizer_Hypercharge_Audit.wl
wolframscript -file wolfram/G2_Unoriented_Line_Quotient_Audit.wl
wolframscript -file wolfram/Projection_BRST_Skeleton.wl
```

These scripts verify the symbolic/numeric overlap integral, the Branch A
stabilizer/hypercharge normalization, the unoriented-line quotient consequence,
and the BRST skeleton. They do not prove the `nu_i=3|Y_i|/5` rule, anomaly
freedom, regulator safety, or radiative stability.

## Main Outputs

The following files are generated locally by the reproduction command:

- `paper4_submission_source/main.tex`
- `paper4_submission_source/main.pdf`
- `paper4_submission_source/references.bib`
- `paper4_submission_source/figures/*.pdf`
- `figures/*.svg`
- `arxiv_submission_source.zip`
- `studies/tau_core_higgs_module_v01/packet_v01_seed/*.csv`
- `studies/tau_core_higgs_module_v01/packet_v01_seed/wolfram_audit_logs/*.log`

The packet includes a quartic-overlap sensitivity audit and an explicit
`nu_rule` readiness blocker. The value `I4(3/10)` should therefore be read as
mathematical motivation, not standalone evidence.

## Scope

This repository is a reproducibility package for the Higgs-sector module only.
It should be read as a candidate mechanism and validation roadmap, not as a
claim that the full parent projection theory has been proven.
