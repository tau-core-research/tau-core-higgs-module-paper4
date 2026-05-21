import csv
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper4_submission_source"
PACKET = ROOT / "studies/tau_core_higgs_module_v01/packet_v01_seed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_regeneration_script_runs():
    result = subprocess.run(
        ["python", "studies/tau_core_higgs_module_v01/make_paper4_submission_source_v01.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paper4_submission_source/main.tex" in result.stdout


def test_publication_package_files_exist():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "DATA_NOTICE.md",
        ROOT / "requirements.txt",
        PAPER / "main.tex",
        PAPER / "references.bib",
        ROOT / "paper4_full_derivation/full_derivation.tex",
        ROOT / "paper4_full_derivation/full_derivation.pdf",
        PAPER / "figures/paper4_higgs_zero_mode_profiles.pdf",
        PAPER / "figures/paper4_mechanism_flow.pdf",
        PAPER / "figures/paper4_quartic_overlap_curve.pdf",
        PAPER / "figures/paper4_top_mass_derivative_toy_stress.pdf",
        ROOT / "paper4_full_derivation/figures/paper4_top_mass_derivative_toy_stress.pdf",
        ROOT / "figures/paper4_higgs_zero_mode_profiles.svg",
        ROOT / "figures/paper4_mechanism_flow.svg",
        ROOT / "figures/paper4_quartic_overlap_curve.svg",
        ROOT / "figures/paper4_top_mass_derivative_toy_stress.svg",
        ROOT / "arxiv_submission_source.zip",
        ROOT / "wolfram/Higgs_Quartic_Overlap_Verification.wl",
        ROOT / "wolfram/BranchA_Stabilizer_Hypercharge_Audit.wl",
        ROOT / "wolfram/G2_Unoriented_Line_Quotient_Audit.wl",
        ROOT / "wolfram/Projection_BRST_Skeleton.wl",
        ROOT / "wolfram/Compact_Gate_Ledger_Audit.wl",
        PACKET / "paper4_higgs_overlap_scan_v01.csv",
        PACKET / "paper4_quartic_sensitivity_audit_v01.csv",
        PACKET / "paper4_compact_spectrum_pilot_v01.csv",
        PACKET / "paper4_q_paired_spectrum_demo_v01.csv",
        PACKET / "paper4_anomaly_bridge_audit_v01.csv",
        PACKET / "paper4_top_mass_derivative_toy_trace_v01.csv",
        PACKET / "paper4_higgs_module_summary_v01.csv",
        PACKET / "paper4_readiness_table_v01.csv",
        PACKET / "wolfram_audit_logs/Higgs_Quartic_Overlap_Verification.log",
        PACKET / "wolfram_audit_logs/BranchA_Stabilizer_Hypercharge_Audit.log",
        PACKET / "wolfram_audit_logs/G2_Unoriented_Line_Quotient_Audit.log",
        PACKET / "wolfram_audit_logs/Projection_BRST_Skeleton.log",
        PACKET / "wolfram_audit_logs/Compact_Gate_Ledger_Audit.log",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_core_numbers_are_reproducible():
    summary = {row["quantity"]: row for row in read_csv(PACKET / "paper4_higgs_module_summary_v01.csv")}
    assert summary["nu_D"]["value"] == "0.300000000"
    assert abs(float(summary["I4_nu_D"]["value"]) - 0.133756) < 1e-6
    assert summary["Branch_A_metric_factor"]["interpretation"].startswith("assumption")


def test_submission_source_claim_boundaries():
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "not a completed Standard Model derivation" in tex
    assert "not a proof" in tex
    assert "of any parent projection theory" in tex
    assert "not an empirical claim" in tex
    assert "not solve the Higgs hierarchy problem" in tex
    assert "not yet prove radiative stability or solve the hierarchy problem" in tex
    assert "Wolfram Language Audit" in tex
    assert "They verify the normalization source of $3/5$" in tex
    assert "Spectral origin of the normalization" in tex
    assert "a heat-kernel/index residue of the protected hypercharge line" in tex
    assert "they do not derive the localization postulate" in tex
    assert "Under the stated Branch A rule" in tex
    assert "Branch A projection/cohomology Higgs module" in tex
    assert "Theorem Status Summary" in tex
    assert "detailed derivation ledger" in tex
    assert "Related Mathematical Structures" in tex
    assert "spectral triples and spectral action" in tex
    assert "domain-wall and defect localization" in tex
    assert "BRST and cohomological control" in tex
    assert "does not claim priority over, equivalence with, or derivation from these" in tex
    assert "Single-Package Parent-Action Update" in tex
    assert "y_t^{\\rm parent}=0.9615522319" in tex
    assert "\\epsilon_\\tau$ scale gate" in tex
    assert "Parent-Selection Refinements" in tex
    assert "two protected visible clusters" in tex
    assert "\\epsilon_3\\text{ closure}+\\epsilon_2\\text{ pairing}" in tex
    assert "invariant/anomaly bridge" in tex
    assert "not representation derivation" in tex
    assert "unoriented $T_Y$ line quotient" in tex
    assert "paper4_mechanism_flow.pdf" in tex
    assert "Candidate Route To The Localization Rule" in tex
    assert "Stabilizer Origin Of The Hypercharge Direction" in tex
    assert "Minimal Wall Profile And Quotient Vacua" in tex
    assert "Charge-to-exponent gate" in tex
    assert "The remaining nontrivial step is to derive the wall connection coefficient" in tex
    assert "A_{\\tau,i}(x)=\\kappa_\\tau^2 Y_i \\tanh x" in tex
    assert "\\nu_i=\\left|\\kappa_\\tau^2 Y_i\\right|=\\kappa_\\tau^2|Y_i|" in tex
    assert "\\operatorname{span}(T_Y)=\\operatorname{span}(-T_Y)" in tex
    assert "\\Sigma\\sim-\\Sigma" in tex
    assert "G2\\_Unoriented\\_Line\\_Quotient\\_Audit.wl" in tex
    assert "Compact\\_Gate\\_Ledger\\_Audit.wl" in tex
    assert "there is only one Abelian generator" in tex
    assert "f(x)=\\tanh x" in tex
    assert "Why This Is Not Just Numerology" in tex
    assert "Remaining Proof Gates" in tex
    assert "Next Theorem Handoff" in tex
    assert "protected-form selection" in tex
    assert "zeta determinant and heat-kernel methods" in tex
    assert "A future paper can close one row" in tex
    assert "Claim-Upgrade Ladder" in tex
    assert "At the present stage the paper sits at the first level" in tex
    assert "Detailed Gate Ledger And Companion Scope" in tex
    assert "compact gate ledger for the main" in tex
    assert "Localization derivation" in tex
    assert "BRST/anomaly/regulator consistency" in tex
    assert "Top determinant and radiative stability" in tex
    assert "paper4_top_mass_derivative_toy_stress.pdf" in tex
    assert "Toy mass-derivative stress test for the top determinant gate" in tex
    assert "regulator-sensitive residue" in tex
    assert "Compact Spectral Protection Program" in tex
    assert "Q\\text{-paired self-adjoint operators}" in tex
    assert "zeta-determinant cancellation" in tex
    assert "This is why the compact spectral program is central to the paper" in tex
    assert "the construction is only a localization model" in tex
    assert "a conditional compact spectral theorem is established only within" in tex
    assert "the parent action has not yet selected the" in tex
    assert "required compact domain" in tex
    assert "conditional positive-spectrum zeta-determinant cancellation follows" in tex
    assert "physical top finite-residue extraction gate" in tex
    assert "finite residue has not yet been computed from the parent" in tex
    assert "QFT" in tex
    assert "same-domain anomaly/Ward and representation trace-cancellation gates are" in tex
    assert "parent-derived representation complex has not yet been" in tex
    assert "a UV/continuum admissibility criterion and microscopic completion target" in tex
    assert "no convergent parent action family is supplied" in tex
    assert "a physical matching map and endpoint protocol are formulated" in tex
    assert "numerical matching to the measured Higgs vev" in tex
    assert "matching to the measured Higgs vev" in tex
    assert "C0 Candidate: Finite-Fiber Wall Spectral Cell" in tex
    assert "Adversarial Audit Of The C0 Candidate" in tex
    assert "C1 Forcing Route For The Inserted Structures" in tex
    assert "C2 Closure Route For Projector, Determinant, And Scale" in tex
    assert "C3 Explicit Compact Spectral Geometry Candidate" in tex
    assert "Adversarial Audit Of The C3 Candidate" in tex
    assert "C4 Selection-Functional Target" in tex
    assert "I_{\\rm compact}" in tex
    assert "all six penalties vanish" in tex
    assert "C5 Local Spectrum Gate" in tex
    assert "C6 Compact Spectrum Boundary-Value Problem" in tex
    assert "C7 Competitor Audit For The C4 Minimum" in tex
    assert "C8 Minimality Theorem Candidate" in tex
    assert "C9 Compact Spectrum Pilot" in tex
    assert "C10 Uniqueness Proof-Obligation Split" in tex
    assert "C11 Algebraic Minimality Lemmas" in tex
    assert "C12 Protected-Form Selection Gate" in tex
    assert "C13 Cohomology/Index Selection Route" in tex
    assert "C14 Toy Complex Sanity Check" in tex
    assert "C15 Operator-Domain Ansatz" in tex
    assert "C16 Shared-Domain Compatibility Gate" in tex
    assert "C17 Self-Adjoint Domain Candidate" in tex
    assert "C18 Paired-Spectrum Verification Target" in tex
    assert "positive-spectrum cancellation target" in tex
    assert "C20 Residue Classification Gate" in tex
    assert "H_-&=Q_\\nu^\\dagger Q_\\nu" in tex
    assert "{\\cal K}_\\tau=S^1/\\mathbb Z_2" in tex
    assert "Compact Tau Geometry As The Next Central Gate" in tex
    assert "{\\cal K}_\\tau=S^1/\\mathbb Z_2" in tex
    assert "This is not yet the final parent action" in tex
    assert "orbifold Q-domain theorem" in tex
    assert "\\left[\\psi^*\\phi\\right]_0^L=0" in tex
    assert "finite-fiber wall spectral cell" in tex
    assert "\\zeta_{D_\\tau}(s)" in tex
    assert "Near-Term Falsifiable Prediction" in tex
    assert "I_4(3/10)" in tex
    assert "\\includegraphics" in tex


def test_numerology_and_nu_rule_gates_are_explicit():
    sensitivity = read_csv(PACKET / "paper4_quartic_sensitivity_audit_v01.csv")
    readiness = {row["Item"]: row for row in read_csv(PACKET / "paper4_readiness_table_v01.csv")}
    summary = {row["quantity"]: row for row in read_csv(PACKET / "paper4_higgs_module_summary_v01.csv")}
    assert any(row["band"] == "moderate" for row in sensitivity)
    assert readiness["nu_rule"]["Status"] == "conditional_forcing_route"
    assert "F1-F8" in readiness["nu_rule"]["Detail"]
    assert readiness["branch_a_parent_selection"]["Status"] == "conditional_gate_chain"
    assert summary["parent_selection_gate_count"]["value"] == "5"
    assert summary["localization_forcing_update"]["value"] == "F1-F8"
    assert summary["compact_tau_geometry_candidate"]["value"] == "C0_finite_fiber_wall_spectral_cell"
    assert summary["c0_attack_status"]["value"] == "survives_as_candidate"
    assert summary["c1_forcing_update"]["value"] == "two_cluster_plus_bogomolny_route"
    assert summary["c2_closure_update"]["value"] == "projector_determinant_scale_route"
    assert summary["c3_explicit_geometry_candidate"]["value"] == "orbifold_interval_with_finite_two_role_fiber"
    assert summary["c3_attack_status"]["value"] == "explicit_but_selection_not_derived"
    assert summary["c4_selection_functional"]["value"] == "six_penalty_parent_selection_target"
    assert summary["c5_spectrum_gate"]["value"] == "local_poschl_teller_spectrum"
    assert summary["nu_D_continuum_threshold"]["value"] == "0.090000000"
    assert summary["c6_compact_spectrum_gate"]["value"] == "orbifold_boundary_quantization_problem"
    assert summary["c7_competitor_audit"]["value"] == "minimal_two_role_ladder"
    assert summary["c8_minimality_theorem_candidate"]["value"] == "two_role_no_free_spectrum_gate"
    assert summary["c9_compact_spectrum_pilot"]["value"] == "finite_box_dirichlet_eigenvalue_audit"
    assert summary["q_paired_spectrum_demo"]["value"] == "same_domain_q_first_toy_pairing"
    assert summary["c10_uniqueness_proof_obligations"]["value"] == "algebraic_vs_parent_action_split"
    assert summary["c11_algebraic_minimality_lemmas"]["value"] == "rank3_closure_rank2_pairing_targets"
    assert summary["c12_protected_form_selection_gate"]["value"] == "parent_selected_exterior_forms_required"
    assert summary["c13_cohomology_index_selection_route"]["value"] == "first_residue_form_selection_candidate"
    assert summary["c14_toy_complex_sanity_check"]["value"] == "koszul_like_residue_model"
    assert summary["c15_operator_domain_ansatz"]["value"] == "differential_adjoint_laplacian_index_target"
    assert summary["c16_shared_domain_gate"]["value"] == "wall_index_regulator_determinant_same_domain"
    assert summary["c17_self_adjoint_domain_candidate"]["value"] == "orbifold_parity_q_pairing_boundary_candidate"
    assert summary["c18_paired_spectrum_verification_target"]["value"] == "positive_mode_bijection_and_zero_residue"
    assert summary["c19_zeta_determinant_cancellation_target"]["value"] == "positive_spectrum_cancellation_index_residue"
    assert summary["c20_residue_classification_gate"]["value"] == "allowed_vs_forbidden_determinant_residues"
    assert readiness["c0_adversarial_audit"]["Status"] == "candidate_survives_but_not_forced"
    assert readiness["c1_forcing_route"]["Status"] == "theorem_candidate"
    assert readiness["c2_closure_route"]["Status"] == "theorem_candidate"
    assert readiness["c3_explicit_geometry"]["Status"] == "explicit_candidate_not_final_action"
    assert readiness["c3_adversarial_audit"]["Status"] == "explicit_inputs_exposed"
    assert readiness["c4_selection_functional"]["Status"] == "formulated_not_minimized"
    assert readiness["c5_spectrum_gate"]["Status"] == "local_spectrum_derived_compact_spectrum_open"
    assert readiness["c6_compact_spectrum_gate"]["Status"] == "compact_problem_formulated_not_solved"
    assert readiness["c7_competitor_audit"]["Status"] == "formulated_not_exhaustive"
    assert readiness["c8_minimality_theorem_candidate"]["Status"] == "conditional_not_proven"
    assert readiness["c9_compact_spectrum_pilot"]["Status"] == "numerical_pilot_not_theorem"
    assert readiness["c10_uniqueness_proof_obligations"]["Status"] == "proof_stack_split_not_closed"
    assert readiness["c11_algebraic_minimality_lemmas"]["Status"] == "lemma_formulated_not_fully_proven"
    assert readiness["c12_protected_form_selection_gate"]["Status"] == "selection_gate_formulated"
    assert readiness["c13_cohomology_index_selection_route"]["Status"] == "candidate_route_not_closed"
    assert readiness["c14_toy_complex_sanity_check"]["Status"] == "toy_model_not_parent_derivation"
    assert readiness["c15_operator_domain_ansatz"]["Status"] == "operator_target_not_solved"
    assert readiness["c16_shared_domain_gate"]["Status"] == "compatibility_gate_formulated"
    assert readiness["c17_self_adjoint_domain_candidate"]["Status"] == "candidate_not_classification"
    assert readiness["c18_paired_spectrum_verification_target"]["Status"] == "theorem_target_not_verified"
    assert readiness["c19_zeta_determinant_cancellation_target"]["Status"] == "determinant_target_not_proven"
    assert readiness["c20_residue_classification_gate"]["Status"] == "classification_gate_formulated"
    assert "not evidence by itself" in sensitivity[0]["interpretation"]


def test_full_derivation_companion_is_generated():
    tex = (ROOT / "paper4_full_derivation/full_derivation.tex").read_text(encoding="utf-8")
    assert "Full Derivation Ledger for the Branch A Higgs Module" in tex
    assert "F1--F8 Forcing Theorem Candidate" in tex
    assert "Yukawa Product Cohomology Gate" in tex
    assert "Physical Matching Admissibility Map" in tex
    assert "(v_{\\rm obs},m_H^{\\rm obs},m_t^{\\rm obs})" in tex
    assert "forbidden reverse map" in tex
    assert "Numerical Endpoint Matching Protocol" in tex
    assert "\\Theta_{\\rm pred}" in tex
    assert "\\chi_{\\rm end}^2" in tex
    assert "derive, freeze, run, compare" in tex
    assert "Charge-To-Exponent Theorem Target" in tex
    assert "Spectral Origin Of The Normalization Gate" in tex
    assert "the factor $3/5$ becomes a spectral number" in tex
    assert "\\delta A_i" in tex
    assert "Q-exact, quotient-null, or massive/leakage-suppressed" in tex
    assert "y_t^{\\rm parent}" in tex
    assert "\\epsilon_\\tau$ Origin Gate" in tex
    assert "C0 Finite-Fiber Wall Spectral Cell" in tex
    assert "Adversarial Audit Of C0" in tex
    assert "C1 And C2 Forcing Routes" in tex
    assert "C3 Explicit Geometry Candidate" in tex
    assert "C3.1 Compact Geometry v0.1 Computation Target" in tex
    assert "C3.1 Master Compact Spectral Theorem Statement" in tex
    assert "This is the real breakthrough theorem" in tex
    assert "{\\rm Spec}_+(H_-)&={\\rm Spec}_+(H_+)" in tex
    assert "parent-selection and anomaly-compatible domain closure" in tex
    assert "C3.1a Orbifold Q-Domain And Index Theorem Target" in tex
    assert "\\langle Q_\\nu\\psi,\\phi\\rangle_{+}" in tex
    assert "C3.1b Proof Of The Q-Pairing Lemma" in tex
    assert "This proves the compact-index algebraic core" in tex
    assert "\\ker H_-=\\ker Q" in tex
    assert "C3.1c Explicit Relative-Orbifold Domain Proof" in tex
    assert "{\\cal D}(Q_\\nu)=H^1([0,L])" in tex
    assert "{\\rm ind}(Q_\\nu)=1" in tex
    assert "such a compact Fredholm" in tex
    assert "C3.1c$'$ Q-Compatible Orbifold Extension Classification" in tex
    assert "\\mathcal B_-=" in tex
    assert "\\mathcal B_+=\\{0\\}" in tex
    assert "first minimal survivor" in tex
    assert "C3.1d Conditional Compact Self-Adjoint Spectrum Theorem" in tex
    assert "{\\rm Spec}_+(H_-)= {\\rm Spec}_+(H_+)" in tex
    assert "\\det_\\zeta H_-^{\\rm pos}" in tex
    assert "\\log\\Delta_{\\rm anomaly}" in tex
    assert "C3.1d$'$ Protected Index Residue Theorem" in tex
    assert "\\ker H_-=\\operatorname{span}\\{\\psi_0\\}" in tex
    assert "{\\rm ind}(Q_\\nu)=1" in tex
    assert "with no allowed positive-mode bulk determinant residue" in tex
    assert "\\log\\Delta_i=-\\zeta_i^{-}\\prime(0)+\\zeta_i^{+}\\prime(0)" in tex
    assert "C3.1e Compact-Resolvent Check For The Relative-Orbifold Domain" in tex
    assert "regular Sturm--Liouville operators" in tex
    assert "(\\partial_x+\\nu f)\\psi\\big|_{0,L}=0" in tex
    assert "assumption in C3.1d is not an extra fitting freedom" in tex
    assert "C3.1f Variational Boundary-Selection Lemma" in tex
    assert "S_{\\rm dom}[\\psi,\\phi]" in tex
    assert "positive orbifold boundary" in tex
    assert "derive the existence, coefficient" in tex
    assert "C3.1g Thin-Barrier Origin Of The Boundary Obstruction" in tex
    assert "\\rho_\\epsilon\\rightharpoonup \\delta_0+\\delta_L" in tex
    assert "sharp-orbifold limit of a positive" in tex
    assert "the defect profile $\\rho_\\epsilon$" in tex
    assert "C3.1h Minimal Defect-Measure Selection Lemma" in tex
    assert "\\mu_b=a\\,\\delta_0+b\\,\\delta_L" in tex
    assert "\\mu_b=\\delta_0+\\delta_L" in tex
    assert "equal weights are forced" in tex
    assert "C3.1i Coupling-Independence Of The Protected Domain" in tex
    assert "Changing $M_b$ changes the gap" in tex
    assert "no longer a domain-selection parameter" in tex
    assert "C3.1j Action-Origin Lemma For The Leakage Term" in tex
    assert "S_{\\rm bdry}" in tex
    assert "Variation of a non-negative quadratic action" in tex
    assert "deriving $S_{\\rm bdry}$ itself" in tex
    assert "C3.1k Compact-Cell To Boundary Reduction Target" in tex
    assert "S_{\\rm cell,leak}" in tex
    assert "transverse gap/stiffness residue" in tex
    assert "S_{\\rm cell}" in tex
    assert "C3.1l Tubular-Neighborhood Collapse Lemma" in tex
    assert "r(x)=\\operatorname{dist}(x,F)" in tex
    assert "\\rho_\\epsilon(x)\\,dx\\rightharpoonup \\delta_0+\\delta_L" in tex
    assert "coarea/tubular-neighborhood reduction" in tex
    assert "C3.1m Stable-Cell Hessian Origin Of Leakage Energy" in tex
    assert "\\delta^2S_{\\rm cell}" in tex
    assert "gapped leakage complement" in tex
    assert "the compact spectral theorem route fails" in tex
    assert "C3.1n Explicit Toy Compact-Cell Background And Hessian" in tex
    assert "\\Psi_0(x)=f_0(x)T_Y" in tex
    assert "L_{\\rm wall}" in tex
    assert "{\\rm Spec}(L_{\\rm leak})\\subseteq[m_\\perp^2,\\infty)" in tex
    assert "C3.2 What The v0.1 Geometry Would Force" in tex
    assert "Adversarial Audit Of C3" in tex
    assert "C4 Penalty Interpretation" in tex
    assert "C4.1 Parent-Selection Lower-Bound Lemma Candidate" in tex
    assert "prove a lower bound excluding all lower-complexity competitors" in tex
    assert "single explicit competitor with ${\\cal S}_{\\rm sel}=0" in tex
    assert "C4.2 No-Free-Weights Selection Rule" in tex
    assert "lexicographic rather than tunably weighted" in tex
    assert "No continuous coefficient may be" in tex
    assert "C5 Local Spectrum Gate" in tex
    assert "C6 Compact Spectrum Boundary-Value Problem" in tex
    assert "C7 Competitor Audit" in tex
    assert "C8 Minimality Theorem Candidate" in tex
    assert "C9 Compact Spectrum Pilot" in tex
    assert "C9.1 Q-Paired Spectrum Toy Demonstrator" in tex
    assert "paper4\\_q\\_paired\\_spectrum\\_demo\\_v01.csv" in tex
    assert "singular-value decomposition" in tex
    assert "C10 Uniqueness Proof-Obligation Split" in tex
    assert "C11 Algebraic Minimality Lemmas" in tex
    assert "C12 Protected-Form Selection Gate" in tex
    assert "C13 Cohomology/Index Selection Route" in tex
    assert "C14 Toy Complex Sanity Check" in tex
    assert "C14.1 Anomaly-Bridge Closure Gate" in tex
    assert "{\\cal A}_{ABC}" in tex
    assert "bridge-closed, trace-neutral, Q-compatible visible content" in tex
    assert "without adding endpoint-tuned spectator fields" in tex
    assert "C14.2 Bridge-Trace Normalization Gate" in tex
    assert "g_t^{\\rm bridge}" in tex
    assert "6\\kappa_\\tau^2=\\frac{18}{5}" in tex
    assert "different bridge multiplicity or additional light" in tex
    assert "paper4\\_anomaly\\_bridge\\_audit\\_v01.csv" in tex
    assert "C15 Operator-Domain Ansatz" in tex
    assert "C16 Shared-Domain Compatibility Gate" in tex
    assert "C16.1 Anomaly-Compatible Domain Closure Criterion" in tex
    assert "G_A\\mathcal D_{\\rm rel}\\subseteq\\mathcal D_{\\rm rel}" in tex
    assert "\\operatorname{Tr}_{\\mathcal D_{\\rm rel}}" in tex
    assert "not compatible with the compact spectral theorem" in tex
    assert "C16.2 Same-Domain Ward Identity Gate" in tex
    assert "C16.4 Representation Trace-Cancellation Gate" in tex
    assert "{\\cal T}_{ABC}^{\\rm parent}" in tex
    assert "parent-derived representation complex is not yet" in tex
    assert "C16.3 UV/Continuum Admissibility Criterion" in tex
    assert "converges to a closed Fredholm" in tex
    assert "strong-resolvent sense" in tex
    assert "construct a regulator family" in tex
    assert "C16.3a Microscopic Continuum Completion Target" in tex
    assert "stationary backgrounds" in tex
    assert "but the target is now" in tex
    assert "mathematically explicit" in tex
    assert "\\delta_A S_{\\rm eff}^\\Lambda" in tex
    assert "{\\cal B}_A^\\Lambda" in tex
    assert "{\\cal O}_{\\rm tr}(\\Lambda^{-1})" in tex
    assert "formal pairing artifact" in tex
    assert "C17 Self-Adjoint Domain Candidate" in tex
    assert "{\\cal D}(H_-)" in tex
    assert "inherited from the first-order closed operator pair" in tex
    assert "C17.1 Parent-Domain Selection Functional" in tex
    assert "{\\cal S}_{\\rm dom}" in tex
    assert "unique zero-obstruction" in tex
    assert "lower-complexity zero-obstruction competitor" in tex
    assert "C18 Paired-Spectrum Verification Target" in tex
    assert "Q_\\nu:\\ker(H_- -\\lambda)\\longrightarrow\\ker(H_+-\\lambda)" in tex
    assert "This proves the abstract pairing lemma" in tex
    assert "C19 Conditional Zeta-Determinant Cancellation Lemma" in tex
    assert "\\zeta_\\pm^{\\rm pos}(s)" in tex
    assert "positive-spectrum determinant" in tex
    assert "same-domain setup" in tex
    assert "C20 Residue Classification Gate" in tex
    assert "C20.1 Regulator-Independence Residue Test" in tex
    assert "\\Delta_{12}^{\\rm pos}" in tex
    assert "\\Pi_-^{\\rm pos}-\\Pi_+^{\\rm pos}" in tex
    assert "Any regulator-dependent scalar mass term outside this span is a failure mode" in tex
    assert "Top Determinant No-Mass-Rescue Gate" in tex
    assert "\\Delta\\Gamma_t^{\\rm forbidden}" in tex
    assert "\\Pi_{\\rm phys}\\Delta\\Gamma_t^{\\rm forbidden}=0" in tex
    assert "may not hide the hierarchy problem" in tex
    assert "Top Determinant Mass-Derivative Test" in tex
    assert "\\mu_t^2(\\delta)" in tex
    assert "\\frac{\\partial^2\\Gamma_t^\\Lambda}{\\partial H^\\dagger\\partial H}" in tex
    assert "compute this same-domain trace" in tex
    assert "paper4\\_top\\_mass\\_derivative\\_toy\\_trace\\_v01.csv" in tex
    assert "paired same-domain case" in tex
    assert "mismatch-sensitivity scan" in tex
    assert "exponential and rational" in tex
    assert "regulator choice should be invisible" in tex
    assert "paper4_top_mass_derivative_toy_stress.pdf" in tex
    assert "Toy top-mass derivative stress test" in tex
    assert "not used as independent evidence for radiative stability" in tex
    assert "Top Determinant Finite-Residue Extraction Gate" in tex
    assert "R_t^{\\rm fin}" in tex
    assert "\\mu_{t,{\\rm fin}}^2" in tex
    assert "finite-residue extraction gate" in tex
    assert "Negative status at closure" in tex
    assert "conditional compact spectral theorem and" in tex
    assert "within stated operator-domain" in tex
    assert "parent-domain selection functional formulated" in tex
    assert "exhaustive domain" in tex
    assert "positive-spectrum determinant cancellation follows" in tex
    assert "finite-residue extraction gate is formulated" in tex
    assert "no computed" in tex
    assert "physical top finite residue" in tex
    assert "conditional anomaly/Ward and representation trace-cancellation gates" in tex
    assert "no computed parent-derived representation complex" in tex
    assert "UV/continuum admissibility criterion and microscopic completion target" in tex
    assert "no convergent parent action family" in tex
    assert "physical matching" in tex
    assert "endpoint protocol formulated" in tex
    assert "no numerical endpoint matching yet" in tex
    assert "0.679920103656139" in tex
    assert "completed Higgs-sector proof" in tex
    assert (ROOT / "paper4_full_derivation/full_derivation.pdf").stat().st_size > 10_000


def test_compact_spectrum_pilot_is_generated_and_guardrailed():
    rows = read_csv(PACKET / "paper4_compact_spectrum_pilot_v01.csv")
    assert len(rows) >= 20
    assert any(row["operator"] == "H_minus" for row in rows)
    assert any(row["operator"] == "H_plus" for row in rows)
    assert any(row["operator"] == "pairing_residual" for row in rows)
    assert all("not Q-paired theorem spectrum" in row["interpretation"] or "pilot domain" in row["interpretation"] for row in rows)


def test_q_paired_spectrum_demo_shows_same_domain_pairing():
    rows = read_csv(PACKET / "paper4_q_paired_spectrum_demo_v01.csv")
    assert any(row["quantity"] == "q_shape" for row in rows)
    assert any(row["quantity"] == "index_residue" and row["value"] == "1" for row in rows)
    residual = next(float(row["value"]) for row in rows if row["quantity"] == "max_pairing_residual")
    assert residual < 1e-10
    assert any("Q-first toy pairing" in row["interpretation"] for row in rows)


def test_anomaly_bridge_audit_is_frozen_and_falsifiable():
    rows = read_csv(PACKET / "paper4_anomaly_bridge_audit_v01.csv")
    cases = {row["case"]: row for row in rows}
    assert cases["c3_single_bridge_target"]["status"] == "candidate_survivor"
    assert cases["c3_single_bridge_target"]["bridge_multiplicity"] == "6"
    assert cases["bridge_removed"]["status"] == "fail_cohomology_bridge"
    assert cases["extra_unpaired_light_doublet"]["status"] == "fail_or_requires_new_parent_partner"
    assert cases["different_bridge_multiplicity"]["status"] == "falsifies_18_over_5_bridge_normalization"
    assert all(row["guardrail"] == "higgs_module_candidate_not_standard_model_derivation" for row in rows)


def test_top_mass_derivative_toy_trace_exposes_failure_mode():
    rows = read_csv(PACKET / "paper4_top_mass_derivative_toy_trace_v01.csv")
    summaries = {row["case"]: row for row in rows if row["mode"] == "summary"}
    paired = float(summaries["paired_same_domain_summary"]["mass_curvature_difference"])
    unpaired = float(summaries["unpaired_regulator_failure_summary"]["mass_curvature_difference"])
    assert abs(paired) < 1e-12
    assert abs(unpaired) > 1e-6
    assert summaries["paired_same_domain_summary"]["status"] == "passes_toy_no_mass_rescue"
    assert summaries["unpaired_regulator_failure_summary"]["status"] == "fails_toy_no_mass_rescue"
    scan = [row for row in rows if row["case"] == "mismatch_sensitivity_scan"]
    assert len(scan) == 5
    scan_abs = [abs(float(row["mass_curvature_difference"])) for row in scan]
    assert all(value > 0 for value in scan_abs)
    assert scan_abs == sorted(scan_abs)
    cutoff_paired = [row for row in rows if row["case"] == "cutoff_regulator_scan_paired"]
    cutoff_shifted = [row for row in rows if row["case"] == "cutoff_regulator_scan_shifted"]
    assert len(cutoff_paired) == 8
    assert len(cutoff_shifted) == 8
    assert all(abs(float(row["mass_curvature_difference"])) < 1e-12 for row in cutoff_paired)
    shifted_values = {round(float(row["mass_curvature_difference"]), 12) for row in cutoff_shifted}
    assert len(shifted_values) > 2
    assert all(abs(float(row["mass_curvature_difference"])) > 1e-8 for row in cutoff_shifted)
    assert all(row["guardrail"] == "higgs_module_candidate_not_standard_model_derivation" for row in rows)


def test_arxiv_zip_is_source_only():
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip") as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "references.bib" in names
    assert "main.pdf" not in names
    assert "figures/paper4_higgs_zero_mode_profiles.pdf" in names
    assert "figures/paper4_mechanism_flow.pdf" in names
    assert "figures/paper4_quartic_overlap_curve.pdf" in names
    assert "figures/paper4_top_mass_derivative_toy_stress.pdf" in names
    assert not any(name.endswith((".aux", ".log", ".out")) for name in names)


def test_no_private_theory_dump():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not any(path.startswith("source_material/") for path in tracked)
    assert not any("tau-core-theory" in path for path in tracked)


def test_generated_outputs_are_not_tracked():
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    generated_prefixes = (
        "paper4_submission_source/",
        "paper4_full_derivation/",
        "studies/tau_core_higgs_module_v01/packet_v01_seed/",
    )
    generated_exact = {"arxiv_submission_source.zip"}
    assert not any(path.startswith(generated_prefixes) for path in tracked)
    assert not any(path in generated_exact for path in tracked)
    assert not any(path.startswith("figures/") and path.endswith(".svg") for path in tracked)


def test_wolfram_audit_logs_record_expected_checks():
    overlap = (PACKET / "wolfram_audit_logs/Higgs_Quartic_Overlap_Verification.log").read_text(encoding="utf-8")
    stabilizer = (PACKET / "wolfram_audit_logs/BranchA_Stabilizer_Hypercharge_Audit.log").read_text(encoding="utf-8")
    g2 = (PACKET / "wolfram_audit_logs/G2_Unoriented_Line_Quotient_Audit.log").read_text(encoding="utf-8")
    brst = (PACKET / "wolfram_audit_logs/Projection_BRST_Skeleton.log").read_text(encoding="utf-8")
    ledger = (PACKET / "wolfram_audit_logs/Compact_Gate_Ledger_Audit.log").read_text(encoding="utf-8")
    assert "quarticIntegralCheck = True" in overlap
    assert "I4(3/10) = 0.133756" in overlap
    assert "T_Sigma == -T_Y = True" in stabilizer
    assert "conditional_forcing_route" in stabilizer
    assert "nu_i = 3 |Y_i| / 5 remains theorem-candidate" in stabilizer
    assert "Line projector P_TY equals P_-TY = True" in g2
    assert "Guardrail: proves quotient consequences" in g2
    assert "Q_D h_D = 0 check = True" in brst
    assert "does not prove anomaly freedom" in brst
    assert "compactGateLedgerCount = 21" in ledger
    assert "compactGateLedgerIdsContinuous = True" in ledger
    assert "compactGateLedgerNoSolvedClaim = True" in ledger
    assert "compactGateLedgerAllSubGatesPresent = True" in ledger
    assert "compactGateLedgerAllClaimBoundaryChecks = True" in ledger
    assert "compactGateLedgerNoForbiddenStrongPhrases = True" in ledger
    assert "Top Determinant Finite-Residue Extraction Gate -> True" in ledger
    assert "Numerical Endpoint Matching Protocol -> True" in ledger
    assert "not a completed Higgs proof" in ledger
