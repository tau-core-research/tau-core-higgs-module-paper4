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
        ROOT / "figures/paper4_higgs_zero_mode_profiles.svg",
        ROOT / "figures/paper4_mechanism_flow.svg",
        ROOT / "figures/paper4_quartic_overlap_curve.svg",
        ROOT / "arxiv_submission_source.zip",
        ROOT / "wolfram/Higgs_Quartic_Overlap_Verification.wl",
        ROOT / "wolfram/BranchA_Stabilizer_Hypercharge_Audit.wl",
        ROOT / "wolfram/G2_Unoriented_Line_Quotient_Audit.wl",
        ROOT / "wolfram/Projection_BRST_Skeleton.wl",
        PACKET / "paper4_higgs_overlap_scan_v01.csv",
        PACKET / "paper4_quartic_sensitivity_audit_v01.csv",
        PACKET / "paper4_compact_spectrum_pilot_v01.csv",
        PACKET / "paper4_higgs_module_summary_v01.csv",
        PACKET / "paper4_readiness_table_v01.csv",
        PACKET / "wolfram_audit_logs/Higgs_Quartic_Overlap_Verification.log",
        PACKET / "wolfram_audit_logs/BranchA_Stabilizer_Hypercharge_Audit.log",
        PACKET / "wolfram_audit_logs/G2_Unoriented_Line_Quotient_Audit.log",
        PACKET / "wolfram_audit_logs/Projection_BRST_Skeleton.log",
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
    assert "not a proof of any parent projection theory" in tex
    assert "not an empirical claim" in tex
    assert "not solve the Higgs hierarchy problem" in tex
    assert "not yet prove radiative stability or solve the hierarchy problem" in tex
    assert "Wolfram Language Audit" in tex
    assert "They verify the normalization source of $3/5$" in tex
    assert "they do not derive the localization postulate" in tex
    assert "concrete conditional mathematical result" in tex
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
    assert "A_{\\tau,i}(x)=\\kappa_\\tau^2 Y_i \\tanh x" in tex
    assert "\\nu_i=\\left|\\kappa_\\tau^2 Y_i\\right|=\\kappa_\\tau^2|Y_i|" in tex
    assert "\\operatorname{span}(T_Y)=\\operatorname{span}(-T_Y)" in tex
    assert "\\Sigma\\sim-\\Sigma" in tex
    assert "G2\\_Unoriented\\_Line\\_Quotient\\_Audit.wl" in tex
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
    assert "C19 Zeta-Determinant Cancellation Target" in tex
    assert "C20 Residue Classification Gate" in tex
    assert "H_-&=Q_\\nu^\\dagger Q_\\nu" in tex
    assert "{\\cal K}_\\tau=S^1/\\mathbb Z_2" in tex
    assert "Compact Tau Geometry As The Next Central Gate" in tex
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
    assert "y_t^{\\rm parent}" in tex
    assert "\\epsilon_\\tau$ Origin Gate" in tex
    assert "C0 Finite-Fiber Wall Spectral Cell" in tex
    assert "Adversarial Audit Of C0" in tex
    assert "C1 And C2 Forcing Routes" in tex
    assert "C3 Explicit Geometry Candidate" in tex
    assert "Adversarial Audit Of C3" in tex
    assert "C4 Penalty Interpretation" in tex
    assert "C5 Local Spectrum Gate" in tex
    assert "C6 Compact Spectrum Boundary-Value Problem" in tex
    assert "C7 Competitor Audit" in tex
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
    assert "C19 Zeta-Determinant Cancellation Target" in tex
    assert "C20 Residue Classification Gate" in tex
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


def test_arxiv_zip_is_source_only():
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip") as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "references.bib" in names
    assert "main.pdf" not in names
    assert "figures/paper4_higgs_zero_mode_profiles.pdf" in names
    assert "figures/paper4_mechanism_flow.pdf" in names
    assert "figures/paper4_quartic_overlap_curve.pdf" in names
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
    assert "quarticIntegralCheck = True" in overlap
    assert "I4(3/10) = 0.133756" in overlap
    assert "T_Sigma == -T_Y = True" in stabilizer
    assert "nu_i = 3 |Y_i| / 5 remains a theorem-candidate" in stabilizer
    assert "Line projector P_TY equals P_-TY = True" in g2
    assert "Guardrail: proves quotient consequences" in g2
    assert "Q_D h_D = 0 check = True" in brst
    assert "does not prove anomaly freedom" in brst
