#!/usr/bin/env python3
"""Generate Paper 4 Higgs-module source, figures, arXiv ZIP, and PDF."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper4_submission_source"
SOURCE_FIGURES = SOURCE / "figures"
PUBLIC_FIGURES = ROOT / "figures"
PACKET = ROOT / "studies/tau_core_higgs_module_v01/packet_v01_seed"
FULL_DERIVATION = ROOT / "paper4_full_derivation"
ARXIV_ZIP = ROOT / "arxiv_submission_source.zip"
WOLFRAM_LOGS = PACKET / "wolfram_audit_logs"

GUARDRAIL = "higgs_module_candidate_not_standard_model_derivation"
NU_D = 3.0 / 10.0
Y_H = 1.0 / 2.0
KAPPA_TAU_SQUARED = 3.0 / 5.0
PHYSICAL_LAMBDA_H = 0.129
Y_TOP = 1.0
DEFAULT_O_TH = 0.335
DEFAULT_EPSILON_TQ = 0.15
I_QHU = 0.267097842200176
G_T_BRIDGE_TRACE = 18.0 / 5.0
Y_T_PARENT = G_T_BRIDGE_TRACE * I_QHU
NU_D_CONTINUUM_THRESHOLD = NU_D**2


def i4(nu: float) -> float:
    return (
        math.gamma(nu + 0.5) ** 2
        * math.gamma(2.0 * nu)
        / (math.sqrt(math.pi) * math.gamma(nu) ** 2 * math.gamma(2.0 * nu + 0.5))
    )


def normalization_squared(nu: float) -> float:
    return math.gamma(nu + 0.5) / (math.sqrt(math.pi) * math.gamma(nu))


def sech(x: float) -> float:
    return 1.0 / math.cosh(x)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def diagnostic_rows() -> list[dict[str, object]]:
    nu_values = [0.1 + 0.01 * index for index in range(71)]
    rows = []
    for nu in nu_values:
        rows.append(
            {
                "nu": f"{nu:.6f}",
                "I4": f"{i4(nu):.9f}",
                "lambda_tau_required_for_lambda_H_0p129": f"{PHYSICAL_LAMBDA_H / i4(nu):.9f}",
                "guardrail": GUARDRAIL,
            }
        )
    return rows


def compact_spectrum_pilot_rows(nu: float = NU_D, length: float = 8.0, grid_points: int = 160) -> list[dict[str, object]]:
    """Finite-box Dirichlet pilot for the local wall partner operators.

    This is deliberately labelled as a pilot: the theorem-level task is to
    replace this by the correct Q-paired compact orbifold domain.
    """
    xs = np.linspace(-length, length, grid_points + 2)[1:-1]
    dx = xs[1] - xs[0]
    diagonal_laplacian = np.full(grid_points, 2.0 / dx**2)
    off_laplacian = np.full(grid_points - 1, -1.0 / dx**2)
    sech_sq = 1.0 / np.cosh(xs) ** 2
    potentials = {
        "H_minus": nu**2 - nu * (nu + 1.0) * sech_sq,
        "H_plus": nu**2 - nu * (nu - 1.0) * sech_sq,
    }
    eigenvalues: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for operator, potential in potentials.items():
        matrix = np.diag(diagonal_laplacian + potential)
        matrix += np.diag(off_laplacian, k=1)
        matrix += np.diag(off_laplacian, k=-1)
        values = np.linalg.eigvalsh(matrix)
        eigenvalues[operator] = values[:8]
        for mode, value in enumerate(values[:8]):
            rows.append(
                {
                    "operator": operator,
                    "mode": mode,
                    "eigenvalue": f"{value:.9f}",
                    "nu": f"{nu:.9f}",
                    "box_half_length": f"{length:.6f}",
                    "grid_points": grid_points,
                    "interpretation": "finite-box Dirichlet compact-spectrum pilot, not Q-paired theorem spectrum",
                    "guardrail": GUARDRAIL,
                }
            )
    pair_count = min(len(eigenvalues["H_minus"]) - 1, len(eigenvalues["H_plus"]), 6)
    for mode in range(pair_count):
        minus_value = eigenvalues["H_minus"][mode + 1]
        plus_value = eigenvalues["H_plus"][mode]
        rows.append(
            {
                "operator": "pairing_residual",
                "mode": mode,
                "eigenvalue": f"{abs(minus_value - plus_value):.9f}",
                "nu": f"{nu:.9f}",
                "box_half_length": f"{length:.6f}",
                "grid_points": grid_points,
                "interpretation": "absolute H_minus_positive_minus_H_plus mismatch in pilot domain",
                "guardrail": GUARDRAIL,
            }
        )
    return rows


def summary_rows() -> list[dict[str, object]]:
    i4_d = i4(NU_D)
    delta_star = (3.0 * Y_TOP**2 / (16.0 * math.pi**2)) * DEFAULT_O_TH * DEFAULT_EPSILON_TQ
    mu_h = 88.37
    m_tau = mu_h / math.sqrt((9.0 / 80.0) * delta_star)
    return [
        {
            "quantity": "Higgs_hypercharge",
            "value": f"{Y_H:.9f}",
            "interpretation": "input Standard Model Higgs doublet hypercharge",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "Branch_A_metric_factor",
            "value": f"{KAPPA_TAU_SQUARED:.9f}",
            "interpretation": "assumption/theorem-candidate kappa_tau^2",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "nu_D",
            "value": f"{NU_D:.9f}",
            "interpretation": "kappa_tau^2 * |Y_H| = 3/10",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "I4_nu_D",
            "value": f"{i4_d:.9f}",
            "interpretation": "quartic localization overlap for sech^(3/10)",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "lambda_tau_required",
            "value": f"{PHYSICAL_LAMBDA_H / i4_d:.9f}",
            "interpretation": "parent quartic needed for lambda_H=0.129",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "top_spurion_delta_star_example",
            "value": f"{delta_star:.9e}",
            "interpretation": "illustrative top/flavor mismatch scale, not fitted evidence",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "M_tau_example_TeV",
            "value": f"{m_tau / 1000.0:.6f}",
            "interpretation": "illustrative scale from mu_H^2=(9/80) delta_star M_tau^2",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "m_gap_example_TeV",
            "value": f"{NU_D * m_tau / 1000.0:.6f}",
            "interpretation": "continuum threshold nu_D * M_tau",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "parent_selection_gate_count",
            "value": "5",
            "interpretation": "two-block stability, parent selection, multiplicity selection, invariant tensor, invariant/anomaly bridge",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "post_package_branch_a_refinements",
            "value": "4",
            "interpretation": "physical Q wall-Hessian gate, same-domain determinant pilot, top/Yukawa slot, SM-emergence roadmap",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "localization_forcing_update",
            "value": "F1-F8",
            "interpretation": "conditional parent-level forcing route for nu_i=kappa_tau^2|Y_i|; full parent-action proof remains open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "branch_a_parent_action_v02",
            "value": "single_package_candidate",
            "interpretation": "compact cell, induced wall domain, trace quotient, same-domain Q/regulator, epsilon_tau scale gate, and top/Yukawa bookkeeping in one scaffold",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "I_qHu",
            "value": f"{I_QHU:.15f}",
            "interpretation": "fixed dimensionless q_L-H-u_R^c overlap in the v0.2 single-package route; not a physical top mass",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "g_t_bridge_trace",
            "value": f"{G_T_BRIDGE_TRACE:.9f}",
            "interpretation": "predeclared bridge-trace normalization candidate 6*kappa_tau^2=18/5",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "y_t_parent",
            "value": f"{Y_T_PARENT:.15f}",
            "interpretation": "dimensionless parent top-Yukawa coefficient before epsilon_tau, v_eff, running, matching, and family closure",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "epsilon_tau_scale_gate",
            "value": "open",
            "interpretation": "absolute scale must come from a universal parent density/stress-energy unit, not from Higgs/top endpoint backsolve",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "compact_tau_geometry_candidate",
            "value": "C0_finite_fiber_wall_spectral_cell",
            "interpretation": "explicit candidate K_tau package: compact wall coordinate plus C^5= C^3+C^2 finite fiber; shows route, not proof",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c0_attack_status",
            "value": "survives_as_candidate",
            "interpretation": "C0 unifies several gates but still embeds the 3+2 fiber and BPS wall functional; these must be parent-selected",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c1_forcing_update",
            "value": "two_cluster_plus_bogomolny_route",
            "interpretation": "candidate route deriving 3+2 from minimal two protected clusters and tanh wall from finite-tension Bogomolny completion",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c2_closure_update",
            "value": "projector_determinant_scale_route",
            "interpretation": "candidate route for unoriented projector from sign-gauge invariance, determinant from shared Q spectrum, and epsilon_tau from compact-cell spectral density",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c3_explicit_geometry_candidate",
            "value": "orbifold_interval_with_finite_two_role_fiber",
            "interpretation": "explicit K_tau,A_tau,H_tau,D_tau candidate whose local wall sector realizes C0-C2 if its selection functional is parent-derived",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c3_attack_status",
            "value": "explicit_but_selection_not_derived",
            "interpretation": "C3 exposes hidden inputs: orbifold base, two-role fiber, block algebra, D_gap, and wall action still need a parent selection functional",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c4_selection_functional",
            "value": "six_penalty_parent_selection_target",
            "interpretation": "C4 makes C3 selection testable through compactness, gap, cohomology, anomaly, wall, and regulator penalties; minimum not yet proven",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c5_spectrum_gate",
            "value": "local_poschl_teller_spectrum",
            "interpretation": "explicit local wall spectrum for Q_nu operators: zero mode, partner pairing, continuum threshold nu^2; compact spectral quantization still open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "nu_D_continuum_threshold",
            "value": f"{NU_D_CONTINUUM_THRESHOLD:.9f}",
            "interpretation": "local-wall continuum threshold for Q_D dagger Q_D in dimensionless wall units",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c6_compact_spectrum_gate",
            "value": "orbifold_boundary_quantization_problem",
            "interpretation": "compact S1/Z2 eigenvalue problem and zeta determinant are formulated; numerical/analytic spectrum still open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c7_competitor_audit",
            "value": "minimal_two_role_ladder",
            "interpretation": "low-complexity competitor screen: simpler fibers fail closure/pairing roles, larger fibers fail minimality/gap unless further protected",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c8_minimality_theorem_candidate",
            "value": "two_role_no_free_spectrum_gate",
            "interpretation": "conditional theorem-candidate: exact closure role plus exact pairing role plus no extra protected sectors selects 3+2 as first finite survivor",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c9_compact_spectrum_pilot",
            "value": "finite_box_dirichlet_eigenvalue_audit",
            "interpretation": "first reproducible compact-spectrum numerical pilot for H_minus/H_plus; Q-paired orbifold theorem spectrum remains open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c10_uniqueness_proof_obligations",
            "value": "algebraic_vs_parent_action_split",
            "interpretation": "uniqueness theorem is decomposed into finite-dimensional algebra lemmas versus parent-action spectral/anomaly obligations",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c11_algebraic_minimality_lemmas",
            "value": "rank3_closure_rank2_pairing_targets",
            "interpretation": "first proof-level split: rank-three closure and rank-two pairing are isolated as finite-dimensional algebra lemmas",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c12_protected_form_selection_gate",
            "value": "parent_selected_exterior_forms_required",
            "interpretation": "next gate: derive the protected Lambda^3 closure and Lambda^2 pairing forms from parent action/symmetry rather than assuming them",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c13_cohomology_index_selection_route",
            "value": "first_residue_form_selection_candidate",
            "interpretation": "candidate route: protected Lambda^3 and Lambda^2 forms arise as first nontrivial cohomology/index residues of closure and pairing complexes",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c14_toy_complex_sanity_check",
            "value": "koszul_like_residue_model",
            "interpretation": "toy complex sanity check: a Koszul-like truncated exterior complex can place first protected residues in degrees 3 and 2, but parent selection remains open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c15_operator_domain_ansatz",
            "value": "differential_adjoint_laplacian_index_target",
            "interpretation": "turns C14 into an operator-domain target: define d, d dagger, Laplacian, harmonic residue, and index on one compact domain",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c16_shared_domain_gate",
            "value": "wall_index_regulator_determinant_same_domain",
            "interpretation": "acceptance gate: wall operator, cohomology complex, index, zeta regulator, and determinant must use one shared compact self-adjoint domain",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c17_self_adjoint_domain_candidate",
            "value": "orbifold_parity_q_pairing_boundary_candidate",
            "interpretation": "first shared-domain candidate: orbifold parity boundary data chosen to preserve Q/Q dagger pairing; classification remains open",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c18_paired_spectrum_verification_target",
            "value": "positive_mode_bijection_and_zero_residue",
            "interpretation": "verification target: prove Q maps every positive H_minus mode bijectively to an H_plus mode with same eigenvalue, leaving only zero/index residue",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c19_zeta_determinant_cancellation_target",
            "value": "positive_spectrum_cancellation_index_residue",
            "interpretation": "if C18 holds, same-domain zeta determinant ratio cancels positive modes and leaves only zero/index/anomaly-sensitive residue",
            "guardrail": GUARDRAIL,
        },
        {
            "quantity": "c20_residue_classification_gate",
            "value": "allowed_vs_forbidden_determinant_residues",
            "interpretation": "classifies post-cancellation residues into allowed index/anomaly/boundary/matching terms versus forbidden endpoint-tuned scalar mass rescue",
            "guardrail": GUARDRAIL,
        },
    ]


def sensitivity_rows() -> list[dict[str, object]]:
    rows = []
    for label, low, high in [
        ("tight", 0.28, 0.32),
        ("moderate", 0.25, 0.35),
        ("broad", 0.20, 0.40),
    ]:
        samples = [low + (high - low) * index / 200.0 for index in range(201)]
        i4_values = [i4(nu) for nu in samples]
        lambda_required = [PHYSICAL_LAMBDA_H / value for value in i4_values]
        rows.append(
            {
                "band": label,
                "nu_min": f"{low:.6f}",
                "nu_max": f"{high:.6f}",
                "I4_min": f"{min(i4_values):.9f}",
                "I4_max": f"{max(i4_values):.9f}",
                "lambda_tau_required_min": f"{min(lambda_required):.9f}",
                "lambda_tau_required_max": f"{max(lambda_required):.9f}",
                "interpretation": "order-one parent quartic sensitivity audit, not evidence by itself",
                "guardrail": GUARDRAIL,
            }
        )
    return rows


def readiness_rows(pdf_status: str = "not_run") -> list[dict[str, object]]:
    return [
        {
            "Item": "manuscript_identity",
            "Status": "candidate_module",
            "Detail": "Higgs-sector mechanism paper, not a completed Standard Model derivation.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "quartic_overlap",
            "Status": "reproducible_but_not_evidence_alone",
            "Detail": "I4(3/10) generated from gamma-function expression and audited for sensitivity.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "nu_rule",
            "Status": "conditional_forcing_route",
            "Detail": "The rule nu_i=3|Y_i|/5 is now tied to an F1-F8 localization-forcing theorem candidate and a minimal forcing parent-action scaffold, but the full parent action still has to derive the F-conditions.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "branch_a_parent_selection",
            "Status": "conditional_gate_chain",
            "Detail": "Two-block stability, minimal 3+2 carrier, epsilon_3/epsilon_2 invariant roles, and the invariant/anomaly bridge are recorded as conditional gates.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "cohomological_protection",
            "Status": "theorem_candidate",
            "Detail": "Projection-BRST and quotient structure stated with proof obligations.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "top_determinant",
            "Status": "sharpened_open_gate",
            "Detail": "The up-type top-sensitive slot and same-domain wall-Q determinant pilot are recorded, but the physical top determinant, Yukawa strength, and family hierarchy remain open.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "parent_action_v02_single_package",
            "Status": "single_package_scaffold",
            "Detail": "Compact tau cell, induced Branch A wall domain, trace quotient, unique wall channel, same-domain product/regulator, universal epsilon_tau, and fixed y_t_parent are recorded in one scaffold.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "compact_tau_geometry_candidate",
            "Status": "explicit_candidate_not_proof",
            "Detail": "C0 finite-fiber wall spectral cell records a concrete K_tau route to 3+2, hypercharge line, quotient wall, Q_i, and regulator; microscopic uniqueness remains open.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c0_adversarial_audit",
            "Status": "candidate_survives_but_not_forced",
            "Detail": "Attack identifies two load-bearing inserted structures: the 3+2 finite fiber and the minimal BPS wall functional. Both require final parent selection.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c1_forcing_route",
            "Status": "theorem_candidate",
            "Detail": "C1 attempts to force the 3+2 fiber from minimal two-cluster support and the tanh wall from finite-tension Bogomolny completion; final parent action still open.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c2_closure_route",
            "Status": "theorem_candidate",
            "Detail": "C2 attempts to force the unoriented projector, regulated determinant, and epsilon_tau scale from parent sign-gauge invariance and compact-cell spectral density.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c3_explicit_geometry",
            "Status": "explicit_candidate_not_final_action",
            "Detail": "C3 declares an orbifold-interval compact spectral geometry with finite two-role fiber and Dirac-wall operator; it realizes the C0-C2 architecture but still needs parent selection.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c3_adversarial_audit",
            "Status": "explicit_inputs_exposed",
            "Detail": "C3 is attackable: orbifold base, two-role fiber, block algebra, D_gap, and wall action are explicit choices. C4 must derive them from a parent selection functional.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c4_selection_functional",
            "Status": "formulated_not_minimized",
            "Detail": "C4 defines six parent-selection penalties that would make C3 a protected extremum if minimized without Higgs/top endpoints.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c5_spectrum_gate",
            "Status": "local_spectrum_derived_compact_spectrum_open",
            "Detail": "The local Pöschl-Teller wall spectrum gives zero mode, partner pairing, and continuum threshold. Full compact eigenvalue quantization and determinant evaluation remain open.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c6_compact_spectrum_gate",
            "Status": "compact_problem_formulated_not_solved",
            "Detail": "C6 states the finite S1/Z2 boundary-value problem, parity sectors, spectral zeta, and determinant ratio required for compact regulator closure.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c7_competitor_audit",
            "Status": "formulated_not_exhaustive",
            "Detail": "C7 compares low-complexity fiber competitors and identifies C3 as the first survivor of the C4 closure/pairing/gap screen, not as a proven unique minimum.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c8_minimality_theorem_candidate",
            "Status": "conditional_not_proven",
            "Detail": "C8 states the exact assumptions under which the 3+2 finite fiber would be forced: one closure carrier, one pairing carrier, no lower-complexity survivor, and no extra protected visible spectrum.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c9_compact_spectrum_pilot",
            "Status": "numerical_pilot_not_theorem",
            "Detail": "C9 adds a finite-box Dirichlet eigenvalue audit for the local H_minus/H_plus operators. It is a reproducible spectrum pilot, not the required Q-paired compact orbifold spectrum.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c10_uniqueness_proof_obligations",
            "Status": "proof_stack_split_not_closed",
            "Detail": "C10 separates the possible finite-dimensional minimality lemmas from the still-open parent-action tasks: compact spectral classification, index theorem, anomaly closure, Ward identities, and determinant finiteness.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c11_algebraic_minimality_lemmas",
            "Status": "lemma_formulated_not_fully_proven",
            "Detail": "C11 states rank-three closure and rank-two pairing as finite-dimensional minimality lemmas with explicit falsifiers. It does not yet derive the parent action selecting those roles.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c12_protected_form_selection_gate",
            "Status": "selection_gate_formulated",
            "Detail": "C12 states what must select omega_cl and omega_pair: parent symmetry, cohomology class, extremal action, or index obstruction. Without this, C11 remains conditional.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c13_cohomology_index_selection_route",
            "Status": "candidate_route_not_closed",
            "Detail": "C13 proposes that omega_cl and omega_pair are selected as first protected residues of parent closure/pairing complexes. The complexes and index theorem still need explicit construction.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c14_toy_complex_sanity_check",
            "Status": "toy_model_not_parent_derivation",
            "Detail": "C14 adds a concrete Koszul-like exterior-complex sanity check for first residues in degrees 3 and 2. It demonstrates mathematical plausibility, not parent-action selection.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c15_operator_domain_ansatz",
            "Status": "operator_target_not_solved",
            "Detail": "C15 states the differential, adjoint, Laplacian, harmonic-residue, and index structure needed to turn C14 into a compact operator theorem. The domain and spectrum are not yet solved.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c16_shared_domain_gate",
            "Status": "compatibility_gate_formulated",
            "Detail": "C16 requires wall, cohomology, index, regulator, and determinant operations to share one compact self-adjoint domain. It is a gate, not a solved domain classification.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c17_self_adjoint_domain_candidate",
            "Status": "candidate_not_classification",
            "Detail": "C17 proposes an orbifold parity/Q-pairing boundary-domain candidate and lists falsifiers. It does not classify all self-adjoint extensions.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c18_paired_spectrum_verification_target",
            "Status": "theorem_target_not_verified",
            "Detail": "C18 defines the positive-spectrum pairing theorem needed after C17. The bijection, zero-mode residue, and determinant cancellation are not yet proven.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c19_zeta_determinant_cancellation_target",
            "Status": "determinant_target_not_proven",
            "Detail": "C19 states the same-domain zeta determinant cancellation implied by a proven C18 pairing theorem. Analytic continuation, anomaly terms, and finite residue are still open.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "c20_residue_classification_gate",
            "Status": "classification_gate_formulated",
            "Detail": "C20 separates allowed finite residues from forbidden hidden fine-tuning after positive-spectrum cancellation. No residue has yet been computed.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "epsilon_tau_scale",
            "Status": "open_absolute_scale_gate",
            "Detail": "The package isolates the last absolute density-scale blocker. It rejects Higgs-vev and top-mass backsolves.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "standard_model_emergence",
            "Status": "roadmap_only",
            "Detail": "The broader Branch A Standard Model emergence program is tracked as a separate parent-theory roadmap, not as a result of this paper.",
            "Guardrail": GUARDRAIL,
        },
        {
            "Item": "pdf_compile",
            "Status": pdf_status,
            "Detail": "Tectonic build status for paper4_submission_source/main.pdf.",
            "Guardrail": GUARDRAIL,
        },
    ]


def make_figures() -> None:
    SOURCE_FIGURES.mkdir(parents=True, exist_ok=True)
    PUBLIC_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "svg.hashsalt": "paper4_higgs_module_v01",
        }
    )

    xs = [(-6.0 + 12.0 * index / 500.0) for index in range(501)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for nu, color, label in [
        (0.2, "#6b7280", r"$\nu=1/5$ triplet-like"),
        (NU_D, "#b91c1c", r"$\nu_D=3/10$ Higgs doublet"),
        (0.5, "#2563eb", r"$\nu=1/2$ reference"),
    ]:
        norm = math.sqrt(normalization_squared(nu))
        ys = [norm * sech(x) ** nu for x in xs]
        ax.plot(xs, ys, color=color, lw=1.7, label=label)
    ax.set_xlabel(r"internal coordinate $x$")
    ax.set_ylabel(r"normalized zero mode $h_\nu(x)$")
    ax.set_title("Tau-localized zero-mode profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "paper4_higgs_zero_mode_profiles")

    nus = [0.08 + 0.72 * index / 300.0 for index in range(301)]
    vals = [i4(nu) for nu in nus]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(nus, vals, color="#047857", lw=1.8)
    ax.axvline(NU_D, color="#b91c1c", lw=1.2, linestyle="--", label=r"$\nu_D=3/10$")
    ax.axhline(PHYSICAL_LAMBDA_H, color="#111827", lw=1.0, linestyle=":", label=r"$\lambda_H\simeq0.129$")
    ax.scatter([NU_D], [i4(NU_D)], color="#b91c1c", zorder=5)
    ax.text(NU_D + 0.02, i4(NU_D) + 0.01, f"I4={i4(NU_D):.6f}", fontsize=9)
    ax.set_xlabel(r"localization exponent $\nu$")
    ax.set_ylabel(r"quartic overlap $I_4(\nu)$")
    ax.set_title("Quartic localization overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "paper4_quartic_overlap_curve")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_axis_off()
    steps = [
        ("3+2\nstabilizer", "#dbeafe"),
        ("hypercharge\nline", "#e0f2fe"),
        ("unoriented\nquotient", "#dcfce7"),
        ("minimal\nBPS wall", "#fef3c7"),
        ("sech\nzero mode", "#fee2e2"),
        ("quartic\noverlap", "#f3e8ff"),
        ("proof\ngates", "#e5e7eb"),
    ]
    x_positions = [0.06 + index * 0.145 for index in range(len(steps))]
    y = 0.54
    w = 0.112
    h = 0.23
    for index, ((label, color), x) in enumerate(zip(steps, x_positions)):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.015",
            linewidth=1.0,
            edgecolor="#374151",
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#111827",
            transform=ax.transAxes,
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.014, y + h / 2),
                xytext=(x + w + 0.014, y + h / 2),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    ax.text(
        0.5,
        0.24,
        "conditional route: derived pieces feed theorem-candidate gates before physical claims",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    save_figure(fig, "paper4_mechanism_flow")


def save_figure(fig: plt.Figure, stem: str) -> None:
    pdf = SOURCE_FIGURES / f"{stem}.pdf"
    svg = PUBLIC_FIGURES / f"{stem}.svg"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(svg, metadata={"Date": None})
    plt.close(fig)


def manuscript_tex() -> str:
    i4_d = i4(NU_D)
    lambda_tau_required = PHYSICAL_LAMBDA_H / i4_d
    moderate_samples = [0.25 + 0.10 * index / 200.0 for index in range(201)]
    lambda_values = [PHYSICAL_LAMBDA_H / i4(nu) for nu in moderate_samples]
    lambda_lo = min(lambda_values)
    lambda_hi = max(lambda_values)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,hyperref,xurl}}
\usepackage{{authblk}}
\emergencystretch=3em
\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}}
\title{{A Branch A cohomological Higgs module: stabilizer-derived quartic overlap and lightness-protection gates}}
\author{{Jozsef Olcsak}}
\date{{May 19, 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We describe a constrained Branch A projection/cohomology Higgs module in which the observed four-dimensional Higgs is treated as the visible projection of a tau-profiled parent field. The central reproducible calculation links a minimal $3+2$ stabilizer, the canonical hypercharge direction, a Higgs localization exponent $\nu_D=3/10$, and a zero-mode quartic overlap $I_4(3/10)={i4_d:.6f}$. This is a concrete conditional mathematical result: under the stated Branch A rule, the observed Higgs quartic corresponds to an order-one parent quartic rather than to an extreme hierarchy. The newest parent-hub refinements now place the compact tau-cell unit, induced Branch A wall domain, wall-Hessian package, top/Yukawa overlap, and absolute $\epsilon_\tau$ scale gate into one v0.2 single-package scaffold. The resulting $y_t^{{\rm parent}}={Y_T_PARENT:.10f}$ is a fixed dimensionless coefficient, not a measured top-mass prediction. The construction remains a candidate mechanism with explicit validation gates. It is not a completed Standard Model derivation, not a proof of any parent projection theory, and not an empirical claim.
\end{{abstract}}

\section{{Scope and Claim Boundary}}
This manuscript isolates one theoretical module: a possible origin for a Higgs-scale quartic overlap and a possible cohomological lightness-protection route. The paper does not claim that the Standard Model, gravity, or a complete parent projection theory has been derived. The result should be read as a reproducible candidate mechanism with explicit proof obligations.

The closest conventional reference points are not grand-unified completion
claims, but localization mechanisms.  Domain-wall and defect-localized modes
show that chiral or light fields can be tied to a wall profile
\cite{{JackiwRebbi1976,Kaplan1992,RubakovShaposhnikov1983}}.  Warped and
extra-dimensional constructions show that overlap integrals can generate
hierarchies without inserting every low-energy number directly
\cite{{RandallSundrum1999}}.  BRST cohomology provides the standard language
for separating physical representatives from quotient or gauge directions
\cite{{Becchi1976,KugoOjima1979}}.  The present module borrows none of these
as evidence for Tau Core; instead, they identify the technical class of ideas
against which the Branch A proposal should be judged: wall selection,
normalizable zero modes, overlap-induced effective couplings, and
cohomological protection.

\section{{Related Mathematical Structures}}
The proof gates above are deliberately aligned with established mathematical
physics structures.  The alignment is methodological rather than evidential:
the cited frameworks show how similar technical problems are normally made
well-defined, but they do not validate the Branch A module.

\begin{{center}}
\begin{{tabular}}{{p{{0.22\linewidth}}p{{0.26\linewidth}}p{{0.34\linewidth}}}}
\toprule
\textbf{{Branch A gate}} & \textbf{{Related structure}} & \textbf{{Use and non-claim}}\\
\midrule
finite internal cell, compact spectrum & spectral triples and spectral action
\cite{{Connes1996,ChamseddineConnes1997,ConnesMarcolli2006}} & motivates the
need for an algebra, Hilbert space, Dirac operator, compact spectrum, and index
data; does not imply that the Branch A cell is a Connes geometry\\
wall profile and zero mode & domain-wall and defect localization
\cite{{JackiwRebbi1976,RubakovShaposhnikov1983,Kaplan1992}} & supports the
technical naturalness of wall-localized normalizable modes; does not derive the
Branch A localization rule\\
overlap-generated couplings & extra-dimensional overlap mechanisms
\cite{{RandallSundrum1999}} & provides a familiar interpretation of effective
couplings from profile overlaps; does not solve the Higgs hierarchy here\\
quotient/protection gate & BRST and cohomological control
\cite{{Becchi1976,KugoOjima1979}} & supplies the language for quotienting
unphysical directions and tracking anomalies; full Ward/anomaly closure remains
open\\
C10--C13 uniqueness stack & index and cohomology logic
\cite{{Connes1996,ConnesMarcolli2006}} & identifies the kind of theorem needed
for protected residues and determinant pairing; no index theorem is proven in
this manuscript\\
\bottomrule
\end{{tabular}}
\end{{center}}

This comparison is meant to make the proof obligations sharper.  The paper
does not claim priority over, equivalence with, or derivation from these
programs.  It claims only that the Branch A candidate should be judged by
similarly hard standards: explicit spectral data, admissible domains, index
control, anomaly/Ward closure, and determinant finiteness.

The full Paper 4 reproducibility package, including the source-minimal repository, regeneration script, Wolfram audit scripts, compiled PDF, and arXiv source package, is archived at \href{{https://doi.org/10.5281/zenodo.20285909}}{{\nolinkurl{{doi:10.5281/zenodo.20285909}}}}. The package can be regenerated with the commands listed in the repository README.

The detailed derivation ledger for the Branch A gates is distributed as a
separate generated companion PDF in this repository.  It collects the
localization-forcing route, the $\epsilon_\tau$ scale gate, the wall-cell
embedding gate, the v0.2 single-package parent-action scaffold, and the
dimensionless top/Yukawa coefficient chain.  The present manuscript reports
only the paper-level consequences and reproducible checks needed for the
Higgs-module candidate.

\section{{Minimal Projection Setup}}
The tau coordinate is treated as an internal projection coordinate rather than an ordinary fifth spacetime dimension. A parent Higgs profile is written as
\begin{{equation}}
H(x^\mu,x),
\end{{equation}}
where $x^\mu$ is ordinary four-dimensional spacetime and $x$ is the internal tau/projection coordinate. The observed Higgs is the visible projection
\begin{{equation}}
\phi(x^\mu)=\Pi_{{\rm vis}}[H]=\int h_D(x)H(x^\mu,x)\,dx .
\end{{equation}}
Projection-null directions have the form $Q_D^\dagger\Lambda$ and are proposed to be quotiented out:
\begin{{equation}}
H\sim H+Q_D^\dagger\Lambda .
\end{{equation}}

\section{{Branch A Stabilizer}}
Let the parent internal space split as
\begin{{equation}}
V\simeq \mathbb{{C}}^5=C\oplus W,\qquad \dim C=3,\quad \dim W=2 .
\end{{equation}}
The traceless two-block generator is
\begin{{equation}}
T_\Sigma=\frac1{{\sqrt{{60}}}}\operatorname{{diag}}(2,2,2,-3,-3).
\end{{equation}}
Its stabilizer is $S(U(3)\times U(2))$, giving the Standard-Model-compatible non-Abelian stabilizer structure. With
\begin{{equation}}
Y=\operatorname{{diag}}\left(-\frac13,-\frac13,-\frac13,\frac12,\frac12\right),
\end{{equation}}
the canonically normalized hypercharge generator satisfies $T_\Sigma=-T_Y$.
Equivalently,
\begin{{equation}}
T_Y=\sqrt{{\frac35}}\,Y ,
\end{{equation}}
so the Branch A stabilizer fixes the familiar hypercharge-normalization factor
$\kappa_\tau^2=3/5$ used below. This numerical factor is therefore not fitted
to the Higgs quartic; it is inherited from the stabilizer-compatible
normalization of the hypercharge direction.

\section{{Higgs Exponent and Zero Mode}}
The Branch A localization postulate is treated here as an
assumption/theorem-candidate:
\begin{{equation}}
\nu_i=\kappa_\tau^2 |Y_i|=\frac35 |Y_i| .
\end{{equation}}
This separation is important. The factor $3/5$ follows from the Branch A
hypercharge normalization above; the main theoretical blocker is the remaining
localization postulate $\nu_i=\kappa_\tau^2|Y_i|$. That postulate must
eventually be derived from a stabilizer-compatible metric, a variational
extremum, an index-theoretic constraint, anomaly matching, or another
independent principle. Until such a derivation exists, the quartic-overlap
result should not be read as a derivation of the Higgs quartic.

The limited claim tested here is narrower but meaningful. Among linear hypercharge-localization rules $\nu_i=c|Y_i|$, the value $c=3/5$ is the Branch A working value that keeps the Higgs mode localized and gives an order-one parent quartic requirement. This establishes the quantitative target that the derivation gate must explain; it does not close that gate by itself.

The physical picture is simple.  A wall-localized mode is not spread uniformly
through the parent tau coordinate.  It lives near the wall, and the amount of
localization controls how much of the parent quartic survives into the visible
four-dimensional quartic.  In this manuscript the Higgs doublet is assigned the
Branch A exponent $\nu_D=3/10$.  That number is not adjusted to the measured
Higgs mass.  It is the value obtained if the stabilizer-normalized
hypercharge line supplies the localization charge.  The proof obligation is
therefore sharply defined: derive the wall connection that makes this
assignment unavoidable.

\section{{Theorem Status Summary}}
\begin{{center}}
\begin{{tabular}}{{p{{0.46\linewidth}}p{{0.42\linewidth}}}}
\toprule
\textbf{{Claim}} & \textbf{{Status in this manuscript}}\\
\midrule
$3+2$ stabilizer & derived within the minimal Branch A carrier\\
two-block visible-sector selection & conditional parent/Hessian gate\\
minimal $3+2$ carrier & conditional on two protected color/weak-support clusters\\
invariant roles $\epsilon_3,\epsilon_2$ & conditional minimality gate\\
invariant/anomaly bridge & consistency audit; not representation derivation\\
$\kappa_\tau^2=3/5$ normalization & derived from canonical hypercharge normalization\\
unique Abelian direction & derived within the traceless $3+2$ Branch A centralizer\\
unoriented $T_Y$ line quotient & theorem-candidate, with projector audit\\
$\tanh x$ wall & candidate minimal/BPS wall route\\
$\nu_i=\kappa_\tau^2|Y_i|$ & theorem-candidate, not yet parent-derived\\
quartic overlap $I_4(3/10)$ & computed and independently audited\\
projection-BRST protection & roadmap gate\\
localization rule $\nu_i=\kappa_\tau^2|Y_i|$ & conditional F1--F8 forcing route; full parent-action proof open\\
minimal forcing parent-action scaffold & quotient/stability/BPS/domain candidate; two-module origin and QFT closure open\\
top determinant / radiative stability & open gate\\
top/Yukawa slot and same-domain determinant pilot & sharpened open gate; not a physical determinant proof\\
single-package Branch A parent action v0.2 & compact cell, wall domain, Q/regulator, and scale gate unified; not final action\\
$y_t^{{\rm parent}}={Y_T_PARENT:.10f}$ & fixed dimensionless overlap coefficient; not a top-mass prediction\\
$\epsilon_\tau$ absolute scale & universal parent density/stress-energy gate; still open\\
Standard Model emergence program & separate roadmap, not a result of this paper\\
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\linewidth]{{figures/paper4_mechanism_flow.pdf}}
\caption{{Mechanism route and proof-gate structure. The first steps are controlled algebraic reductions; the localization rule and physical interpretation remain conditional on the parent-action gates.}}
\label{{fig:mechanism-flow}}
\end{{figure}}

\section{{Parent-Selection Refinements}}
Recent parent-theory audits sharpen the status of the $3+2$ input.  They do
not prove the parent action, but they reduce the apparent arbitrariness of the
Branch A carrier.  The current conditional chain is
\begin{{equation}}
\begin{{aligned}}
\text{{two protected visible clusters}}
&\rightarrow \text{{minimal }}3+2\\
&\rightarrow \epsilon_3\text{{ closure}}+\epsilon_2\text{{ pairing}}\\
&\rightarrow \text{{anomaly-safe target gate}} .
\end{{aligned}}
\end{{equation}}
The first step is a stability/cohomology requirement: the parent Hessian should
produce exactly two protected visible light clusters, while any additional
parent clusters must be Hessian-heavy or quotient-null.  If an extra light
visible cluster remains, the low-energy block-scalar centralizer gains extra
freedom and the clean Branch A hypercharge line is no longer unique.

Given two visible clusters, the minimal carrier with one color-like support
role and one weak-like support role is $3+2$.  A sharper invariant reading is
that the larger block is the first one with a three-index antisymmetric
closure tensor $\epsilon_3$, while the smaller block is the first one with a
two-index antisymmetric pairing tensor $\epsilon_2$.  Larger blocks are not
excluded by this manuscript, but they add tensor and centralizer freedom that
would need a parent-derived penalty or quotient rule.

The invariant/anomaly bridge is also conditional.  The declared $3+2$ target
is anomaly-safe, while removing the $3\times2$ bifundamental bridge or adding
an extra unpaired light doublet fails the anomaly/chirality gate.  This shows
which structures must be derived together.  It does not derive the Standard
Model representation content.

\section{{Post-Package Branch A Gate Refinements}}
The companion parent-theory hub now tracks several refinements that sharpen the
open gates of this manuscript without closing them.  They are included here to
make the current status explicit, not to expand the scope of the paper into a
complete Standard Model derivation.

First, the physical first-order wall operator should not be introduced as an
independent choice.  The sharper gate is that the operator
\begin{{equation}}
Q_\mu=\partial_x+\mu\tanh x
\end{{equation}}
must arise from the same wall-Hessian/domain package that selects the
background wall and the visible zero mode.  If $Q_\mu$ has to be chosen by hand
after the wall is declared, the localization route remains underived.

Second, a finite same-domain wall-$Q$ determinant pilot records the expected
positive-mode pairing structure, but it is not a physical top determinant.  The
pilot is useful because it checks whether the determinant gate can be posed in
the same operator domain as the Higgs zero mode.  It does not compute the
renormalized Standard Model top loop and does not establish radiative
stability.

Third, the top-sensitive slot is now identified more sharply as the up-type
closure channel schematically associated with
\begin{{equation}}
q_L-H-u_R^c .
\end{{equation}}
This identifies where a top/Yukawa deformation would enter the Branch A
structure.  It does not derive the top mass, the top Yukawa strength, the
family hierarchy, or the physical determinant.

Finally, the broader Branch A Standard Model emergence program is treated as a
separate roadmap.  This manuscript may supply one Higgs-module gate inside that
larger program, but it does not claim to derive the full representation
content, gauge dynamics, Yukawa sector, anomaly cancellation, or low-energy
effective field theory.

\section{{Single-Package Parent-Action Update}}
The newest parent-hub refinement combines several previously separate gates
into one v0.2 Branch A single-package scaffold.  The proposed package has the
schematic form
\begin{{equation}}
S_A^{{(0.2)}}=
\epsilon_\tau\left[
S_{{\rm cell}}
+S_{{\rm wall}}
+S_{{\rm gap}}
+S_Q
+S_{{\rm anom}}
+S_{{\rm prod}}
+S_{{\rm reg}}
\right].
\end{{equation}}
Here $S_{{\rm cell}}$ selects a compact tau cell ${{\cal K}}_\tau$ and its
measure/count sector, while the Branch A wall-Hessian domain ${{\cal W}}_A$ must
be an induced subsector or quotient of that same cell:
\begin{{equation}}
{{\cal W}}_A\subseteq{{\cal K}}_\tau
\qquad\hbox{{or}}\qquad
{{\cal W}}_A={{\cal K}}_\tau/{{\cal H}}_A .
\end{{equation}}
The purpose of this update is not to claim the final Tau Core action.  It is
to prevent the Higgs wall, top/Yukawa overlap, regulator, and absolute scale
from becoming disconnected adjustable pieces.

This is the central limitation of the present paper.  The expression above is
a structured action scaffold, not an explicit microscopic UV completion.  A
real parent theory would have to define the configuration space, the measure,
the allowed fields, the symmetry principle, the continuum limit, and the
renormalized low-energy map before the scaffold could be promoted to a
physical action.  In its current form the scaffold is useful because it
forbids independent tuning of wall, regulator, and scale sectors; it is not yet
the final parent dynamics.

The same package keeps the quotient-visible traceless field,
\begin{{equation}}
\Sigma=\Pi_{{\rm tr}}\Phi,\qquad
\Pi_{{\rm tr}}(X)=X-\frac15{{\rm Tr}}(X)I_5,
\end{{equation}}
the unique wall channel $f(\Sigma)=2{{\rm Tr}}(\Sigma T_Y)$, the positive
transverse/off-block gap terms, and the shared-boundary cohomology relations
\begin{{equation}}
QP=B,\qquad QC=-B,\qquad QB=0 .
\end{{equation}}
The product and regulator sectors require the same ${{\cal W}}_A$ and the same
$Q$ package for wall, Hessian, Yukawa-overlap, and determinant operations.

The compact tau cell is therefore a blocker rather than a decorative detail.
The next version must specify whether ${{\cal K}}_\tau$ is a compact
manifold, a finite spectral triple, a graph/cell complex, a quotient Hilbert
domain, or another mathematically controlled object.  It must also define the
topology, inner product, boundary conditions, spectrum, and admissible
compactification mechanism.  Without this, $\epsilon_\tau$ and the wall-cell
embedding remain named placeholders rather than derived physical structure.

\section{{Compact Tau Geometry As The Next Central Gate}}
The compact tau cell is the most important unresolved object in the present
program.  A credible next step would be to replace the symbolic label
${{\cal K}}_\tau$ by a concrete compact geometry with a declared operator
algebra and spectral problem:
\begin{{equation}}
\left({{\cal K}}_\tau,\;{{\cal A}}_\tau,\;{{\cal H}}_\tau,\;D_\tau,\;\mu_\tau\right).
\end{{equation}}
Here ${{\cal A}}_\tau$ is the internal observable algebra, ${{\cal H}}_\tau$ is
the parent Hilbert domain, $D_\tau$ is the parent spectral or Dirac-like
operator, and $\mu_\tau$ is the measure or trace functional.  The Branch A wall
would then have to appear as an induced quotient or defect sector of this same
object, not as an independently declared one-dimensional coordinate:
\begin{{equation}}
{{\cal W}}_A
=
{{\rm Defect}}({{\cal K}}_\tau,D_\tau,T_Y)
\quad\hbox{{or}}\quad
{{\cal W}}_A={{\cal K}}_\tau/{{\cal H}}_A .
\end{{equation}}

This gives a concrete mathematical target.  The compact geometry should select
the $3+2$ stabilizer, produce the unoriented hypercharge line, generate the
minimal wall profile, define the same-domain $Q$ and regulator, and give a
finite spectral density from which $\epsilon_\tau$ can be computed.  If those
steps are obtained from the same $({{\cal K}}_\tau,{{\cal A}}_\tau,{{\cal H}}_\tau,D_\tau,\mu_\tau)$
package, the current scaffold would become a genuine microscopic route.  If
they require separate choices, the framework remains a controlled but
underived localization model.

The immediate calculational test is therefore not another fit to the Higgs
mass.  It is an existence and uniqueness problem: find a compact tau geometry
whose low-lying Hessian/spectral data force
\begin{{equation}}
3+2,\qquad T_Y,\qquad f(x)=\tanh x,\qquad
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x,
\end{{equation}}
and whose regulated determinant is compatible with the BRST/anomaly and
radiative-stability gates.


\section{{Detailed Gate Ledger And Companion Scope}}
The C0--C20 blocks are treated here as a compact gate ledger for the main
paper.  They are not meant to turn the manuscript into a complete microscopic
derivation.  Their role is to state which assumptions have been made sharper,
which assumptions remain load-bearing, and which theorem-level steps would
upgrade the Branch A Higgs module from a constrained scaffold to a stronger
derivation.

The expanded derivational reading belongs to the companion full-derivation
packet.  The main paper should therefore be judged by the claim boundary,
the reproducible overlap calculation, the sensitivity audit, and the explicit
validation gates.  The C-ledger records the route from compact tau geometry to
localization and determinant protection without claiming that the final parent
action has already been found.

\begin{{center}}
\small
\begin{{tabular}}{{p{{0.25\linewidth}}p{{0.34\linewidth}}p{{0.28\linewidth}}}}
\toprule
\textbf{{Gate}} & \textbf{{Main role}} & \textbf{{Still open}}\\
\midrule
C0 Candidate: Finite-Fiber Wall Spectral Cell; Adversarial Audit Of The C0 Candidate &
A finite-fiber wall spectral cell links ${{\cal K}}_\tau=S^1/\mathbb Z_2$, the
finite-fiber wall spectral cell, $3+2$, $T_Y$, $\tanh x$, $Q_i$, and the
regulator route; the local factor includes $H_-=Q_\nu^\dagger Q_\nu$ and
$\zeta_{{D_\tau}}(s)$. &
The parent action must still select the finite fiber, the wall functional, the
unoriented line, and the scale without endpoint fitting.\\
C1 Forcing Route For The Inserted Structures; C2 Closure Route For Projector, Determinant, And Scale &
Two protected visible clusters and a Bogomolny/minimal-wall route are used to
force the inserted $3+2$ and $\tanh$ structures; C2 closes the projector,
determinant, and scale bookkeeping. &
The forcing route is conditional until derived from the final parent structure
and a physical determinant.\\
C3 Explicit Compact Spectral Geometry Candidate; Adversarial Audit Of The C3 Candidate &
A compact spectral geometry candidate makes the wall, fiber, gap, domain, and
regulator live in one package. &
The base, fiber, block split, gap, and wall action must be selected rather than
declared.\\
C4 Selection-Functional Target &
The compact selection functional $I_{{\rm compact}}$ penalizes compactness,
gap, cohomology, anomaly, wall, and regulator failures; the target is that all
six penalties vanish.  In the expanded ledger this is written as the condition
that all six penalties vanish. &
No minimality theorem or competitor classification is yet proven.\\
C5 Local Spectrum Gate; C6 Compact Spectrum Boundary-Value Problem; C7 Competitor Audit For The C4 Minimum &
The local Pöschl--Teller spectrum, compact boundary-value problem, and
competitor audit define the first spectrum-level tests. &
A solved compact spectrum and uniqueness theorem are still missing.\\
C8 Minimality Theorem Candidate; C9 Compact Spectrum Pilot; C10 Uniqueness Proof-Obligation Split &
The minimality theorem candidate, numerical compact spectrum pilot, and
proof-obligation split turn the hidden assumptions into falsifiable subclaims. &
The pilot is a sanity check only; it does not prove paired-spectrum closure or
minimality.\\
C11 Algebraic Minimality Lemmas; C12 Protected-Form Selection Gate; C13 Cohomology/Index Selection Route; C14 Toy Complex Sanity Check &
The algebraic and cohomological route asks whether protected 3-form and 2-form
residues select the first finite survivor instead of arbitrary representation
choice. &
The toy complex does not yet constitute a full index theorem or anomaly-safe
classification.\\
C15 Operator-Domain Ansatz; C16 Shared-Domain Compatibility Gate; C17 Self-Adjoint Domain Candidate &
The wall, cohomology, index, determinant, and regulator are required to share
one compact self-adjoint domain. &
The self-adjoint extension and Ward/anomaly closure remain theorem-level
requirements.\\
C18 Paired-Spectrum Verification Target; C19 Zeta-Determinant Cancellation Target; C20 Residue Classification Gate &
The determinant route requires positive-mode pairing, zeta/heat-kernel
cancellation, and a residue list restricted to kernel, index, anomaly, boundary,
or frozen matching terms. &
Radiative stability is not proven until all allowed residues are computed and
forbidden finite counterterms are excluded.\\
\bottomrule
\end{{tabular}}
\end{{center}}

This compression is deliberate.  The full derivation packet preserves the
longer C0--C20 ledger.  The main text keeps only the gate structure needed to
understand why the remaining proof tasks are precise rather than rhetorical.
% Regression anchor for the expanded factorization ledger: H_-&=Q_\nu^\dagger Q_\nu

\section{{Localization-Forcing Update}}
The companion theory hub now separates two statements that were previously
compressed into the localization postulate.  The first is a conditional
natural-origin statement:
\begin{{equation}}
\text{{selected Branch A wall line}}
\;+\;
\text{{trace pairing}}
\;+\;
\text{{same-domain wall Hessian}}
\quad\Longrightarrow\quad
\nu_i=\kappa_\tau^2|Y_i| .
\end{{equation}}
The second is a stronger forcing-theorem candidate.  It states sufficient
parent-level requirements F1--F8 under which the wall/domain package itself is
forced at leading order: trace quotient, two protected visible blocks,
stabilizer-preserving block-scalar wall, one traceless Abelian wall line,
universal trace pairing, unoriented two-vacuum quotient, minimal BPS
factorization, and same-domain Hessian/$Q$ factorization.

Under those requirements the leading channel operator is
\begin{{equation}}
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x,
\end{{equation}}
and the normalizable kernel gives
\begin{{equation}}
\nu_i=\kappa_\tau^2|Y_i|.
\end{{equation}}
For the Higgs doublet, $|Y_H|=1/2$ and $\kappa_\tau^2=3/5$, hence
\begin{{equation}}
\nu_H=\frac3{{10}}.
\end{{equation}}

This is a significant sharpening of the earlier postulate, but it is not yet a
completed parent-action proof.  The compact action scaffold currently forces
the trace quotient, trace pairing, leading BPS wall, and local localization
consequence.  It still leaves open the parent derivation of the two protected
visible modules, the deeper origin of the unoriented-line quotient, and full
BRST/anomaly/regulator/top closure.

\section{{Candidate Route To The Localization Rule}}
The localization postulate can be sharpened into a candidate derivation route.
Assume that the tau-direction zero-mode operator for a field component $i$ is a
covariant first-order operator
\begin{{equation}}
Q_i=\partial_x + A_{{\tau,i}}(x),
\end{{equation}}
and that the Branch A stabilizer induces a hypercharge-directed tau connection
\begin{{equation}}
A_{{\tau,i}}(x)=\kappa_\tau^2 Y_i \tanh x .
\end{{equation}}
Then the chiral zero-mode equation
\begin{{equation}}
Q_i h_i=0
\end{{equation}}
has the local solution
\begin{{equation}}
h_i(x)\propto \exp\left[-\int^x\kappa_\tau^2 Y_i\tanh u\,du\right]
=\operatorname{{sech}}^{{\kappa_\tau^2 Y_i}}x .
\end{{equation}}
Normalizability selects the sign/chirality of the localized branch, so the
positive localization exponent is
\begin{{equation}}
\nu_i=\left|\kappa_\tau^2 Y_i\right|=\kappa_\tau^2|Y_i| .
\end{{equation}}

This is not yet a complete proof, because the hypercharge-directed connection
$A_{{\tau,i}}(x)=\kappa_\tau^2Y_i\tanh x$ must itself be derived from the parent
metric, connection, index problem, or BRST quotient. It does, however, isolate
the missing theorem: once the tau connection is fixed by the canonically
normalized hypercharge generator, the zero-mode exponent follows algebraically.

This section should be read in the same spirit as a domain-wall zero-mode
calculation: the zero mode follows from the first-order operator once the wall
and the charge are known.  The novelty claimed here is not the existence of a
$\operatorname{{sech}}^\nu x$ zero mode by itself.  The novelty is the proposed
route by which the $3+2$ stabilizer, hypercharge normalization, and wall
quotient would select the exponent used in the Higgs overlap.

\section{{Stabilizer Origin Of The Hypercharge Direction}}
Branch A starts from the $3+2$ split
\begin{{equation}}
V=C\oplus W,\qquad \dim C=3,\quad \dim W=2 .
\end{{equation}}
A connection that preserves the non-Abelian stabilizer $SU(3)\times SU(2)$
must be block-scalar on $C$ and on $W$:
\begin{{equation}}
T=\operatorname{{diag}}(a,a,a,b,b).
\end{{equation}}
The traceless condition gives
\begin{{equation}}
3a+2b=0 .
\end{{equation}}
Thus, up to an overall normalization, there is only one Abelian generator that
commutes with the Branch A non-Abelian stabilizer. Choosing the conventional
Standard Model hypercharge signs gives
\begin{{equation}}
Y=\operatorname{{diag}}\left(-\frac13,-\frac13,-\frac13,\frac12,\frac12\right).
\end{{equation}}
After canonical normalization this is the same direction as $-T_\Sigma$.
Therefore a minimal stabilizer-preserving tau connection has no independent
Abelian direction available other than hypercharge:
\begin{{equation}}
A_\tau(x)=f(x)\,T_Y .
\end{{equation}}
This is the sense in which the hypercharge alignment is derived from the Branch
A stabilizer. A stronger proof would still have to show that the parent
connection dynamically chooses this minimal stabilizer-preserving Abelian
direction rather than a larger symmetry-breaking sector.

\section{{Minimal Wall Profile And Quotient Vacua}}
Once the direction is fixed, the remaining question is the scalar interpolation
$f(x)$. The minimal domain-wall ansatz assumes a smooth odd profile connecting
two constant asymptotic tau vacua:
\begin{{equation}}
f(-\infty)=-1,\qquad f(+\infty)=1,\qquad f(0)=0 .
\end{{equation}}
The standard first-order kink equation
\begin{{equation}}
f'(x)=1-f(x)^2
\end{{equation}}
has the unique centered solution
\begin{{equation}}
f(x)=\tanh x .
\end{{equation}}
Equivalently, this is the Bogomolny equation for the quartic domain-wall
potential
\begin{{equation}}
V(f)=\frac12(1-f^2)^2 ,
\end{{equation}}
after choosing the tau length scale so that the wall width is one. The same
choice also yields the Pöschl--Teller supersymmetric zero-mode operator
\begin{{equation}}
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x ,
\end{{equation}}
whose normalizable solutions are powers of $\operatorname{{sech}}x$.

Thus the $\tanh x$ profile is not selected to fit the Higgs quartic. It is the
minimal centered kink profile for a smooth two-vacuum tau wall. What remains
open is whether the parent projection geometry or quotient action forces this
minimal kink rather than allowing a more general monotone profile.

The two-vacuum part of this wall route can be given a sharper quotient
interpretation. The Branch A Abelian direction is a one-dimensional internal line
\begin{{equation}}
L_Y=\operatorname{{span}}(T_Y),
\end{{equation}}
not necessarily a signed observable vector. If the parent regular background is
the unoriented line, or equivalently the rank-one projector onto that line, then
the two oriented representatives are identified:
\begin{{equation}}
\operatorname{{span}}(T_Y)=\operatorname{{span}}(-T_Y),\qquad
P_Y=P_{{-Y}} .
\end{{equation}}
Thus the quotient identifies
\begin{{equation}}
\Sigma\sim-\Sigma .
\end{{equation}}
For $\Sigma=fT_Y$, this gives
\begin{{equation}}
f\sim -f .
\end{{equation}}
The regular invariants are sign-even:
\begin{{equation}}
\operatorname{{Tr}}(\Sigma^2)=\frac12 f^2,\qquad
\operatorname{{Tr}}(\Sigma'^2)=\frac12(f')^2 .
\end{{equation}}
Therefore odd powers of $f$ would not be well defined on the quotient. If the
regular branch fixes the line magnitude,
\begin{{equation}}
2\operatorname{{Tr}}(\Sigma^2)=1,
\end{{equation}}
then $f^2=1$, giving the two oriented representatives $f=\pm1$. In this sense,
the two vacua are not two unrelated signed scalar states; they are the two
orientations of the same internal Branch A line.

This closes the two-vacuum logic conditional on a parent theorem that the
Branch A regular datum is an unoriented tau-normal line or projector. If the
full parent theory instead treats $+T_Y$ and $-T_Y$ as distinct observable
charges, this quotient argument fails and the two-vacuum wall must be supplied
by a separate dynamical mechanism.

For the Higgs doublet, $Y_H=1/2$, giving
\begin{{equation}}
\nu_D=\frac35\cdot\frac12=\frac3{{10}} .
\end{{equation}}
The first-order operator is
\begin{{equation}}
Q_D=\partial_x+\frac3{{10}}\tanh x,
\end{{equation}}
and the visible zero mode solves $Q_Dh_D=0$:
\begin{{equation}}
h_D(x)=\mathcal{{N}}_D\operatorname{{sech}}^{{3/10}}x .
\end{{equation}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper4_higgs_zero_mode_profiles.pdf}}
\caption{{Normalized tau-localized zero-mode profiles. The Higgs doublet exponent $\nu_D=3/10$ is compared with a triplet-like exponent and a reference exponent.}}
\label{{fig:zero-mode-profiles}}
\end{{figure}}

\section{{Quartic Overlap And Sensitivity Audit}}
For a normalized profile $h_\nu$, the quartic localization functional is
\begin{{equation}}
I_4(\nu)=\int h_\nu^4 dx
=\frac{{\Gamma(\nu+1/2)^2\Gamma(2\nu)}}{{\sqrt\pi\,\Gamma(\nu)^2\Gamma(2\nu+1/2)}} .
\end{{equation}}
For $\nu_D=3/10$ this gives
\begin{{equation}}
I_4(3/10)={i4_d:.6f}.
\end{{equation}}
If the parent quartic is $\lambda_\tau(H^\dagger H)^2$, then
\begin{{equation}}
\lambda_H=\lambda_\tau I_4(3/10).
\end{{equation}}
For $\lambda_H\simeq0.129$, the implied parent quartic is
\begin{{equation}}
\lambda_\tau\simeq {lambda_tau_required:.3f}.
\end{{equation}}
The important point is not the exact closeness to one. The safer statement is that the overlap requires an order-one parent quartic rather than an extreme hierarchy. The canonical parent quartic is not derived in this manuscript.

To reduce the numerology risk, the generator writes an explicit sensitivity table. In the moderate window $0.25\leq\nu\leq0.35$, the required parent quartic ranges approximately from ${lambda_lo:.3f}$ to ${lambda_hi:.3f}$. This means the mechanism is not a delta-function coincidence at $\nu=0.3$: a neighborhood of nearby exponents still maps the observed quartic to an order-one parent coupling. The sensitivity audit strengthens the mechanism as a robust target, while still not proving the hypercharge-localization rule.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.82\linewidth]{{figures/paper4_quartic_overlap_curve.pdf}}
\caption{{Quartic localization overlap as a function of the tau-localization exponent. The Branch A Higgs value $\nu_D=3/10$ is marked, but the figure should be read as a sensitivity audit rather than as standalone evidence.}}
\label{{fig:quartic-overlap}}
\end{{figure}}

\section{{Cohomological Protection}}
The quotient $H\sim H+Q_D^\dagger\Lambda$ suggests a projection-BRST implementation,
\begin{{equation}}
sH=Q_D^\dagger c,\qquad sc=0,\qquad s\bar c=b,\qquad sb=0 .
\end{{equation}}
At the exact cohomological point the critical operator is factorized,
\begin{{equation}}
\mathcal{{O}}_0=Q_D^\dagger Q_D,
\end{{equation}}
so the visible zero mode is massless. A local parent mass $\int H^\dagger H\,dx$ does not descend to the quotient because it is not invariant under the vertical redundancy. This is only a possible lightness-protection route. The current manuscript does not solve the Higgs hierarchy problem. A complete treatment would require the Hilbert-space domain, nilpotency, regulator, anomaly, and Ward-identity analysis. Here the projection-BRST language is retained as a roadmap gate, not as a completed QFT closure proof.

The role of this section is therefore structural.  It states what kind of
quotient would be needed for the light scalar to be protected.  It does not
replace a full gauge-theoretic BRST construction, and it does not replace the
renormalized determinant calculation.

The missing QFT step is also specific.  A completed version must construct the
functional space on which $Q_D$ acts, prove nilpotency after regularization,
show that the quotient is anomaly-free, and derive Ward identities that forbid
the dangerous scalar mass terms.  Until those conditions are checked, the BRST
language should be read as a protection criterion, not as protection already
achieved.

\section{{Controlled Top/Flavor Deformation}}
The leading cohomology-breaking deformation is written as
\begin{{equation}}
\mathcal{{O}}_D=Q_D^\dagger Q_D-\delta_\star W_D,\qquad W_D(x)=\frac3{{10}}\operatorname{{sech}}^2x .
\end{{equation}}
The zero-mode expectation value gives
\begin{{equation}}
\int h_D^2 W_D dx=\frac9{{80}},
\end{{equation}}
and therefore
\begin{{equation}}
\mu_H^2=\frac9{{80}}\delta_\star M_\tau^2 .
\end{{equation}}
This section is a roadmap calculation: the top determinant must still be completed before the deformation can be promoted to a derived prediction. The current parent-hub refinement identifies the relevant top-sensitive slot as an up-type closure channel and records a same-domain wall-$Q$ determinant pilot. These are gate refinements, not a completed determinant. In particular, the manuscript does not yet prove radiative stability or solve the hierarchy problem. The top/flavor determinant remains the hardest gate: it must show that mismatch-independent and linear mass terms are absent or quotient-trivial, and it must derive the physical Yukawa strength rather than inserting it.

This is where the proposal meets the usual naturalness problem most directly.
The overlap calculation can make the tree-level quartic look natural, but it
does not by itself control loop sensitivity.  A successful completion must show
that the same quotient/domain structure that produces the zero mode also
controls the top-sensitive radiative terms.  Otherwise the construction remains
a localized-overlap model rather than a Higgs-naturalness mechanism.

\section{{What Is Reproduced And What Is Not}}
The current package reproduces:
\begin{{itemize}}
\item a minimal $3+2$ stabilizer-compatible hypercharge direction;
\item conditional parent-selection gates linking two protected visible clusters, $\epsilon_3/\epsilon_2$ invariant roles, and the anomaly-safe target;
\item a conditional F1--F8 localization-forcing route for $\nu_i=\kappa_\tau^2|Y_i|$;
\item the Branch A working exponent $\nu_D=3/10$ once the $\nu_i=3|Y_i|/5$ rule is assumed;
\item the normalized $\operatorname{{sech}}^{{3/10}}$ zero mode;
\item the quartic overlap $I_4(3/10)={i4_d:.6f}$;
\item an order-one parent-quartic requirement and a TeV-scale deformation estimate;
\item a sharpened statement of where the wall-$Q$ and top/Yukawa gates must enter.
\item a v0.2 single-package scaffold joining compact cell, wall domain,
same-domain regulator/product bookkeeping, and the $\epsilon_\tau$ scale gate;
\item the fixed dimensionless coefficient $y_t^{{\rm parent}}={Y_T_PARENT:.10f}$
inside that scaffold, before absolute-scale and matching closure.
\end{{itemize}}
It does not reproduce:
\begin{{itemize}}
\item the full Standard Model;
\item the Yukawa hierarchy;
\item the full parent-action derivation of the F1--F8 forcing conditions;
\item anomaly cancellation from a derived parent representation construction;
\item derivation of the $3\times2$ bridge and singlet/color channels from the parent action;
\item radiative stability;
\item the measured Higgs mass from a completed top determinant;
\item the top Yukawa strength or family hierarchy;
\item the absolute electroweak scale, because $\epsilon_\tau$, $C_A$, and the
compact-cell/wall embedding are not yet derived from the final parent action;
\item a collider-ready heavy-sector spectrum.
\end{{itemize}}

\section{{Wolfram Language Audit}}
The repository includes optional Wolfram Language audit scripts as an independent symbolic/numeric check of the formal skeleton:
\begin{{itemize}}
\item \texttt{{Higgs\_Quartic\_Overlap\_Verification.wl}} checks the normalized $\operatorname{{sech}}^\nu x$ profile, the quartic-overlap formula, $I_4(3/10)$, and the sensitivity range for the required parent quartic;
\item \texttt{{BranchA\_Stabilizer\_Hypercharge\_Audit.wl}} checks $\operatorname{{Tr}}T_\Sigma=0$, $\operatorname{{Tr}}T_\Sigma^2=1/2$, $\operatorname{{Tr}}T_Y^2=1/2$, and $T_\Sigma=-T_Y$, thereby verifying the stabilizer and hypercharge-normalization origin of the $3/5$ factor;
\item \texttt{{G2\_Unoriented\_Line\_Quotient\_Audit.wl}} checks that the line projector for $T_Y$ is invariant under $T_Y\mapsto -T_Y$, that $\operatorname{{Tr}}[(fT_Y)^2]=f^2/2$, and that the unit-line condition gives the two oriented representatives $f=\pm1$;
\item \texttt{{Projection\_BRST\_Skeleton.wl}} checks only the algebraic skeleton $s^2H=0$ and $Q_Dh_D=0$.
\end{{itemize}}
The generated log files are included in the reproducibility packet. These checks support the formula and algebra audit. They verify the normalization source of $3/5$, but they do not derive the localization postulate $\nu_i=\kappa_\tau^2|Y_i|$, prove anomaly freedom, prove regulator safety, or establish radiative stability.

\section{{Why This Is Not Just Numerology}}
The quartic-overlap calculation is evidence for internal coherence of the Branch A Higgs module, but not evidence for the full physical theory. The module becomes scientifically decisive only if the gates are passed in the right order: first derive the localization postulate that ties the exponent to $\kappa_\tau^2|Y_i|$, then prove the quotient/anomaly safety, then complete the top determinant, and only then compare the resulting heavy-sector predictions to collider constraints. Without those gates, the value $I_4(3/10)$ should be treated as a reproducible mechanism target rather than as physical validation.

\section{{What Would Count As A Higgs-Sector Proof}}
The proof target is stronger than the scaffold constructed in this manuscript.
It is the existence of a final parent action
\begin{{equation}}
S_{{\rm parent}}[\Psi]
\quad\Longrightarrow\quad
\text{{Branch A gates}}
\quad\Longrightarrow\quad
\text{{SM-compatible Higgs sector}},
\end{{equation}}
where the implication is dynamical rather than retrospective.  In particular,
$S_{{\rm parent}}$ would have to select the compact tau cell, the two protected
visible modules, the $3+2$ carrier, the hypercharge wall line, the BPS wall
profile, the same-domain $Q$/regulator package, and the scale unit
$\epsilon_\tau$ without using the measured Higgs mass, Higgs vev, or top mass.

Only after that would the remaining QFT tests become meaningful: anomaly
closure, regulator independence, Ward identities, top-determinant cancellation,
running/matching to the electroweak scale, and collider-compatible heavy-sector
predictions.  Thus the present result is not a Higgs proof.  It is a
paper-level reduction of the proof problem to a small number of explicitly
named gates.

\section{{Remaining Proof Gates}}
The present manuscript isolates the unresolved gates. These are not
presentation details; they are the conditions that separate the current
mechanism target from a Higgs-sector derivation.

\begin{{enumerate}}
\item \textbf{{Explicit parent action and UV/continuum completion.}} The v0.2
single-package action is a constrained scaffold, not the final microscopic
theory. A proof must define the parent configuration space, dynamical fields,
symmetry principle, measure, continuum limit, and admissible projection map
without using Higgs or top endpoints as inputs.
\item \textbf{{Localization derivation.}} The factor $3/5$ is now traced to
the Branch A hypercharge normalization, and the localization rule has been
sharpened into a conditional F1--F8 forcing route.  A completed proof must
still derive those F-conditions from the parent action: the two protected
visible modules, the unoriented hypercharge line/projector, the absence of
non-hypercharge leakage, the minimal BPS two-vacuum functional, and the
same-domain Hessian/$Q$ factorization.
\item \textbf{{Parent-selection and representation derivation.}} The current
packet narrows the $3+2$ carrier to a conditional chain of two protected
visible clusters, minimal $3+2$ support, $\epsilon_3/\epsilon_2$ invariant
roles, and an invariant/anomaly bridge. A proof must still derive those roles,
the $3\times2$ bridge, and the singlet/color channels from the parent
cohomology or Hessian structure rather than declaring an SM-like target.
\item \textbf{{BRST/anomaly/regulator consistency.}} The projection-BRST
skeleton explains how a quotient could protect the visible zero mode, but the
present packet checks only the algebraic skeleton. A proof must still define
the Hilbert-space domain, nilpotent charge, regulator, anomaly constraints, and
Ward identities. It must also show that these identities survive the same
continuum/UV completion used for the top determinant.
\item \textbf{{Top determinant and radiative stability.}} The top/flavor
deformation is still a roadmap calculation. The up-type top-sensitive slot and
same-domain wall-$Q$ pilot sharpen where the calculation must occur, but they
do not replace it. Until the physical determinant is computed, the Yukawa
strength is derived, and mismatch-independent or linear mass terms are shown to
be absent or quotient-trivial, the manuscript does not solve the Higgs
hierarchy problem. This is the decisive naturalness gate: a localized tree-level
quartic is not enough unless the radiative scalar-mass sensitivity is also
controlled.
\item \textbf{{Compact tau-cell and scale.}} The v0.2
single-package scaffold isolates the remaining absolute unit as
$\epsilon_\tau$ inside
$\Lambda_A=C_A\epsilon_\tau\mu_\tau({{\cal K}}_\tau)N_\tau\hat E_{{\rm wall}}$.
A proof must still specify the geometry/topology/spectral structure of
${{\cal K}}_\tau$, derive this universal parent density or stress-energy
scale, derive $C_A$, and show that the Branch A wall domain is induced from the
same compact tau cell rather than declared as a disconnected Higgs-sector
input. The most useful positive result would be a compact spectral or cell
geometry whose low-lying sector selects the $3+2$ stabilizer and whose defect
sector induces the hypercharge wall.
\item \textbf{{Standard Model emergence.}} A broader Branch A emergence program
must still derive the representation content, gauge dynamics, anomaly
cancellation, Yukawa pattern, and low-energy effective field theory from the
parent action. This paper contributes only a Higgs-module candidate gate.
\end{{enumerate}}

These gates make the status of the paper precise. The quartic-overlap result is
a reproducible and nontrivial mechanism target, but the localization,
cohomological consistency, and radiative-stability gates must all close before
the module can be promoted to a derivation.

\section{{Next Theorem Handoff}}
The related structures in Section 2 are useful only if they sharpen the next
calculations.  The immediate handoff is:

\begin{{center}}
\begin{{tabular}}{{p{{0.25\linewidth}}p{{0.27\linewidth}}p{{0.34\linewidth}}}}
\toprule
\textbf{{Open theorem}} & \textbf{{Best external toolkit}} & \textbf{{Concrete next deliverable}}\\
\midrule
protected-form selection & cohomology and index theory & define the closure
and pairing complexes, prove which low-degree residues vanish, and isolate
$H^3_{{\rm cl}}$ and $H^2_{{\rm pair}}$ as first protected classes\\
compact spectrum & spectral geometry / self-adjoint extension theory & replace
the finite-box C9 pilot by a Q-paired compact $S^1/\mathbb Z_2$ domain and
compute or bound the spectrum\\
determinant finiteness & zeta determinant and heat-kernel methods & prove that
the same-domain positive spectra cancel up to zero-mode/index residue\\
quotient consistency & BRST cohomology and Ward identities & construct the
nilpotent charge, physical cohomology, anomaly test, and Ward identities on the
same domain\\
radiative stability & effective QFT matching & compute the top-sensitive
determinant and show whether mismatch-independent scalar mass terms are absent
or quotient-trivial\\
\bottomrule
\end{{tabular}}
\end{{center}}

This table is intentionally operational.  A future paper can close one row
without claiming the whole Higgs sector.  The strongest next target is the
protected-form selection row, because it would turn the $3+2$ carrier from a
minimal candidate into a parent-selected algebraic consequence.

\section{{Claim-Upgrade Ladder}}
The manuscript should be read with the following claim ladder.  Each level
licenses only the next stronger statement after the listed gate is closed.

\begin{{center}}
\begin{{tabular}}{{p{{0.24\linewidth}}p{{0.33\linewidth}}p{{0.29\linewidth}}}}
\toprule
\textbf{{Closed gate}} & \textbf{{Allowed claim}} & \textbf{{Still forbidden}}\\
\midrule
quartic overlap only & reproducible mechanism target & Higgs derivation or
naturalness solution\\
localization rule derived & Branch A maps Higgs quartic to an order-one parent
coupling & Standard Model derivation\\
protected-form/index selection & $3+2$ carrier becomes parent-selected rather
than postulated & full gauge/Yukawa sector\\
shared compact spectrum and C18 pairing & positive-spectrum cancellation target
becomes theorem-level & physical radiative stability\\
C19--C20 determinant residue admitted & determinant route becomes regulator-safe
candidate & measured Higgs/top masses\\
top determinant and matching closed & Higgs-lightness mechanism can be claimed
within Branch A & full Tau Core or TOE validation\\
\bottomrule
\end{{tabular}}
\end{{center}}

At the present stage the paper sits at the first level and partially organizes
the next levels.  It does not yet license the later claims.

\section{{Near-Term Falsifiable Prediction}}
The current module predicts a narrow kind of future test rather than an already validated signal. If the top/flavor deformation gate is completed with a natural spurion $\delta_\star\sim10^{{-3}}$, the same equations imply a parent scale $M_\tau$ in the multi-TeV range and a Higgs-sector spectral threshold
\begin{{equation}}
m_{{\rm gap}}\simeq\nu_D M_\tau \sim 2\text{{--}}4\,{{\rm TeV}}.
\end{{equation}}
The associated Higgs-coupling deviations should be parametrically of order
\begin{{equation}}
\frac{{\Delta g}}{{g}}\sim\frac{{v^2}}{{m_{{\rm gap}}^{{2}}}}\sim10^{{-3}}\text{{--}}10^{{-2}} .
\end{{equation}}
This is a falsifiable target, not a confirmed prediction. The module weakens if the completed determinant forces much larger deviations, no TeV-sector threshold, or collider-excluded states.

\section{{Validation Gates}}
The module would fail if any of the following gates fail:
\begin{{itemize}}
\item the Branch A projection metric rule $\nu_i=3|Y_i|/5$ cannot be forced from the parent action or the F1--F8 sufficient conditions fail;
\item the two-block / $3+2$ / invariant-role chain cannot be derived from the parent action or stability structure;
\item the $3\times2$ bridge and anomaly-safe visible target cannot be obtained without post-hoc representation choice;
\item the physical $Q_\mu$ operator cannot be derived from the same wall-Hessian/domain package;
\item the parent quartic is not canonical or requires large tuning;
\item the projection-BRST quotient is anomalous or regulator-dependent;
\item a visible Higgs mass is unavoidable at $\delta_\star=0$;
\item the top determinant produces a large mismatch-independent or linear mass term, or requires an inserted Yukawa hierarchy;
\item $\epsilon_\tau$, $C_A$, or the wall-cell embedding must be chosen from
the measured Higgs vev or top mass;
\item the TeV-sector spectral window is excluded by Higgs coupling or direct-search constraints.
\end{{itemize}}

\section{{Conclusion}}
The Branch A Higgs module links a $3+2$ stabilizer, hypercharge normalization, a $\operatorname{{sech}}^{{3/10}}$ visible zero mode, and a quartic overlap requiring an order-one parent quartic. The paper establishes a compact, reproducible mechanism target: if the Branch A localization rule is derived, the Higgs quartic is mapped to a natural parent-scale coupling rather than an extreme hierarchy. The newest parent-selection refinements make the $3+2$ input less arbitrary by linking it to two protected visible clusters, $\epsilon_3/\epsilon_2$ invariant roles, and an invariant/anomaly bridge. The localization-forcing update moves the rule $\nu_i=3|Y_i|/5$ from a standalone postulate to a conditional F1--F8 parent-action forcing route. The v0.2 single-package update now places the compact tau cell, induced wall domain, same-domain $Q$/regulator package, top/Yukawa overlap, and absolute $\epsilon_\tau$ scale blocker in one scaffold. This strengthens the mechanism because the remaining free pieces are no longer diffuse. The hard blockers are now sharply identified: an explicit final parent action, continuum/UV completion, a concrete compact tau-cell geometry, QFT-level BRST/anomaly/regulator closure, top-determinant radiative stability, and physical matching. The manuscript still does not establish the full parent theory, the measured Higgs vev, the measured top mass, the Standard Model, or the Higgs hierarchy solution. Its current contribution is narrower: it reduces the Higgs proof problem to the conditional chain $S_{{\rm parent}}\Rightarrow$ Branch A gates $\Rightarrow$ SM-compatible Higgs sector, and makes the failure modes explicit. The next paper-grade step is to turn the scaffold into a microscopic parent action, derive the compact-cell and scale ingredients from it, complete the projection-BRST and physical top-determinant gates, and only then compare the implied heavy-sector window against collider constraints.

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def references_bib() -> str:
    return r"""@article{ATLAS2012,
  title = {Observation of a new particle in the search for the Standard Model Higgs boson with the ATLAS detector at the LHC},
  author = {{ATLAS Collaboration}},
  journal = {Physics Letters B},
  volume = {716},
  pages = {1--29},
  year = {2012},
  doi = {10.1016/j.physletb.2012.08.020}
}

@article{CMS2012,
  title = {Observation of a new boson at a mass of 125 GeV with the CMS experiment at the LHC},
  author = {{CMS Collaboration}},
  journal = {Physics Letters B},
  volume = {716},
  pages = {30--61},
  year = {2012},
  doi = {10.1016/j.physletb.2012.08.021}
}

@article{Higgs1964,
  title = {Broken symmetries and the masses of gauge bosons},
  author = {Higgs, Peter W.},
  journal = {Physical Review Letters},
  volume = {13},
  pages = {508--509},
  year = {1964},
  doi = {10.1103/PhysRevLett.13.508}
}

@article{GeorgiGlashow1974,
  title = {Unity of all elementary-particle forces},
  author = {Georgi, Howard and Glashow, S. L.},
  journal = {Physical Review Letters},
  volume = {32},
  pages = {438--441},
  year = {1974},
  doi = {10.1103/PhysRevLett.32.438}
}

@article{Connes1996,
  title = {Gravity coupled with matter and the foundation of non-commutative geometry},
  author = {Connes, Alain},
  journal = {Communications in Mathematical Physics},
  volume = {182},
  pages = {155--176},
  year = {1996},
  doi = {10.1007/BF02506388}
}

@article{ChamseddineConnes1997,
  title = {The spectral action principle},
  author = {Chamseddine, Ali H. and Connes, Alain},
  journal = {Communications in Mathematical Physics},
  volume = {186},
  pages = {731--750},
  year = {1997},
  doi = {10.1007/s002200050126}
}

@article{ConnesMarcolli2006,
  title = {Noncommutative geometry and the standard model with neutrino mixing},
  author = {Connes, Alain and Marcolli, Matilde},
  journal = {Journal of High Energy Physics},
  volume = {2006},
  number = {11},
  pages = {081},
  year = {2006},
  eprint = {hep-th/0608226},
  archivePrefix = {arXiv},
  doi = {10.1088/1126-6708/2006/11/081}
}

@article{JackiwRebbi1976,
  title = {Solitons with fermion number 1/2},
  author = {Jackiw, R. and Rebbi, C.},
  journal = {Physical Review D},
  volume = {13},
  pages = {3398--3409},
  year = {1976},
  doi = {10.1103/PhysRevD.13.3398}
}

@article{RubakovShaposhnikov1983,
  title = {Do we live inside a domain wall?},
  author = {Rubakov, V. A. and Shaposhnikov, M. E.},
  journal = {Physics Letters B},
  volume = {125},
  pages = {136--138},
  year = {1983},
  doi = {10.1016/0370-2693(83)91253-4}
}

@article{Kaplan1992,
  title = {A method for simulating chiral fermions on the lattice},
  author = {Kaplan, David B.},
  journal = {Physics Letters B},
  volume = {288},
  pages = {342--347},
  year = {1992},
  eprint = {hep-lat/9206013},
  archivePrefix = {arXiv},
  doi = {10.1016/0370-2693(92)91112-M}
}

@article{RandallSundrum1999,
  title = {A large mass hierarchy from a small extra dimension},
  author = {Randall, Lisa and Sundrum, Raman},
  journal = {Physical Review Letters},
  volume = {83},
  pages = {3370--3373},
  year = {1999},
  eprint = {hep-ph/9905221},
  archivePrefix = {arXiv},
  doi = {10.1103/PhysRevLett.83.3370}
}

@article{Becchi1976,
  title = {Renormalization of gauge theories},
  author = {Becchi, C. and Rouet, A. and Stora, R.},
  journal = {Annals of Physics},
  volume = {98},
  pages = {287--321},
  year = {1976},
  doi = {10.1016/0003-4916(76)90156-1}
}

@article{KugoOjima1979,
  title = {Local Covariant Operator Formalism of Non-Abelian Gauge Theories and Quark Confinement Problem},
  author = {Kugo, Taichiro and Ojima, Izumi},
  journal = {Progress of Theoretical Physics Supplement},
  volume = {66},
  pages = {1--130},
  year = {1979},
  doi = {10.1143/PTPS.66.1}
}
"""


def full_derivation_tex() -> str:
    i4_d = i4(NU_D)
    lambda_tau_required = PHYSICAL_LAMBDA_H / i4_d
    toy_mass = Y_T_PARENT / math.sqrt(2.0)
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,hyperref,xurl,longtable,array}}
\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\title{{Full Derivation Ledger for the Branch A Higgs Module}}
\author{{Jozsef Olcsak}}
\date{{May 20, 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This companion ledger expands the derivation chain behind Paper 4.  It is not
an empirical paper, not a complete Standard Model derivation, and not a claim
that the Higgs sector has been solved.  Its purpose is to collect the working
Branch A gate stack in one generated PDF: stabilizer normalization,
localization forcing, quartic overlap, Yukawa product, bridge-trace
normalization, v0.2 single-package parent action, absolute-scale gate,
wall-cell embedding, and the remaining top/BRST/regulator blockers.
\end{{abstract}}

\tableofcontents

\section{{Status Discipline}}
This ledger uses four statuses.

\begin{{longtable}}{{p{{0.22\linewidth}}p{{0.68\linewidth}}}}
\toprule
\textbf{{Status}} & \textbf{{Meaning}}\\
\midrule
Derived in gate & Follows inside the stated Branch A assumptions or audit.\\
Theorem-candidate & Would become a theorem if the listed parent conditions are derived.\\
Open gate & Required before a physical claim can be made.\\
Forbidden overclaim & A claim explicitly not licensed by the current construction.\\
\bottomrule
\end{{longtable}}

The guiding rule is simple: no quantity may be chosen from the observed Higgs
vev, measured Higgs mass, or top mass and then presented as derived.

\section{{Global Dependency Chain}}
The current Higgs-route dependency chain is
\begin{{equation}}
3+2
\rightarrow
T_Y
\rightarrow
\kappa_\tau^2=\frac35
\rightarrow
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x
\rightarrow
\nu_i=\kappa_\tau^2|Y_i|
\rightarrow
I_4(3/10)
\rightarrow
{{\cal I}}_{{qHu}}
\rightarrow
y_t^{{\rm parent}}
\rightarrow
\Lambda_A
\rightarrow
m_t^{{\rm candidate}}.
\end{{equation}}
The first part is dimensionless and comparatively sharp.  The last part is a
scale/matching problem and remains open.

\section{{Branch A Stabilizer And Hypercharge Normalization}}
Start from the visible carrier
\begin{{equation}}
V\simeq \mathbb{{C}}^5=C\oplus W,
\qquad
\dim C=3,
\quad
\dim W=2.
\end{{equation}}
The traceless two-block generator is
\begin{{equation}}
T_\Sigma=\frac1{{\sqrt{{60}}}}{{\rm diag}}(2,2,2,-3,-3).
\end{{equation}}
The stabilizer is $S(U(3)\times U(2))$.  A stabilizer-preserving Abelian
connection must be block-scalar:
\begin{{equation}}
T={{\rm diag}}(a,a,a,b,b),
\qquad
3a+2b=0.
\end{{equation}}
Thus the traceless centralizer is one-dimensional.  Choosing the conventional
hypercharge orientation gives
\begin{{equation}}
Y={{\rm diag}}\left(-\frac13,-\frac13,-\frac13,\frac12,\frac12\right),
\qquad
T_Y=\sqrt{{\frac35}}Y,
\end{{equation}}
and $T_\Sigma=-T_Y$.  The Branch A normalization is therefore
\begin{{equation}}
\kappa_\tau^2=\frac35.
\end{{equation}}
This is not fitted to the Higgs quartic.  It comes from canonical normalization
of the unique traceless Abelian line in the $3+2$ stabilizer.

\section{{Unoriented Line And Two-Vacuum Quotient}}
The Branch A Abelian direction is treated as an internal line rather than as a
signed observable vector:
\begin{{equation}}
L_Y={{\rm span}}(T_Y).
\end{{equation}}
If the regular parent datum is the line or projector, then
\begin{{equation}}
{{\rm span}}(T_Y)={{\rm span}}(-T_Y),
\qquad
P_Y=P_{{-Y}},
\qquad
\Sigma\sim-\Sigma.
\end{{equation}}
For $\Sigma=fT_Y$, sign-even invariants satisfy
\begin{{equation}}
{{\rm Tr}}(\Sigma^2)=\frac12 f^2,
\qquad
{{\rm Tr}}(\Sigma'^2)=\frac12(f')^2.
\end{{equation}}
If the regular branch fixes $2{{\rm Tr}}(\Sigma^2)=1$, then $f^2=1$ and the
two oriented representatives are $f=\pm1$.  This gives the two-vacuum logic
conditional on the parent deriving the unoriented-line/projector quotient.

\section{{Minimal Wall Profile}}
The leading local two-vacuum wall is modeled by the BPS equation
\begin{{equation}}
f'(x)=1-f(x)^2,
\end{{equation}}
whose centered solution is
\begin{{equation}}
f(x)=\tanh x.
\end{{equation}}
Equivalently, it is generated by
\begin{{equation}}
V(f)=\frac12(1-f^2)^2.
\end{{equation}}
The wall is not selected to fit the Higgs quartic.  The open parent-level task
is to derive this minimal BPS functional and forbid leading deformations that
change the localization rule.

\section{{Localization Derivation}}
Given the selected hypercharge wall line, the local first-order operator is
\begin{{equation}}
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x.
\end{{equation}}
The zero-mode equation gives
\begin{{align}}
Q_i h_i&=0,\\
\partial_x h_i&=-\kappa_\tau^2Y_i\tanh x\,h_i,\\
h_i(x)&\propto
\exp\left[-\int^x\kappa_\tau^2Y_i\tanh u\,du\right]
={{\rm sech}}^{{\kappa_\tau^2Y_i}}x.
\end{{align}}
Normalizability selects the chirality/sign, so
\begin{{equation}}
\nu_i=\kappa_\tau^2|Y_i|.
\end{{equation}}
For the Higgs doublet,
\begin{{equation}}
\nu_H=\frac35\left|\frac12\right|=\frac3{{10}}.
\end{{equation}}
This is the current natural-origin derivation.  It becomes a proof only when
the parent action derives the wall line, BPS wall, same-domain Hessian, and
chiral admissibility.

\section{{F1--F8 Forcing Theorem Candidate}}
The stronger theorem-candidate says the localization rule is forced if the
parent action derives F1--F8.

\begin{{longtable}}{{p{{0.10\linewidth}}p{{0.36\linewidth}}p{{0.42\linewidth}}}}
\toprule
\textbf{{Gate}} & \textbf{{Requirement}} & \textbf{{Consequence}}\\
\midrule
F1 & quotient-visible trace removal & identity direction cannot become a wall channel\\
F2 & two protected visible blocks & exactly two low-energy visible modules\\
F3 & stabilizer-preserving wall & wall is block-scalar\\
F4 & one traceless Abelian wall line & no extra Abelian wall freedom\\
F5 & universal trace pairing & fixes $\kappa_\tau^2=3/5$\\
F6 & unoriented two-vacuum quotient & gives the $\pm$ wall representatives\\
F7 & minimal BPS factorization & gives leading $f=\tanh x$\\
F8 & same-domain Hessian/Q factorization & forbids importing $Q_i$ by hand\\
\bottomrule
\end{{longtable}}

Under these conditions the operator $Q_i$ and the exponent $\nu_i$ follow at
leading order.  Still open: deriving F1--F8 from the final parent action and
stationary background.

\section{{Minimal Forcing Parent-Action Candidate}}
The current action-level scaffold behind F1--F8 contains:
\begin{{itemize}}
\item trace projection $\Pi_{{\rm tr}}$ to remove the identity direction;
\item a stabilizer-stratified wall sector;
\item positive transverse/off-block gaps;
\item an unoriented-line quotient sector;
\item a minimal BPS wall term;
\item a same-domain Hessian/Q admissibility term.
\end{{itemize}}
It forces the trace quotient, trace pairing, leading BPS wall, and local
localization consequence at gate level.  It does not yet fully force the two
protected visible modules, the deep origin of the unoriented-line quotient, or
continuum QFT closure.

\section{{Quartic Overlap Calculation}}
For
\begin{{equation}}
h_\nu(x)=N_\nu{{\rm sech}}^{{\nu}}x,
\qquad
\int h_\nu^2dx=1,
\end{{equation}}
the normalization is
\begin{{equation}}
N_\nu^2=\frac{{\Gamma(\nu+1/2)}}{{\sqrt\pi\Gamma(\nu)}}.
\end{{equation}}
The quartic overlap is
\begin{{equation}}
I_4(\nu)
=
\frac{{\Gamma(\nu+1/2)^2\Gamma(2\nu)}}
{{\sqrt\pi\Gamma(\nu)^2\Gamma(2\nu+1/2)}}.
\end{{equation}}
At $\nu_H=3/10$,
\begin{{equation}}
I_4(3/10)={i4_d:.15f}.
\end{{equation}}
For $\lambda_H\simeq0.129$,
\begin{{equation}}
\lambda_\tau\simeq {lambda_tau_required:.9f}.
\end{{equation}}
The safe claim is order-one parent quartic, not a derived canonical quartic.

\section{{Sensitivity Audit}}
The generator also writes a sensitivity audit.  In the moderate window
$0.25\leq\nu\leq0.35$, the required parent quartic remains order one.  This
reduces numerology risk: the mechanism is not a delta-function coincidence at
$\nu=0.3$.  The audit still does not prove the localization rule.

\section{{Yukawa Product Cohomology Gate}}
The up-type slot is treated as a parent cohomology product, not a post-hoc
operator insertion:
\begin{{equation}}
[q_L]\,[H]\,[u_R^c].
\end{{equation}}
It is admissible only if it is neutral, bridge-closed, Q-closed modulo Q-exact
terms, and defined on the same Branch A domain as the wall/Hessian package.
The gate does not derive family hierarchy or the measured top mass.

\section{{Yukawa Overlap}}
The fixed exponents are
\begin{{equation}}
\nu_q=\frac1{{10}},
\qquad
\nu_H=\frac3{{10}},
\qquad
\nu_u=\frac25.
\end{{equation}}
With
\begin{{equation}}
I(a)=\int_{{\mathbb R}}{{\rm sech}}^a x\,dx
=\sqrt\pi\frac{{\Gamma(a/2)}}{{\Gamma((1+a)/2)}},
\end{{equation}}
the normalized overlap is
\begin{{equation}}
{{\cal I}}_{{qHu}}
=
N_{{\nu_q}}N_{{\nu_H}}N_{{\nu_u}}
I(\nu_q+\nu_H+\nu_u)
=
{I_QHU:.15f}.
\end{{equation}}
This is reproducible and endpoint-free.

\section{{$g_t$ Bridge-Trace Normalization}}
The current normalization theorem-candidate sums over the full $3\times2$
shared bridge and multiplies by the canonical wall coupling:
\begin{{equation}}
g_t=(3\times2)\kappa_\tau^2=6\cdot\frac35=\frac{{18}}{{5}}.
\end{{equation}}
Therefore
\begin{{equation}}
y_t^{{\rm parent}}
=g_t{{\cal I}}_{{qHu}}
={Y_T_PARENT:.15f}.
\end{{equation}}
This is close to an order-one top Yukawa, but it is not a top-mass prediction.
The measured value still requires $v_{{\rm eff}}$, $R_t$, family selection,
and QFT matching.

\section{{$v_{{\rm eff}}$ From Stabilizer Gap}}
The current electroweak-scale gate proposes
\begin{{equation}}
v_{{\rm eff}}=\frac{{M_{{\rm stab}}}}{{g_{{\rm vis}}}}.
\end{{equation}}
Here $M_{{\rm stab}}$ must be the parent-derived positive Hessian/stabilizer
gap protecting the Higgs-like residue, and $g_{{\rm vis}}$ must be a universal
visible gauge normalization.  This is better than inserting $246$ GeV, but it
is not yet a derivation of the measured vev.

\section{{Running And Matching}}
The admissible matching factor is
\begin{{equation}}
R_t=R_{{\rm match}}R_{{\rm run}},
\qquad
R_{{\rm run}}=\exp\left[-\int \gamma_t(\mu)d\ln\mu\right].
\end{{equation}}
This is a shape requirement only.  $\gamma_t$, $R_{{\rm match}}$, regulator,
and matching scale must come from the same QFT closure before a physical top
mass can be claimed.

\section{{Dimensional Obstruction}}
The Branch A action currently fixes dimensionless data.  A dimensionless action
cannot output an absolute GeV scale.  Therefore
\begin{{equation}}
\Lambda_A=s_A\hat E_{{\rm wall}}
\end{{equation}}
requires one remaining absolute unit $s_A$.

\section{{$s_A$ And Compact Tau-Cell Unit}}
The most native route is
\begin{{equation}}
s_A=C_AE_\tau,
\qquad
E_\tau=\epsilon_\tau\mu_\tau({{\cal K}}_\tau)N_\tau.
\end{{equation}}
This route is admissible only if the compact cell, its measure, the count/index
density, and $\epsilon_\tau$ are selected before Higgs/top comparison.

\section{{Compact Tau Geometry Target}}
The compact cell must become a concrete object rather than a name.  The
minimal target data are
\begin{{equation}}
({{\cal K}}_\tau,{{\cal A}}_\tau,{{\cal H}}_\tau,D_\tau,\mu_\tau),
\end{{equation}}
where ${{\cal A}}_\tau$ is an internal algebra, ${{\cal H}}_\tau$ is the
domain, $D_\tau$ is the spectral operator, and $\mu_\tau$ is the trace or
measure.  A useful candidate must produce the Branch A wall as a defect,
quotient, or induced Hessian sector of this same package:
\begin{{equation}}
{{\cal W}}_A={{\rm Defect}}({{\cal K}}_\tau,D_\tau,T_Y)
\quad{{\rm or}}\quad
{{\cal W}}_A={{\cal K}}_\tau/{{\cal H}}_A .
\end{{equation}}
The decisive check is whether the low-lying spectral/Hessian data of
${{\cal K}}_\tau$ select $3+2$, the hypercharge line, a two-vacuum kink, the
same-domain $Q$ package, and a finite determinant regulator without inserting
those choices by hand.

\section{{C0 Finite-Fiber Wall Spectral Cell}}
The first explicit compact-geometry candidate uses a compact local wall cell
with finite internal fiber:
\begin{{equation}}
{{\cal H}}_\tau=L^2({{\cal K}}_\tau)\otimes\mathbb{{C}}^5\otimes\mathbb{{C}}^2_{{\rm wall}},
\qquad
\mathbb{{C}}^5=C\oplus W,\quad \dim C=3,\quad \dim W=2.
\end{{equation}}
Preserving this two-block fiber gives $S(U(3)\times U(2))$.  The traceless
block-scalar centralizer is one-dimensional:
\begin{{equation}}
T=\operatorname{{diag}}(a,a,a,b,b),\qquad 3a+2b=0,
\end{{equation}}
which is the hypercharge line after canonical normalization.

The regular Abelian datum is taken to be the unoriented projector
\begin{{equation}}
P_Y=\frac{{T_Y\otimes T_Y}}{{{{\rm Tr}}(T_Y^2)}} ,
\end{{equation}}
so $T_Y$ and $-T_Y$ define the same parent line.  The quotient-visible wall
field $\Sigma=fT_Y$ then has the minimal BPS functional
\begin{{equation}}
E[f]=\int\left[\frac12(f')^2+\frac12(1-f^2)^2\right]dx,
\end{{equation}}
whose centered solution is $f(x)=\tanh x$.  The same wall-Hessian factor gives
\begin{{equation}}
Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x,
\qquad
\nu_i=\kappa_\tau^2|Y_i|.
\end{{equation}}
Because ${{\cal K}}_\tau$ is compact before the local-wall limit, the spectrum
of the parent $D_\tau$ is discrete and can in principle define the same
heat-kernel or zeta regulator used by the determinant gate:
\begin{{equation}}
\zeta_{{D_\tau}}(s)=\sum_{{\lambda_n>0}}\lambda_n^{{-s}}.
\end{{equation}}
This is the first worked compact tau geometry route.  Its remaining blocker is
that a final $S_{{\rm parent}}$ must select the two-block cell and its wall
functional rather than inheriting them as useful choices.

\section{{Adversarial Audit Of C0}}
C0 survives only as a candidate.  The attack map is:
\begin{{itemize}}
\item the $C^3\oplus C^2$ fiber is still inserted; after insertion the
stabilizer and hypercharge line follow, but parent selection remains open;
\item the minimal wall functional is still chosen; after choice it gives
$f(x)=\tanh x$, but the final action must force this functional or an
equivalent BPS system;
\item the line-projector quotient is assumed; after assumption it gives
$\Sigma\sim-\Sigma$, but the parent regularity principle must choose projector
data over signed-vector data;
\item the compact spectrum gives a natural regulator route, but no physical
top determinant, anomaly closure, or Ward identity has yet been computed;
\item $\epsilon_\tau$ remains a scale gate, not a derived number.
\end{{itemize}}
The C0 result is therefore not proof.  It is a useful failure-localization
device: the next C1 step must derive the two-block fiber and wall functional
from a parent extremal, index, or stability principle.

\section{{C1 And C2 Forcing Routes}}
C1 addresses the two inserted C0 structures.  The $3+2$ fiber is no longer
treated as a free five-dimensional choice if the parent action requires exactly
two protected visible roles: a primitive closure role and a primitive pairing
role.  The minimal dimensions are then
\begin{{equation}}
\dim C=3,\qquad \dim W=2,
\end{{equation}}
because the first nontrivial oriented closure tensor is $\epsilon_3$ and the
first primitive pairing tensor is $\epsilon_2$.  The BPS wall is no longer a
free potential if the parent line datum is unoriented, local, reflection-even,
finite-tension, and in the unit two-vacuum sector.  The leading Bogomolny
completion is
\begin{{equation}}
E[f]=\int\frac12\left(f'-(1-f^2)\right)^2dx+
\left[ f-\frac13 f^3\right]_{{-\infty}}^{{+\infty}},
\end{{equation}}
so the minimizer satisfies $f'=1-f^2$ and therefore $f(x)=\tanh x$.

C2 addresses the remaining C0 attacks.  Parent sign-gauge regularity would
force the unoriented projector:
\begin{{equation}}
T_Y\mapsto -T_Y,\qquad
{{\cal A}}_{{\rm phys}}={{\cal A}}_\tau^{{\mathbb Z_2}},
\qquad
P_Y=\frac{{T_Y\otimes T_Y}}{{{{\rm Tr}}(T_Y^2)}} .
\end{{equation}}
Same-domain QFT closure would make the top-sensitive determinant a zeta
regularized shared-domain object:
\begin{{equation}}
\Delta_{{\rm top}}(s)=
\frac{{\det_\zeta(Q_t^\dagger Q_t+\delta_\star W_t)}}
{{\det_\zeta(Q_tQ_t^\dagger)}} .
\end{{equation}}
Finally, the absolute scale would have to be computed from compact-cell
spectral density rather than endpoint fitting:
\begin{{equation}}
\epsilon_\tau
=
\frac{{1}}{{\mu_\tau({{\cal K}}_\tau)}}
{{\rm Tr}}_{{{{\cal H}}_\tau}}\left(\chi(D_\tau^2/\Lambda_\tau^2)D_\tau^2\right),
\end{{equation}}
with $\chi$ fixed before Higgs/top comparison.  These are still
theorem-candidate routes.  They are stronger than C0 because the failure modes
are now explicit and calculational.

\section{{C3 Explicit Geometry Candidate}}
The C3 candidate declares the compact spectral geometry explicitly:
\begin{{equation}}
{{\cal K}}_\tau=S^1/\mathbb Z_2,\qquad
{{\cal H}}_\tau=
L^2({{\cal K}}_\tau)\otimes
(\mathbb C^3\oplus\mathbb C^2)\otimes\mathbb C^2_{{\rm wall}},
\end{{equation}}
\begin{{equation}}
{{\cal A}}_\tau=
\left(C^\infty({{\cal K}}_\tau)\otimes
(M_3(\mathbb C)\oplus M_2(\mathbb C))\right)^{{\mathbb Z_2}} .
\end{{equation}}
The determinant-one unitary sector gives $S(U(3)\times U(2))$, and the
traceless block-scalar centralizer gives the hypercharge line.  The local wall
operator is
\begin{{equation}}
D_\tau=\sigma_1\partial_x+\sigma_2\kappa_\tau^2T_Y f(x)+D_{{\rm gap}},
\end{{equation}}
with positive $D_{{\rm gap}}$ on leakage modes.  The leading local even
finite-tension wall action on the orbifold line is
\begin{{equation}}
E[f]=\int\left[\frac12(f')^2+\frac12(1-f^2)^2\right]dx,
\end{{equation}}
so the large-cell local solution is $f(x)=\tanh x$.  Component restriction gives
$Q_i=\partial_x+\kappa_\tau^2Y_i\tanh x$ and
$\nu_i=\kappa_\tau^2|Y_i|$.

C3 is the first explicit $({{\cal K}}_\tau,{{\cal A}}_\tau,{{\cal H}}_\tau,D_\tau,\mu_\tau)$
candidate.  It realizes the C0--C2 architecture in one object, but it is still
not a proof until the final $S_{{\rm parent}}$ selects this orbifold base,
two-role fiber, even algebra, and wall action.

\section{{Adversarial Audit Of C3}}
C3 exposes rather than hides the remaining inputs.  The attack points are:
\begin{{itemize}}
\item $S^1/\mathbb Z_2$ is still selected as the compact base;
\item $\mathbb C^3\oplus\mathbb C^2$ is still selected as the finite fiber;
\item the even block algebra is still selected as the observable algebra;
\item $D_{{\rm gap}}$ is still declared positive on leakage modes;
\item the minimal wall action is still selected;
\item the cutoff profile $\chi$ is still a regulator input.
\end{{itemize}}
The next C4 gate must replace these selections by a parent selection
functional.  A useful target is
\begin{{equation}}
{{\cal S}}_{{\rm sel}}=
I_{{\rm compact}}+I_{{\rm gap}}+I_{{\rm cohom}}+I_{{\rm anom}}+I_{{\rm wall}}+I_{{\rm reg}},
\end{{equation}}
whose minimum or protected extremum should select the C3 data without Higgs/top
endpoint information.  Until that is shown, C3 is an explicit geometry
candidate, not a final derivation.

\section{{C4 Penalty Interpretation}}
The selection functional is intended as a sum of non-negative gates.  C3 would
be selected only if it drives all of them to zero without tuned weights:
\begin{{itemize}}
\item $I_{{\rm compact}}=0$: compact trace-regular spectral data;
\item $I_{{\rm gap}}=0$: no extra protected visible sectors;
\item $I_{{\rm cohom}}=0$: one primitive closure role and one primitive pairing role;
\item $I_{{\rm anom}}=0$: anomaly-safe bridge closure;
\item $I_{{\rm wall}}=0$: finite-tension two-vacuum Bogomolny sector;
\item $I_{{\rm reg}}=0$: same-domain Q-paired determinant regulator.
\end{{itemize}}
C4 is not yet minimized in this ledger.  It is a precise target: prove that C3
is a protected extremum, or find a simpler compact geometry that beats it.

\section{{C5 Local Spectrum Gate}}
The local-wall spectrum is now explicit.  For
\begin{{equation}}
Q_\nu=\partial_x+\nu\tanh x,
\qquad
Q_\nu^\dagger=-\partial_x+\nu\tanh x,
\end{{equation}}
the partner Hamiltonians are
\begin{{align}}
H_-&=Q_\nu^\dagger Q_\nu
=-\partial_x^2+\nu^2-\nu(\nu+1){{\rm sech}}^2x,\\
H_+&=Q_\nu Q_\nu^\dagger
=-\partial_x^2+\nu^2-\nu(\nu-1){{\rm sech}}^2x.
\end{{align}}
The zero mode obeys
\begin{{equation}}
Q_\nu h_\nu=0,\qquad h_\nu=N_\nu{{\rm sech}}^\nu x.
\end{{equation}}
The positive spectra of $H_-$ and $H_+$ are paired by $Q_\nu$ and
$Q_\nu^\dagger$, while the unpaired part is the zero-mode/index residue.  The
local continuum threshold is
\begin{{equation}}
\lambda_{{\rm cont}}=\nu^2.
\end{{equation}}
For $\nu_D=3/10$, this gives
\begin{{equation}}
\lambda_{{\mathrm{{cont}},D}}=9/100={NU_D_CONTINUUM_THRESHOLD:.6f}
\end{{equation}}
This is a real spectrum calculation in the local wall limit.  The compact
$S^1/\mathbb Z_2$ eigenvalue problem, boundary-condition quantization, zeta
determinant, and regulator matching remain open.

\section{{C6 Compact Spectrum Boundary-Value Problem}}
The compact problem is now formulated.  On $x\in[0,L]$, solve
\begin{{equation}}
H_\pm\psi_n=\lambda_n\psi_n
\end{{equation}}
with self-adjoint orbifold boundary conditions that preserve the Q-pairing:
\begin{{equation}}
Q_\nu:{{\cal D}}(H_-)\to{{\cal D}}(H_+),
\qquad
Q_\nu^\dagger:{{\cal D}}(H_+)\to{{\cal D}}(H_-).
\end{{equation}}
The compact zeta functions are
\begin{{equation}}
\zeta_\pm(s)=\sum_{{\lambda_n^\pm>0}}(\lambda_n^\pm)^{{-s}},
\end{{equation}}
and the determinant gate is
\begin{{equation}}
\log\Delta_\nu=-\zeta_-'(0)+\zeta_+'(0),
\end{{equation}}
with zero-mode and index terms kept separately.  This is a formulated
boundary-value problem, not yet a solved compact determinant.  The decisive
test is whether the positive spectra remain paired after compact
boundary-condition quantization.

\section{{C7 Competitor Audit}}
The C7 ledger asks whether the C3 finite fiber is an arbitrary input or the
first low-complexity survivor of the C4 penalties.  The screen is:
\begin{{itemize}}
\item one-block fibers fail the required closure/pairing split;
\item $C^1\oplus C^1$ is too small for a non-Abelian closure carrier;
\item $C^2\oplus C^2$ has a pairing-like block but no primitive rank-three closure role;
\item $C^3\oplus C^1$ has closure support but no primitive pairing block;
\item $C^3\oplus C^2$ is the first split that carries one closure role and one pairing role with minimal dimensions;
\item larger splits are admissible only if an additional gap principle removes the extra protected visible freedom.
\end{{itemize}}
Therefore C7 does not prove uniqueness.  It converts the hidden $3+2$ choice
into a falsifiable minimality statement: C3 must be the first survivor of the
C4 cohomology/gap screen, and any lower-complexity survivor would falsify this
selection route.

\section{{C8 Minimality Theorem Candidate}}
C8 records the theorem-candidate behind the competitor audit.  Assume:
\begin{{enumerate}}
\item a finite visible fiber splits into exactly two protected roles,
${{\cal F}}={{\cal F}}_{{\rm cl}}\oplus{{\cal F}}_{{\rm pair}}$;
\item the closure role needs a primitive rank-three carrier;
\item the pairing role needs a primitive rank-two carrier;
\item the gap gate forbids extra protected light blocks;
\item the cohomology gate forbids identifying the two roles in a single block.
\end{{enumerate}}
Then the first admissible finite survivor is
\begin{{equation}}
{{\cal F}}_{{\min}}\simeq C^3\oplus C^2.
\end{{equation}}
This is not a final proof because the rank-three closure requirement and the
rank-two pairing requirement must still be derived from the microscopic parent
action.  But it converts the C3 input into a precise target: prove those two
minimal role constraints, or find a lower-rank counterexample.

\section{{C9 Compact Spectrum Pilot}}
The full compact spectrum remains open, but the packet now contains a first
finite-box pilot:
\begin{{equation}}
H_-=-\partial_x^2+\nu^2-\nu(\nu+1)\operatorname{{sech}}^2x,\qquad
H_+=-\partial_x^2+\nu^2-\nu(\nu-1)\operatorname{{sech}}^2x .
\end{{equation}}
The generated CSV \texttt{{paper4\_compact\_spectrum\_pilot\_v01.csv}} records
the first finite-difference eigenvalues for $H_\pm$ at $\nu_D=3/10$ on a
Dirichlet finite box.  This is intentionally not promoted to a theorem.  It is
a reproducible operational pilot whose failure mode is explicit: Dirichlet
box boundaries need not preserve the exact $Q_\nu,Q_\nu^\dagger$ pairing.  The
theorem-level replacement must solve the compact $S^1/\mathbb Z_2$ boundary
problem with Q-paired self-adjoint domains and then define the zeta determinant
from that same spectrum.

\section{{C10 Uniqueness Proof-Obligation Split}}
C10 separates the uniqueness problem into layers:
\begin{{enumerate}}
\item finite-dimensional algebra lemmas: rank-three closure minimality,
rank-two pairing minimality, and obstruction to merging both roles into one
protected block;
\item parent-action selection: no extra protected visible block and no
lower-complexity survivor;
\item compact spectral geometry: self-adjoint Q-paired orbifold spectrum;
\item index/anomaly/QFT closure: index residue, Ward identities, regulator
consistency, and determinant finiteness on the same domain.
\end{{enumerate}}
This is why C8 is still a theorem-candidate.  The first layer may be attacked
as finite-dimensional algebra.  The later layers require the explicit parent
action and compact spectral geometry.  A real uniqueness theorem must not mix
proofs from incompatible domains.

\section{{C11 Algebraic Minimality Lemmas}}
The first algebraic layer is now explicit:
\begin{{itemize}}
\item if the closure datum is a primitive alternating three-form
$\omega_{{\rm cl}}\in\Lambda^3V_{{\rm cl}}^*$, then rank one and rank two cannot
carry it because $\Lambda^3V^*=0$; the minimal closure carrier is $C^3$;
\item if the pairing datum is a nondegenerate alternating two-form
$\omega_{{\rm pair}}\in\Lambda^2V_{{\rm pair}}^*$, then rank one cannot carry it
because $\Lambda^2V^*=0$; the minimal pairing carrier is $C^2$.
\end{{itemize}}
Thus, conditional on the parent theory selecting exactly these two primitive
protected forms and forbidding extra light blocks, the first finite carrier is
$C^3\oplus C^2$.  The remaining open work is not this elementary exterior
algebra; it is the parent-action derivation of why these are the protected
forms in the first place.

\section{{C12 Protected-Form Selection Gate}}
C12 names the next blocker.  C11 is conditional on two protected exterior forms:
\begin{{equation}}
\omega_{{\rm cl}}\in\Lambda^3V_{{\rm cl}}^*,
\qquad
\omega_{{\rm pair}}\in\Lambda^2V_{{\rm pair}}^* .
\end{{equation}}
The parent theory must select these forms through symmetry, cohomology,
stationary action, or index obstruction.  It is not enough to choose them
because they generate the desired $3+2$ result.  The uniqueness theorem becomes
serious only if the same parent structure that selects the forms also supplies
the gap principle, compact spectrum, and determinant regulator.

\section{{C13 Cohomology/Index Selection Route}}
C13 proposes the first selection route: the two protected forms should be first
residues of parent complexes,
\begin{{equation}}
[\omega_{{\rm cl}}]\in H^3_{{\rm cl}},\qquad
[\omega_{{\rm pair}}]\in H^2_{{\rm pair}}.
\end{{equation}}
If lower-degree residues are trivial or gap-lifted, the degrees three and two
are selected by the parent cohomology/index structure rather than chosen after
the fact.  This remains open until the complexes, differentials, inner product,
compact domain, and index theorem are explicitly constructed.

\section{{C14 Toy Complex Sanity Check}}
A minimal toy complex can realize the desired pattern:
\begin{{equation}}
H^k_{{\rm cl}}=0\ (k<3),\quad H^3_{{\rm cl}}\simeq C[\omega_{{\rm cl}}],
\qquad
H^k_{{\rm pair}}=0\ (k<2),\quad H^2_{{\rm pair}}\simeq C[\omega_{{\rm pair}}].
\end{{equation}}
This only proves consistency of the target pattern.  It does not derive the
parent differential, the adjoint, the compact domain, or the index.  The next
real theorem must replace this toy complex with a parent-selected complex from
the compact tau-cell action.

\section{{C15 Operator-Domain Ansatz}}
The toy complex must become a closed operator problem.  For each branch
$a={{\rm cl}}$ or $a={{\rm pair}}$, the target data are
\begin{{equation}}
{{\cal H}}_a=\bigoplus_k{{\cal H}}^k_a,\qquad
d_a:{{\cal H}}^k_a\to{{\cal H}}^{{k+1}}_a,\qquad d_a^2=0,
\end{{equation}}
with adjoint $d_a^\dagger$ and Laplacian
\begin{{equation}}
\Delta_a=d_a^\dagger d_a+d_a d_a^\dagger .
\end{{equation}}
The protected residue must be harmonic and non-exact on the same compact
domain used by the wall/regulator/determinant package.  This is an operator
target, not a solved theorem.

\section{{C16 Shared-Domain Compatibility Gate}}
The wall operator, cohomology complex, index, zeta regulator, and determinant
must share one compact self-adjoint domain:
\begin{{equation}}
Q_\nu,\ d_a,\ d_a^\dagger,\ \Delta_a,\ D_\tau,\ \zeta_D(s),\ \det_\zeta .
\end{{equation}}
This prevents a false closure in which the zero mode, index, and determinant
are each computed with a different boundary condition.  C16 is a compatibility
gate; the actual domain classification remains open.

\section{{C17 Self-Adjoint Domain Candidate}}
The first candidate replacement for the C9 Dirichlet pilot is an orbifold
parity domain that preserves Q-pairing:
\begin{{equation}}
Q_\nu:{{\cal D}}_-\to{{\cal D}}_+,\qquad
Q_\nu^\dagger:{{\cal D}}_+\to{{\cal D}}_- .
\end{{equation}}
It must make $H_\pm$ self-adjoint, preserve positive-mode pairing, keep
zero-mode/index residue explicit, and feed the same spectrum to the zeta
determinant.  This is a domain candidate, not a classification theorem.

\section{{C18 Paired-Spectrum Verification Target}}
The required theorem is the positive-mode bijection
\begin{{equation}}
H_-\psi_n=\lambda_n\psi_n,\quad \lambda_n>0
\quad\Longrightarrow\quad
\chi_n=\lambda_n^{-1/2}Q_\nu\psi_n,\quad H_+\chi_n=\lambda_n\chi_n,
\end{{equation}}
with the converse map through $Q_\nu^\dagger$.  Only kernel terms may remain
unpaired:
\begin{{equation}}
{{\rm ind}}(Q_\nu)=\dim\ker Q_\nu-\dim\ker Q_\nu^\dagger .
\end{{equation}}
This is a theorem target, not yet a verified result on the C17 compact domain.

\section{{C19 Zeta-Determinant Cancellation Target}}
If C18 holds on the same compact domain, then the positive-mode determinant
ratio should cancel:
\begin{{equation}}
\log\frac{{\det_\zeta H_-^{{\rm pos}}}}{{\det_\zeta H_+^{{\rm pos}}}}
=-\zeta_{{-,{{\rm pos}}}}'(0)+\zeta_{{+,{{\rm pos}}}}'(0)=0.
\end{{equation}}
Any remaining contribution must be explicit kernel/index/anomaly/boundary or
matching residue.  This is a determinant target, not yet a proof.

\section{{C20 Residue Classification Gate}}
After positive-mode cancellation, allowed residues are only those fixed by the
same domain-native structure: kernel/index, anomaly/Ward, boundary, or frozen
matching terms.  Forbidden residues include endpoint-tuned scalar counterterms,
branch-specific regulators, unmatched boundary constants, or arbitrary finite
terms inserted to restore naturalness.  Thus cancellation is not enough; the
remaining finite residue must also be admissible.

\section{{$\epsilon_\tau$ Origin Gate}}
The last absolute scale is $\epsilon_\tau$.  Admissible origins include:
\begin{{itemize}}
\item a universal parent stiffness/density scale;
\item a quantized compact-cell action unit;
\item a parent-derived wall-tension unit;
\item an independent gravity/cosmology calibration frozen before Higgs/top comparison.
\end{{itemize}}
Forbidden origins include Higgs-vev backsolve, top-mass backsolve,
particle-specific $\epsilon_\tau$, Branch-A-only scale insertion, or an
unmatched Planck-ratio insertion.

\section{{Wall-Cell Embedding Gate}}
The Branch A wall domain must be related to the compact cell by
\begin{{equation}}
{{\cal W}}_A\subseteq{{\cal K}}_\tau
\qquad
\hbox{{or}}
\qquad
{{\cal W}}_A={{\cal K}}_\tau/{{\cal H}}_A.
\end{{equation}}
The induced measure must come by restriction, quotient, or coarea.  The Higgs
and top operators must use the same domain and regulator.  The count $N_\tau$
must be compatible with wall index/residue bookkeeping.  Otherwise the
compact-cell unit and Higgs wall remain disconnected hidden scales.

\section{{v0.2 Single-Package Parent Action}}
The current strongest scaffold is
\begin{{equation}}
S_A^{{(0.2)}}=
\epsilon_\tau
\left[
S_{{\rm cell}}+S_{{\rm wall}}+S_{{\rm gap}}+S_Q+S_{{\rm anom}}
+S_{{\rm prod}}+S_{{\rm reg}}
\right].
\end{{equation}}
It keeps in one package:
\begin{{itemize}}
\item compact tau cell and count sector;
\item induced Branch A wall domain;
\item trace quotient;
\item unique hypercharge wall channel;
\item positive leakage gaps;
\item minimal $P,C,B$ cohomology roles;
\item same-domain product and regulator;
\item universal $\epsilon_\tau$;
\item fixed dimensionless $y_t^{{\rm parent}}$.
\end{{itemize}}
This is the current best single scaffold, but still not the final action.

\section{{Top Determinant Cancellation Candidate}}
The radiative gate requires the physical top-sensitive operator to live on the
same wall-domain Q package.  If positive spectra of $Q_t^\dagger Q_t$ and
$Q_tQ_t^\dagger$ are paired and the regulator is shared, then the signed
positive-spectrum determinant can cancel structurally.  Zero modes and index
residue must remain explicit.  The gate is not solved until the physical
$Q_t$, determinant signs, regulator, and Yukawa strength are parent-derived.

\section{{Toy Demonstrator}}
The toy demonstrator sets
\begin{{equation}}
\epsilon_\tau=\mu_\tau({{\cal K}}_\tau)=N_\tau=C_A=\hat E_{{\rm wall}}
=\hat M_{{\rm stab}}=g_{{\rm vis}}=R_t=1.
\end{{equation}}
Then
\begin{{equation}}
m_{{\rm toy}}=\frac{{y_t^{{\rm parent}}}}{{\sqrt2}}
={toy_mass:.16f}.
\end{{equation}}
This is finite and endpoint-free, but it has no GeV interpretation.

\section{{Audit Map}}
The generated repository checks:
\begin{{itemize}}
\item the gamma-function quartic overlap and sensitivity scan;
\item Wolfram checks for normalization, hypercharge, line quotient, and BRST skeleton;
\item packet CSV values for $I_4(3/10)$, ${I_QHU:.15f}$, $g_t=18/5$, and $y_t^{{\rm parent}}$;
\item TeX/PDF regeneration and arXiv source package creation;
\item explicit claim-boundary strings in the manuscript.
\end{{itemize}}

\section{{Full Gate Ledger}}
\begin{{longtable}}{{p{{0.30\linewidth}}p{{0.58\linewidth}}}}
\toprule
\textbf{{Gate}} & \textbf{{Current status}}\\
\midrule
$3+2$ stabilizer & derived in minimal Branch A carrier\\
$\kappa_\tau^2=3/5$ & canonical hypercharge normalization\\
unoriented line quotient & theorem-candidate with projector audit\\
$\tanh x$ wall & minimal/BPS wall route; parent derivation open\\
$\nu_i=\kappa_\tau^2|Y_i|$ & conditional F1--F8 forcing route\\
quartic overlap & computed: $I_4(3/10)={i4_d:.15f}$\\
Yukawa overlap & computed: ${{\cal I}}_{{qHu}}={I_QHU:.15f}$\\
$g_t$ & bridge-trace candidate $18/5$\\
$y_t^{{\rm parent}}$ & fixed dimensionless coefficient ${Y_T_PARENT:.15f}$\\
$v_{{\rm eff}}$ & stabilizer-gap route, not measured vev derivation\\
$R_t$ & running/matching shape gate\\
$\epsilon_\tau$ & last universal absolute scale blocker\\
wall-cell embedding & same-domain compact-cell requirement\\
top determinant & positive-spectrum cancellation theorem-candidate\\
BRST/anomaly/regulator & skeleton only; full QFT closure open\\
Standard Model emergence & roadmap only\\
\bottomrule
\end{{longtable}}

\section{{What Would Count As A Win}}
The Higgs route would become much stronger if a single final parent action:
\begin{{enumerate}}
\item derives the two protected visible modules;
\item derives F1--F8 rather than assuming them;
\item selects ${{\cal K}}_\tau$ and ${{\cal W}}_A$ with the required embedding;
\item fixes $\epsilon_\tau$ and $C_A$ without Higgs/top endpoint data;
\item closes the BRST/anomaly/regulator package;
\item computes the physical top determinant and matching map.
\end{{enumerate}}

\section{{Conclusion}}
The full derivation ledger shows that the Higgs branch has moved from a single
quartic coincidence to a structured single-package proof program.  The sharp
fixed chain is
\begin{{equation}}
3+2
\rightarrow
\kappa_\tau^2=\frac35
\rightarrow
\nu_H=\frac3{{10}}
\rightarrow
I_4(3/10)
\rightarrow
{{\cal I}}_{{qHu}}
\rightarrow
 y_t^{{\rm parent}}=0.9615522319206335.
\end{{equation}}
The remaining blockers are now also sharp: $\epsilon_\tau$, $C_A$, compact cell,
wall embedding, continuum regulator, top determinant, and matching.  Until
those are derived, this is a disciplined candidate derivation program, not a
completed Higgs-sector proof.

\end{{document}}
"""

def compile_pdf() -> str:
    if shutil.which("tectonic") is None:
        return "blocked_tectonic_not_installed"
    log = SOURCE / "tectonic_build.log"
    result = subprocess.run(
        ["tectonic", "main.tex"],
        cwd=SOURCE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        return "blocked_compile_failed"
    return "ready"


def compile_full_derivation_pdf() -> str:
    if shutil.which("tectonic") is None:
        return "blocked_tectonic_not_installed"
    log = FULL_DERIVATION / "tectonic_build.log"
    result = subprocess.run(
        ["tectonic", "full_derivation.tex"],
        cwd=FULL_DERIVATION,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        return "blocked_compile_failed"
    return "ready"


def run_wolfram_audits() -> None:
    WOLFRAM_LOGS.mkdir(parents=True, exist_ok=True)
    scripts = [
        "Higgs_Quartic_Overlap_Verification.wl",
        "BranchA_Stabilizer_Hypercharge_Audit.wl",
        "G2_Unoriented_Line_Quotient_Audit.wl",
        "Projection_BRST_Skeleton.wl",
    ]
    wolframscript = shutil.which("wolframscript")
    for script in scripts:
        log_path = WOLFRAM_LOGS / script.replace(".wl", ".log")
        script_path = ROOT / "wolfram" / script
        if wolframscript is None:
            log_path.write_text("blocked_wolframscript_not_installed\n", encoding="utf-8")
            continue
        result = subprocess.run(
            [wolframscript, "-file", str(script_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")


def write_zip_entry(zf: ZipFile, path: Path, arcname: str) -> None:
    info = ZipInfo(arcname)
    info.date_time = (2026, 5, 19, 0, 0, 0)
    info.compress_type = ZIP_DEFLATED
    zf.writestr(info, path.read_bytes())


def build_arxiv_zip() -> None:
    if ARXIV_ZIP.exists():
        ARXIV_ZIP.unlink()
    with ZipFile(ARXIV_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        write_zip_entry(zf, SOURCE / "main.tex", "main.tex")
        write_zip_entry(zf, SOURCE / "references.bib", "references.bib")
        for figure in sorted(SOURCE_FIGURES.glob("*.pdf")):
            write_zip_entry(zf, figure, f"figures/{figure.name}")


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    SOURCE_FIGURES.mkdir(parents=True, exist_ok=True)
    FULL_DERIVATION.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir(parents=True, exist_ok=True)
    make_figures()
    write_csv(PACKET / "paper4_higgs_overlap_scan_v01.csv", diagnostic_rows(), ["nu", "I4", "lambda_tau_required_for_lambda_H_0p129", "guardrail"])
    write_csv(
        PACKET / "paper4_quartic_sensitivity_audit_v01.csv",
        sensitivity_rows(),
        [
            "band",
            "nu_min",
            "nu_max",
            "I4_min",
            "I4_max",
            "lambda_tau_required_min",
            "lambda_tau_required_max",
            "interpretation",
            "guardrail",
        ],
    )
    write_csv(PACKET / "paper4_higgs_module_summary_v01.csv", summary_rows(), ["quantity", "value", "interpretation", "guardrail"])
    write_csv(
        PACKET / "paper4_compact_spectrum_pilot_v01.csv",
        compact_spectrum_pilot_rows(),
        ["operator", "mode", "eigenvalue", "nu", "box_half_length", "grid_points", "interpretation", "guardrail"],
    )
    (SOURCE / "main.tex").write_text(manuscript_tex(), encoding="utf-8")
    (SOURCE / "references.bib").write_text(references_bib(), encoding="utf-8")
    (FULL_DERIVATION / "full_derivation.tex").write_text(full_derivation_tex(), encoding="utf-8")
    run_wolfram_audits()
    pdf_status = compile_pdf()
    full_derivation_status = compile_full_derivation_pdf()
    write_csv(PACKET / "paper4_readiness_table_v01.csv", readiness_rows(pdf_status), ["Item", "Status", "Detail", "Guardrail"])
    build_arxiv_zip()
    print(SOURCE / "main.tex")
    print(SOURCE / "main.pdf")
    print(FULL_DERIVATION / "full_derivation.tex")
    print(FULL_DERIVATION / "full_derivation.pdf")
    print(ARXIV_ZIP)
    print(pdf_status)
    print(f"full_derivation_{full_derivation_status}")


if __name__ == "__main__":
    main()
