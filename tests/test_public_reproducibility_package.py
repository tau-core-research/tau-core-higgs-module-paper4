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
    assert "Localization derivation" in tex
    assert "BRST/anomaly/regulator consistency" in tex
    assert "Top determinant and radiative stability" in tex
    assert "Near-Term Falsifiable Prediction" in tex
    assert "I_4(3/10)" in tex
    assert "\\includegraphics" in tex


def test_numerology_and_nu_rule_gates_are_explicit():
    sensitivity = read_csv(PACKET / "paper4_quartic_sensitivity_audit_v01.csv")
    readiness = {row["Item"]: row for row in read_csv(PACKET / "paper4_readiness_table_v01.csv")}
    summary = {row["quantity"]: row for row in read_csv(PACKET / "paper4_higgs_module_summary_v01.csv")}
    assert any(row["band"] == "moderate" for row in sensitivity)
    assert readiness["nu_rule"]["Status"] == "main_blocker"
    assert readiness["branch_a_parent_selection"]["Status"] == "conditional_gate_chain"
    assert summary["parent_selection_gate_count"]["value"] == "5"
    assert "not evidence by itself" in sensitivity[0]["interpretation"]


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
