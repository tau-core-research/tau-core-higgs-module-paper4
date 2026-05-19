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
        PAPER / "figures/paper4_quartic_overlap_curve.pdf",
        ROOT / "figures/paper4_higgs_zero_mode_profiles.svg",
        ROOT / "figures/paper4_quartic_overlap_curve.svg",
        ROOT / "arxiv_submission_source.zip",
        PACKET / "paper4_higgs_overlap_scan_v01.csv",
        PACKET / "paper4_higgs_module_summary_v01.csv",
        PACKET / "paper4_readiness_table_v01.csv",
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
    assert "not a proof of Tau Core" in tex
    assert "not an empirical claim" in tex
    assert "I_4(3/10)" in tex
    assert "\\includegraphics" in tex


def test_arxiv_zip_is_source_only():
    with zipfile.ZipFile(ROOT / "arxiv_submission_source.zip") as archive:
        names = set(archive.namelist())
    assert "main.tex" in names
    assert "references.bib" in names
    assert "main.pdf" not in names
    assert "figures/paper4_higgs_zero_mode_profiles.pdf" in names
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
