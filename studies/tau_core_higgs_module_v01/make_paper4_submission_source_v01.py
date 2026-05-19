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
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper4_submission_source"
SOURCE_FIGURES = SOURCE / "figures"
PUBLIC_FIGURES = ROOT / "figures"
PACKET = ROOT / "studies/tau_core_higgs_module_v01/packet_v01_seed"
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
            "Status": "main_blocker",
            "Detail": "The rule nu_i=3|Y_i|/5 must be derived from a stabilizer, variational, index, or anomaly argument before the module can be promoted.",
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
            "Status": "skeleton",
            "Detail": "Top/flavor deformation is a roadmap calculation, not a completed determinant proof.",
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
\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}}
\title{{A Branch A cohomological Higgs module: stabilizer-derived quartic overlap and lightness-protection gates}}
\author{{Jozsef Olcsak}}
\date{{May 19, 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
We describe a constrained Branch A projection/cohomology Higgs module in which the observed four-dimensional Higgs is treated as the visible projection of a tau-profiled parent field. The central reproducible calculation links a minimal $3+2$ stabilizer, the canonical hypercharge direction, a Higgs localization exponent $\nu_D=3/10$, and a zero-mode quartic overlap $I_4(3/10)={i4_d:.6f}$. This is a concrete conditional mathematical result: under the stated Branch A rule, the observed Higgs quartic corresponds to an order-one parent quartic rather than to an extreme hierarchy. The construction is a candidate mechanism with explicit validation gates. It is not a completed Standard Model derivation, not a proof of any parent projection theory, and not an empirical claim.
\end{{abstract}}

\section{{Scope and Claim Boundary}}
This manuscript isolates one theoretical module: a possible origin for a Higgs-scale quartic overlap and a possible cohomological lightness-protection route. The paper does not claim that the Standard Model, gravity, or a complete parent projection theory has been derived. The result should be read as a reproducible candidate mechanism with explicit proof obligations.

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

\section{{Theorem Status Summary}}
\begin{{center}}
\begin{{tabular}}{{p{{0.46\linewidth}}p{{0.42\linewidth}}}}
\toprule
\textbf{{Claim}} & \textbf{{Status in this manuscript}}\\
\midrule
$3+2$ stabilizer & derived within the minimal Branch A carrier\\
$\kappa_\tau^2=3/5$ normalization & derived from canonical hypercharge normalization\\
unique Abelian direction & derived within the traceless $3+2$ Branch A centralizer\\
unoriented $T_Y$ line quotient & theorem-candidate, with projector audit\\
$\tanh x$ wall & candidate minimal/BPS wall route\\
$\nu_i=\kappa_\tau^2|Y_i|$ & theorem-candidate, not yet parent-derived\\
quartic overlap $I_4(3/10)$ & computed and independently audited\\
projection-BRST protection & roadmap gate\\
top determinant / radiative stability & open gate\\
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.96\linewidth]{{figures/paper4_mechanism_flow.pdf}}
\caption{{Mechanism route and proof-gate structure. The first steps are controlled algebraic reductions; the localization rule and physical interpretation remain conditional on the parent-action gates.}}
\label{{fig:mechanism-flow}}
\end{{figure}}

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
This section is a roadmap calculation: the top determinant must still be completed before the deformation can be promoted to a derived prediction. In particular, the manuscript does not yet prove radiative stability or solve the hierarchy problem. The top/flavor determinant is the hardest remaining gate: it must show that mismatch-independent and linear mass terms are absent or quotient-trivial.

\section{{What Is Reproduced And What Is Not}}
The current package reproduces:
\begin{{itemize}}
\item a minimal $3+2$ stabilizer-compatible hypercharge direction;
\item the Branch A working exponent $\nu_D=3/10$ once the $\nu_i=3|Y_i|/5$ rule is assumed;
\item the normalized $\operatorname{{sech}}^{{3/10}}$ zero mode;
\item the quartic overlap $I_4(3/10)={i4_d:.6f}$;
\item an order-one parent-quartic requirement and a TeV-scale deformation estimate.
\end{{itemize}}
It does not reproduce:
\begin{{itemize}}
\item the full Standard Model;
\item the Yukawa hierarchy;
\item anomaly cancellation from the parent construction;
\item radiative stability;
\item the measured Higgs mass from a completed top determinant;
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

\section{{Remaining Proof Gates}}
The present manuscript isolates three unresolved gates. These are not
presentation details; they are the conditions that separate the current
mechanism target from a Higgs-sector derivation.

\begin{{enumerate}}
\item \textbf{{Localization derivation.}} The factor $3/5$ is now traced to
the Branch A hypercharge normalization, but the localization postulate
$\nu_i=\kappa_\tau^2|Y_i|$ still has to be derived. A successful derivation
would need to show that the Branch A regular background is the unoriented
hypercharge line/projector, that non-hypercharge leakage modes are
Hessian-positive or quotient-null, and that the reduced wall functional is the
minimal BPS two-vacuum functional in the canonical tau measure.
\item \textbf{{BRST/anomaly/regulator consistency.}} The projection-BRST
skeleton explains how a quotient could protect the visible zero mode, but the
present packet checks only the algebraic skeleton. A proof must still define
the Hilbert-space domain, nilpotent charge, regulator, anomaly constraints, and
Ward identities.
\item \textbf{{Top determinant and radiative stability.}} The top/flavor
deformation is still a roadmap calculation. Until the determinant is computed
and mismatch-independent or linear mass terms are shown to be absent or
quotient-trivial, the manuscript does not solve the Higgs hierarchy problem.
\end{{enumerate}}

These gates make the status of the paper precise. The quartic-overlap result is
a reproducible and nontrivial mechanism target, but the localization,
cohomological consistency, and radiative-stability gates must all close before
the module can be promoted to a derivation.

\section{{Near-Term Falsifiable Prediction}}
The current module predicts a narrow kind of future test rather than an already validated signal. If the top/flavor deformation gate is completed with a natural spurion $\delta_\star\sim10^{-3}$, the same equations imply a parent scale $M_\tau$ in the multi-TeV range and a Higgs-sector spectral threshold
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
\item the Branch A projection metric rule $\nu_i=3|Y_i|/5$ cannot be derived;
\item the parent quartic is not canonical or requires large tuning;
\item the projection-BRST quotient is anomalous or regulator-dependent;
\item a visible Higgs mass is unavoidable at $\delta_\star=0$;
\item the top determinant produces a large mismatch-independent or linear mass term;
\item the TeV-sector spectral window is excluded by Higgs coupling or direct-search constraints.
\end{{itemize}}

\section{{Conclusion}}
The Branch A Higgs module links a $3+2$ stabilizer, hypercharge normalization, a $\operatorname{{sech}}^{{3/10}}$ visible zero mode, and a quartic overlap requiring an order-one parent quartic. The paper establishes a compact, reproducible mechanism target: if the Branch A localization rule is derived, the Higgs quartic is mapped to a natural parent-scale coupling rather than an extreme hierarchy. It does not establish the full parent theory or solve the Higgs hierarchy problem. The next paper-grade step is to derive the $\nu_i=3|Y_i|/5$ rule, complete the projection-BRST and top-determinant gates, and compare the implied heavy-sector window against collider constraints.

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
    (SOURCE / "main.tex").write_text(manuscript_tex(), encoding="utf-8")
    (SOURCE / "references.bib").write_text(references_bib(), encoding="utf-8")
    run_wolfram_audits()
    pdf_status = compile_pdf()
    write_csv(PACKET / "paper4_readiness_table_v01.csv", readiness_rows(pdf_status), ["Item", "Status", "Detail", "Guardrail"])
    build_arxiv_zip()
    print(SOURCE / "main.tex")
    print(SOURCE / "main.pdf")
    print(ARXIV_ZIP)
    print(pdf_status)


if __name__ == "__main__":
    main()
