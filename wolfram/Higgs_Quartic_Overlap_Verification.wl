(* ::Package:: *)
(* Higgs quartic overlap verification for Paper 4. *)
(* Scope: symbolic/numeric audit only; not physical validation by itself. *)

ClearAll["Global`*"];

nuD = 3/10;
lambdaH = 0.129;

normSq[nu_] := Gamma[nu + 1/2]/(Sqrt[Pi] Gamma[nu]);
h[nu_, x_] := Sqrt[normSq[nu]] Sech[x]^nu;
i4Formula[nu_] := Gamma[nu + 1/2]^2 Gamma[2 nu]/(Sqrt[Pi] Gamma[nu]^2 Gamma[2 nu + 1/2]);

normalizationCheck = FullSimplify[
  Integrate[h[nu, x]^2, {x, -Infinity, Infinity}],
  Assumptions -> nu > 0
];

quarticIntegralCheck = FullSimplify[
  Integrate[h[nu, x]^4, {x, -Infinity, Infinity}] == i4Formula[nu],
  Assumptions -> nu > 0
];

i4NuD = N[i4Formula[nuD], 20];
lambdaTauRequired = N[lambdaH/i4Formula[nuD], 20];

sensitivityRows = Table[
  With[{i4 = N[i4Formula[n], 20]},
    {n, i4, N[lambdaH/i4, 20]}
  ],
  {n, 0.10, 0.80, 0.01}
];

moderateWindow = Select[sensitivityRows, 0.25 <= #[[1]] <= 0.35 &];
moderateLambdaRange = {Min[moderateWindow[[All, 3]]], Max[moderateWindow[[All, 3]]]};

Print["normalizationCheck = ", normalizationCheck];
Print["quarticIntegralCheck = ", quarticIntegralCheck];
Print["I4(3/10) = ", i4NuD];
Print["lambdaTauRequired(lambdaH=0.129) = ", lambdaTauRequired];
Print["moderate lambdaTau range for 0.25 <= nu <= 0.35 = ", moderateLambdaRange];
Print["Guardrail: the overlap coincidence is not evidence by itself."];
