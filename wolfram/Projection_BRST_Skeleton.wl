(* ::Package:: *)
(* Projection-BRST skeleton audit for Paper 4. *)
(* Scope: symbolic skeleton only; not anomaly, domain, or regulator proof. *)

ClearAll["Global`*"];

(* BRST skeleton:
   s[H] = Q^\[Dagger] c, s[c] = 0.
   Therefore s^2[H] = Q^\[Dagger] s[c] = 0 if Q^\[Dagger] is BRST-inert.
   We keep this as a symbolic skeleton instead of a noncommutative algebra
   implementation because this notebook is not the anomaly/regulator proof. *)
s2H = 0;

nuD = 3/10;
q[h_] := D[h[x], x] + nuD Tanh[x] h[x];
hD[x_] := Sech[x]^nuD;
zeroModeCheck = FullSimplify[D[hD[x], x] + nuD Tanh[x] hD[x] == 0, Assumptions -> Element[x, Reals]];

operatorFactorization = "O0 = Q_D^\[Dagger] Q_D";
guardrail = "This notebook does not prove anomaly freedom, regulator safety, or radiative stability.";

Print["s^2 H skeleton = ", s2H];
Print["Q_D h_D = 0 check = ", zeroModeCheck];
Print[operatorFactorization];
Print[guardrail];
