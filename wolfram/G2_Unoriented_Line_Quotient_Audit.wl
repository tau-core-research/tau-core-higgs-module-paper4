(* ::Package:: *)
(* Audit for G2 unoriented-line quotient proof. *)

ClearAll["Global`*"];

Y = DiagonalMatrix[{-1/3, -1/3, -1/3, 1/2, 1/2}];
TY = Sqrt[3/5] Y;
inner[A_, B_] := Tr[A.B];

norm = FullSimplify[inner[TY, TY]];

(* Rank-one projector onto the line span(TY). *)
projector[A_] := FullSimplify[KroneckerProduct[Flatten[A], Flatten[A]]/inner[A, A]];
projectorSignInvariant = FullSimplify[projector[TY] == projector[-TY]];

Sigma[f_] := f TY;
traceInvariant = FullSimplify[Tr[Sigma[f].Sigma[f]]];
traceSignInvariant = FullSimplify[
  Tr[Sigma[f].Sigma[f]] == Tr[Sigma[-f].Sigma[-f]]
];

kineticSignInvariant = FullSimplify[
  Tr[D[Sigma[f[x]], x].D[Sigma[f[x]], x]] ==
    Tr[D[Sigma[-f[x]], x].D[Sigma[-f[x]], x]]
];

magnitudeVacua = Solve[2 traceInvariant == 1, f];

Print["Tr[T_Y^2] = ", norm];
Print["Line projector P_TY equals P_-TY = ", projectorSignInvariant];
Print["Tr[(f T_Y)^2] = ", traceInvariant];
Print["Trace invariant under f -> -f = ", traceSignInvariant];
Print["Kinetic invariant under f -> -f = ", kineticSignInvariant];
Print["Unit line magnitude gives f vacua = ", magnitudeVacua];
Print["Guardrail: proves quotient consequences if physical datum is the unoriented line/projector."];
