(* ::Package:: *)
(* Branch A stabilizer and hypercharge normalization audit for Paper 4. *)
(* Scope: algebraic normalization check, not derivation of the nu-rule. *)

ClearAll["Global`*"];

tSigma = DiagonalMatrix[{2, 2, 2, -3, -3}]/Sqrt[60];
y = DiagonalMatrix[{-1/3, -1/3, -1/3, 1/2, 1/2}];
tY = Sqrt[3/5] y;

traceTSigma = Tr[tSigma];
normTSigma = FullSimplify[Tr[tSigma.tSigma]];
normTY = FullSimplify[Tr[tY.tY]];
relationCheck = FullSimplify[tSigma == -tY];

blockEigenvalues = Tally[Diagonal[tSigma]];
minimalIntegerRatio = {2, -3};
tracelessCheck = 3 minimalIntegerRatio[[1]] + 2 minimalIntegerRatio[[2]];

nuRuleStatus = "conditional_forcing_route: nu_i = 3 |Y_i| / 5 remains theorem-candidate until parent-action derivation";

Print["Tr[T_Sigma] = ", traceTSigma];
Print["Tr[T_Sigma^2] = ", normTSigma];
Print["Tr[T_Y^2] = ", normTY];
Print["T_Sigma == -T_Y = ", relationCheck];
Print["T_Sigma eigenvalue multiplicities = ", blockEigenvalues];
Print["3*(2) + 2*(-3) = ", tracelessCheck];
Print[nuRuleStatus];
