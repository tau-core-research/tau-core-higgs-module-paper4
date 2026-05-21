(* ::Package:: *)
(* Compact gate-ledger audit for Paper 4. *)
(* Scope: full-derivation ledger consistency; not a proof of C0-C20. *)

ClearAll["Global`*"];

gates = {
  {"C0", "finite-fiber wall spectral cell", "candidate_not_parent_action"},
  {"C1", "two-cluster plus Bogomolny route", "theorem_candidate"},
  {"C2", "projector determinant scale route", "theorem_candidate"},
  {"C3", "explicit compact spectral geometry", "explicit_candidate"},
  {"C4", "six-penalty selection functional", "formulated_not_minimized"},
  {"C5", "local Pöschl-Teller spectrum", "local_spectrum_only"},
  {"C6", "compact boundary-value problem", "formulated_not_solved"},
  {"C7", "competitor audit", "not_exhaustive"},
  {"C8", "minimality theorem candidate", "conditional_not_proven"},
  {"C9", "compact spectrum pilot", "numerical_pilot_not_theorem"},
  {"C10", "uniqueness proof split", "not_closed"},
  {"C11", "algebraic minimality lemmas", "formulated_not_fully_proven"},
  {"C12", "protected-form selection gate", "selection_gate"},
  {"C13", "cohomology/index selection route", "candidate_route_not_closed"},
  {"C14", "toy complex sanity check", "toy_model_not_parent_derivation"},
  {"C15", "operator-domain ansatz", "operator_target"},
  {"C16", "shared-domain compatibility gate", "compatibility_gate"},
  {"C17", "self-adjoint domain candidate", "candidate_not_classification"},
  {"C18", "paired-spectrum verification target", "theorem_target_not_verified"},
  {"C19", "zeta-determinant cancellation target", "determinant_target_not_proven"},
  {"C20", "residue classification gate", "classification_gate_formulated"}
};

subGates = {
  "C16.1 Anomaly-Compatible Domain Closure Criterion",
  "C16.2 Same-Domain Ward Identity Gate",
  "C16.3 UV/Continuum Admissibility Criterion",
  "C16.3a Microscopic Continuum Completion Target",
  "C16.4 Representation Trace-Cancellation Gate",
  "C17.1 Parent-Domain Selection Functional",
  "C20.1 Regulator-Independence Residue Test",
  "Top Determinant Finite-Residue Extraction Gate",
  "Numerical Endpoint Matching Protocol"
};

gateCount = Length[gates];
gateIds = gates[[All, 1]];
expectedIds = "C" <> ToString[#] & /@ Range[0, 20];
idsContinuous = gateIds == expectedIds;

proofStatuses = Select[gates[[All, 3]], StringContainsQ[#, "proven"] || StringContainsQ[#, "proof"] &];
forbiddenSolvedClaim = MemberQ[gates[[All, 3]], "solved" | "validated" | "proven"];

scriptDir = DirectoryName[$InputFileName];
repoRoot = ParentDirectory[scriptDir];
fullDerivationPath = FileNameJoin[{repoRoot, "paper4_full_derivation", "full_derivation.tex"}];
mainPath = FileNameJoin[{repoRoot, "paper4_submission_source", "main.tex"}];

fullText = If[FileExistsQ[fullDerivationPath], Import[fullDerivationPath, "Text"], ""];
mainText = If[FileExistsQ[mainPath], Import[mainPath, "Text"], ""];

subGateCoverage = AssociationThread[subGates, StringContainsQ[fullText, #] & /@ subGates];
allSubGatesPresent = And @@ Values[subGateCoverage];

claimBoundaryChecks = <|
  "main_has_final_mechanism_target_sentence" ->
    StringContainsQ[mainText, "The present result is a mechanism"] &&
      StringContainsQ[mainText, "target and spectral proof program"] &&
      StringContainsQ[mainText, "not a completed Higgs-sector derivation"],
  "main_uses_operator_domain_assumption_language" ->
    StringContainsQ[mainText, "established only within" ~~ ___ ~~ "stated operator-domain assumptions"],
  "full_derivation_uses_proposition_language" ->
    StringContainsQ[fullText, "establishes the positive-spectrum determinant" ~~ ___ ~~ "proposition inside the same-domain setup"],
  "figure4_toy_not_physical_determinant" ->
    StringContainsQ[fullText, "not used as independent evidence for radiative stability"],
  "finite_residue_not_computed" ->
    StringContainsQ[fullText, "no computed" ~~ ___ ~~ "physical top finite residue"],
  "endpoint_protocol_freezes_before_compare" ->
    StringContainsQ[fullText, "derive, freeze, run, compare"]
|>;
allClaimBoundaryChecks = And @@ Values[claimBoundaryChecks];

forbiddenStrongPhrases = {
  "This proves the conditional compact self-adjoint spectral theorem",
  "This proves the conditional positive-spectrum determinant cancellation lemma",
  "is a completed Higgs-sector derivation"
};
forbiddenStrongPhraseHits = Select[forbiddenStrongPhrases, StringContainsQ[mainText <> "\n" <> fullText, #] &];
noForbiddenStrongPhrases = forbiddenStrongPhraseHits == {};

Print["compactGateLedgerCount = ", gateCount];
Print["compactGateLedgerIdsContinuous = ", idsContinuous];
Print["compactGateLedgerFirst = ", First[gates]];
Print["compactGateLedgerLast = ", Last[gates]];
Print["compactGateLedgerNoSolvedClaim = ", Not[forbiddenSolvedClaim]];
Print["compactGateLedgerProofStatusWords = ", proofStatuses];
Print["compactGateLedgerSubGateCoverage = ", subGateCoverage];
Print["compactGateLedgerAllSubGatesPresent = ", allSubGatesPresent];
Print["compactGateLedgerClaimBoundaryChecks = ", claimBoundaryChecks];
Print["compactGateLedgerAllClaimBoundaryChecks = ", allClaimBoundaryChecks];
Print["compactGateLedgerForbiddenStrongPhraseHits = ", forbiddenStrongPhraseHits];
Print["compactGateLedgerNoForbiddenStrongPhrases = ", noForbiddenStrongPhrases];
Print["Guardrail: C0-C20 is a compact proof-gate ledger, not a completed Higgs proof."];
