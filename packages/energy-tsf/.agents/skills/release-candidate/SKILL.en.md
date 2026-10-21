---
name: release-candidate
description: Explain Candidates, release gate results, and Active publication state without bypassing deterministic release code.
---

# Release candidate

`run-modeling` with `publish=true` saves an immutable Solution, creates a Candidate, reloads the artifact, runs the smoke gate, and atomically updates Active only if approved. Never edit the Active pointer directly or treat a favorable metric as gate approval.

Report the Solution ID, Candidate path, gate decision and rejection reasons, Active Solution ID, and previous version available for rollback. If publication fails, say that Active did not change; retain Candidate and reports for diagnosis. With `publish=false`, report a Candidate only and do not claim it was published.
