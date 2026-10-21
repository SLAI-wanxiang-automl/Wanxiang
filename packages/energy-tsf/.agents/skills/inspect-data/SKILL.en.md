---
name: inspect-data
description: Inspect time-series data, splits, missingness, available Providers, and a deterministic trial plan without training or publishing.
---

# Inspect data

Use this skill before training when the user asks whether data is suitable or what a modeling run would do. Load the modeling instructions as well when they intend to train.

Run the applicable `prepare-*` command, then `inspect-modeling` with the resolved configuration. Explain row count, original missing and imputed values, 70/20/10 boundaries, daily horizon steps, valid origins per split, planned trial count, and unavailable Providers with reasons.

Inspect is read-only: do not train, create a Candidate, or update Active. Stop if validation or test lacks a complete one-day window and explain the data-length limitation.
