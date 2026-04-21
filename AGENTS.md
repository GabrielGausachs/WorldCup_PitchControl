# AGENTS.md

## Project purpose
Replicate the pitch value model from Fernandez & Bornn (2018) using PFF World Cup 2022 open tracking/event data, then compute space value = pitch control × pitch value.

## Source of truth
- Paper PDF: <paper\fernandez_ssac_2018.pdf>
- The paper is the source of truth for the pitch value model.
- Main.ipynb may contain useful prior exploratory code, but it should be used only for inspiration unless explicitly instructed otherwise.
- Do not depend on notebook code for the production pipeline.

## Important paths
- Raw tracking/event data is NOT stored in the repository.
- Data must be accessed via DATA_ROOT.
- DATA_ROOT is resolved as:
  1. environment variable DATA_ROOT (preferred)
  2. fallback default path defined in config.py (for local development only)
- Processed datasets should be written to a dedicated subdirectory derived from DATA_ROOT, e.g. DATA_ROOT/processed_pitch_value/, unless instructed otherwise.

## Key constraints
- Use databallpy’s pitch control implementation via the library. Do not reimplement pitch control.
- Correctly handle player positions, ball position, and team identity (home/away).
- Ensure attacking-direction normalization is correct and documented.
- Keep raw data immutable.
- Prefer modular scripts over notebooks for reproducibility.
- Log assumptions explicitly.

## Workflow rules
1. Inspect the repository structure and identify current data-loading codepaths before editing.
2. Inspect the paper sections relevant to pitch value before implementation.
3. Propose a plan before making changes.
4. Prefer minimal, localized changes over broad refactors.
5. Before full extraction, validate on a small subset of matches (for example 1–3 matches or another clearly stated sample).
6. Save intermediate datasets in a storage-efficient format with documented schema.

## Validation requirements
- Run small-sample sanity checks before full dataset generation
- Save a short validation report with assumptions and checks

## Output requirements for the dataset stage
- Reproducible dataset-building pipeline
- Saved dataset artifact
- Metadata/schema file
- Short report describing assumptions, filters, and validation checks

## Training gate
Do not train any model until you explain:
1. dataset schema,
2. preprocessing decisions,
3. storage format choice,
4. model design,
5. validation plan,
6. expected compute cost,
and ask for permission.

## Environment setup
- The project uses a conda environment named: counterpress

- Activate environment:
  - conda activate counterpress

- Reproducible setup (if environment needs to be created):
  - conda create -n counterpress python=3.10
  - conda activate counterpress
  - pip install -r requirements.txt

## Dependency rules
- If additional packages are needed, explain why before adding them.
- Update requirements.txt when dependencies are added.