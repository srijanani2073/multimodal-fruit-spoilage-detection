# Changelog

## v1.0.0 — 2026-10-26

Release accompanying the M.Tech thesis (Phase I).

- `src/trimodal/`: one package for the data pipeline, the model and all six ablation variants,
  training, ensemble evaluation, the robustness sweep and the statistics. Loads checkpoints saved
  by the notebooks unchanged.
- `scripts/`: command-line training, evaluation, robustness and comparison, plus
  `reproduce_all.sh`; the robustness sweep is now seeded deterministically.
- `tests/`: smoke tests on a generated stand-in for TR-6, so the code can be checked without the
  dataset.
- The master manifest and split record are back in `Code/Manifest/manifests/`.
- README with the final results, `INSTALL.md`, guides in `docs/guide/`, and the project website in
  `docs/`.
- Trimmed `requirements.txt` to the packages the code imports; notebook extras in
  `requirements-notebooks.txt`.
- MIT licence and `CITATION.cff`.

## Research history (before v1.0)

Notebooks 01–14 were developed between August and October 2026; the commit history up to this
release records that work: the data pipeline and manifest, encoder checks, the fusion model, the
ablation and significance tests, the backbone benchmark, the missing-modality sweep, leave-one-fruit-out,
explainability, uncertainty, the regression diagnosis and the capacity sweep.
