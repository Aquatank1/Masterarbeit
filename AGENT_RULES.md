# Rules for the AI coding agents

Several AI coding agents worked on this project during development: Claude Code (Anthropic), Codex (OpenAI),
Antigravity (Google) and OpenCode. Each had its own folder in the development project and worked on the same
shared data. In the author's opinion Antigravity performed poorly, which is why it was dropped early. Astra, a
version of Codex meant to be on par with Anthropic's Fable model, was left to work with little input to see
whether it would find new ideas; it did not.

This repository contains only the final code of the Claude folder (`Claude/`) and the author's own
acquisition and preprocessing pipeline in the root. The author directed the work, set the rules below and
made every decision. The rules evolved during the project; this file describes their final state.

## Workspace

- Agents may read the whole project, but write only inside their own folder. Root-level files and shared data
  folders are maintained by the author; writing anywhere else requires an explicit instruction.
- The raw data folders (`Data_*`) are read only. They are never changed, moved or deleted, because they are
  expensive to regenerate.
- A cache of trained models or results is never overwritten or deleted. A changed computation gets a new cache folder.
- Notebooks verified by the author (`00_preprocessing.ipynb`, `01_headline_register.ipynb`) are not changed
  without explicit permission.
- Long runs, such as retraining many networks, are started only when the author says so.
- The `AgriFM` conda environment belongs to a different project and is not used.

## Modelling and evaluation

- Every model is evaluated with RMSE, MAE and R². The labels are in dt/ha; the thesis reports t/ha.
- The split is chronological: training on 2016–2022, validation on 2023 (early stopping and every choice),
  test on 2025. At the start of the project the default was a random 70/15/15 split grouped by county; it was
  replaced because predicting an unseen season is the realistic task. Leave-one-year-out is an additional
  evaluation in which everything fitted from data is refitted in every fold.
- Nothing is chosen on the test season. Seeds or models are never picked by their test result; a combination
  chosen on the test season is reported only as an upper bound, not as a model. In practice the 2025 results
  were looked at repeatedly during development, which the thesis discusses as a limitation.
- The historical yield is a leak-free five-year moving average of the seasons strictly before the predicted
  one, never an average over all years.
- Every neural network configuration is evaluated as one ensemble of 30 models (60 only in the ensemble-size
  experiment), without extra hidden models. Differences below 0.01 t/ha RMSE count as training noise.
- In-season forecasts (one forecast per window) and after-season predictions are kept as separate tracks.
- Numbers are always taken from executed outputs and cached results, never from memory.

## Thesis text

Agents drafted sections, reviewed chapters and corrected language on request. Chapter text written by the
author was not changed without being asked. The author rewrote the drafts and decided the final text, as
stated in the declaration of the thesis.
