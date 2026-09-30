"""Oracle ensemble: the subset of headline models (and its size) with the lowest RMSE or MAE on the 2025 test set.

Not a valid model — the models are chosen on the test set itself. It shows the upper bound of what picking
models could ever give. For comparison the same selection is run on the 2023 validation season (valid)
and then applied to 2025.

Pools (identical configuration, only the seed differs): E7's 60 models of 08, and those plus the 30 models of
01 (01 stores no validation predictions, so the valid selection uses E7 only). Only cached predictions are read.

    python Claude/final_results_claude/oracle_ensemble.py             # optimize the RMSE
    python Claude/final_results_claude/oracle_ensemble.py --mae       # optimize the MAE
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

FINAL_RESULTS = os.path.dirname(os.path.abspath(__file__))
os.chdir(FINAL_RESULTS)
notebook = json.load(open("08_experiments.ipynb", encoding="utf-8"))
namespace = {"__name__": "__main__", "display": lambda obj: None}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "setup" in cell["metadata"].get("tags", []):
        exec(compile("".join(cell["source"]), "<setup>", "exec"), namespace)
np, pd, CACHE = namespace["np"], namespace["pd"], namespace["CACHE"]
rmse, mae, cohort_metrics = namespace["rmse"], namespace["mae"], namespace["cohort_metrics"]
OPTIMIZED = "MAE" if "--mae" in sys.argv else "RMSE"
loss = mae if OPTIMIZED == "MAE" else rmse
is_headline_cohort = namespace["is_headline_cohort"]
yield_actual_test, yield_actual_validation = namespace["yield_actual_test"], namespace["yield_actual_validation"]
DT = namespace["DT_HA_PER_T_HA"]

e07_folder = os.path.join(CACHE, f"e07_transformer_bank_{namespace['HEADLINE_WIDTH_TAG']}")
e07 = [np.load(os.path.join(e07_folder, f"model_{i:02d}.npz")) for i in range(60)]
e07_test = np.array([model["test"] for model in e07])
e07_validation = np.array([model["validation"] for model in e07])
headline_01_test = np.asarray(namespace["headline_bank"])
pool_90_test = np.concatenate([e07_test, headline_01_test])
pool_90_names = [f"E7 model {i}" for i in range(60)] + [f"01 model {i}" for i in range(len(headline_01_test))]


def select(predictions, actual):
    """Greedy forward selection without replacement, then one-for-one swaps until nothing improves.
    Returns (chosen indices, loss path of the greedy pass by size)."""
    n_models = len(predictions)
    chosen, remaining, path = [], list(range(n_models)), []
    running_sum = np.zeros(predictions.shape[1])
    for size in range(1, n_models + 1):
        scores = [loss((running_sum + predictions[m]) / size, actual) for m in remaining]
        best = remaining[int(np.argmin(scores))]
        chosen.append(best)
        remaining.remove(best)
        running_sum += predictions[best]
        path.append(min(scores))
    best_size = int(np.argmin(path)) + 1
    subset = chosen[:best_size]
    improved = True
    while improved:                      # swap one model in the subset for one outside it
        improved = False
        current = loss(predictions[subset].mean(axis=0), actual)
        outside = [m for m in range(n_models) if m not in subset]
        for position in range(len(subset)):
            for candidate in outside:
                trial = subset.copy()
                trial[position] = candidate
                score = loss(predictions[trial].mean(axis=0), actual)
                if score < current - 1e-9:
                    subset, current, improved = trial, score, True
                    break
            if improved:
                break
    return subset, path


def row(name, prediction, size):
    metrics = cohort_metrics(prediction)
    return {"ensemble": name, "models": size, "10d RMSE (t/ha)": metrics["headline"]["RMSE"],
            "10d MAE (t/ha)": metrics["headline"]["MAE"], "10d R2": metrics["headline"]["R2"],
            "20d RMSE (t/ha)": metrics["full"]["RMSE"]}


rows = [row("E7 models 0-29 (the reference ensemble)", e07_test[:30].mean(axis=0), 30),
        row("all 60 E7 models", e07_test.mean(axis=0), 60),
        row("01 headline, 30 models", headline_01_test.mean(axis=0), 30)]
single_rmse = [loss(p[is_headline_cohort], yield_actual_test[is_headline_cohort]) for p in pool_90_test]
best_single = int(np.argmin(single_rmse))
rows.append(row(f"best single model on 2025 ({pool_90_names[best_single]})", pool_90_test[best_single], 1))

actual_10d = yield_actual_test[is_headline_cohort]
oracle_60, path_60 = select(e07_test[:, is_headline_cohort], actual_10d)
rows.append(row("ORACLE: chosen on 2025 from the 60 E7 models", e07_test[oracle_60].mean(axis=0), len(oracle_60)))
oracle_90, path_90 = select(pool_90_test[:, is_headline_cohort], actual_10d)
rows.append(row("ORACLE: chosen on 2025 from all 90 models", pool_90_test[oracle_90].mean(axis=0), len(oracle_90)))
valid_60, _ = select(e07_validation, yield_actual_validation)
rows.append(row("valid: chosen on 2023 from the 60 E7 models, applied to 2025", e07_test[valid_60].mean(axis=0), len(valid_60)))

table = pd.DataFrame(rows)
output_folder = os.path.join(CACHE, "oracle_ensemble")
os.makedirs(output_folder, exist_ok=True)
suffix = "" if OPTIMIZED == "RMSE" else "_mae"
table.to_csv(os.path.join(output_folder, f"results{suffix}.csv"), index=False)
pd.DataFrame({"size": range(1, len(path_90) + 1), f"greedy {OPTIMIZED} on 2025, pool of 90 (t/ha)": np.array(path_90) / DT}).to_csv(
    os.path.join(output_folder, f"greedy_path_pool_90{suffix}.csv"), index=False)
print(f"optimized: {OPTIMIZED}")
with pd.option_context("display.width", 200, "display.max_colwidth", 70):
    print(table.round(4).to_string(index=False))
print("\noracle from 90:", sorted(pool_90_names[i] for i in oracle_90))
print("valid selection on 2023:", sorted(f"E7 model {i}" for i in valid_60))
