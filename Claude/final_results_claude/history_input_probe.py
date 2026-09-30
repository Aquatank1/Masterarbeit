"""What does the trained headline model do with its historical yield input?

Retrains E7 models 0-9 of 08_experiments.ipynb with the notebook's own setup and training code (same seeds,
same protocol) and this time keeps their weights in cache/e07_weights_width256/. Then, on the 2025 test
season, the history INPUT of the network is replaced by 0 dt/ha, -1 dt/ha and the training mean, while the
real history is still added back to the network's output (the target is the deviation from the history).

    python Claude/final_results_claude/history_input_probe.py
"""
import ctypes
import json
import os
import time

import matplotlib
matplotlib.use("Agg")

FINAL_RESULTS = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(FINAL_RESULTS, "08_experiments.ipynb")
N_MODELS = 10

# keep Windows awake while training (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

# 1. the notebook's setup cells, with the training function also returning the best weights
os.chdir(FINAL_RESULTS)
notebook = json.load(open(NOTEBOOK, encoding="utf-8"))
namespace = {"__name__": "__main__", "display": lambda obj: None}
RETURN_OLD = """    model.load_state_dict(best_state_dict)
    return dict(validation=predict_dt_per_ha(inputs_val, is_val_row),
                test=predict_dt_per_ha(inputs_test, is_test_row))"""
RETURN_NEW = """    model.load_state_dict(best_state_dict)
    return dict(validation=predict_dt_per_ha(inputs_val, is_val_row),
                test=predict_dt_per_ha(inputs_test, is_test_row),
                state_dict=best_state_dict, target_mean=target_mean_train, target_std=target_std_train)"""
patched = False
for cell in notebook["cells"]:
    if cell["cell_type"] != "code" or "setup" not in cell["metadata"].get("tags", []):
        continue
    source = "".join(cell["source"])
    if "Data_Ashape" in source:
        continue  # the map geometry: not needed here and it needs the GADM county shapefile
    if RETURN_OLD in source:
        source = source.replace(RETURN_OLD, RETURN_NEW)
        patched = True
    exec(compile(source, "<setup>", "exec"), namespace)
assert patched, "the training function of the notebook changed; update RETURN_OLD"
globals().update({name: namespace[name] for name in [
    "np", "pd", "torch", "dev", "CACHE", "YIELD_PREDICTOR", "train_headline_model", "lean_tokens_headline",
    "coverage_headline", "headline_split", "cohort_metrics", "HEADLINE_WIDTH_TAG", "DT_HA_PER_T_HA"]})

# 2. train (or load) the ten models with their weights
weights_folder = os.path.join(CACHE, f"e07_weights_{HEADLINE_WIDTH_TAG}")
os.makedirs(weights_folder, exist_ok=True)
start = time.time()
checkpoints = []
for seed in range(N_MODELS):
    weights_file = os.path.join(weights_folder, f"model_{seed:02d}.pt")
    if not os.path.exists(weights_file):
        result = train_headline_model(lean_tokens_headline, coverage_headline, headline_split, seed)
        temporary_file = weights_file + ".partial"
        torch.save({"state_dict": result["state_dict"], "target_mean": result["target_mean"],
                    "target_std": result["target_std"], "seed": seed, "test": result["test"]}, temporary_file)
        os.replace(temporary_file, weights_file)
        print(f"model {seed}: trained ({time.time() - start:.0f}s)", flush=True)
    checkpoints.append(torch.load(weights_file, weights_only=False))

# 3. the probe on the 2025 rows
is_test_row = headline_split["is_test_row"]
is_train_row = headline_split["is_train_row"]
history_real = headline_split["yield_history"][is_test_row].astype(np.float32)
history_training_mean = float(headline_split["yield_history"][is_train_row].mean())
tokens_test = torch.tensor(lean_tokens_headline[is_test_row], device=dev)
coverage_test = torch.tensor(coverage_headline[is_test_row], device=dev)
HISTORY_INPUTS = {
    "real history": history_real,
    "history input = 0 dt/ha": np.zeros_like(history_real),
    "history input = -1 dt/ha": np.full_like(history_real, -1.0),
    f"history input = training mean ({history_training_mean:.1f} dt/ha)": np.full_like(history_real, history_training_mean),
}

predictions = {name: [] for name in HISTORY_INPUTS}
reproduction_differences = []
for checkpoint in checkpoints:
    model = YIELD_PREDICTOR(lean_tokens_headline.shape[2]).to(dev)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for name, history_input in HISTORY_INPUTS.items():
        with torch.no_grad():
            standardized = model(tokens_test, torch.tensor(history_input[:, None], device=dev), coverage_test).squeeze(1).cpu().numpy()
        # the real history is always added back: only the network's input changes
        predictions[name].append(standardized * checkpoint["target_std"] + checkpoint["target_mean"] + history_real)
    cached = np.load(os.path.join(CACHE, f"e07_transformer_bank_{HEADLINE_WIDTH_TAG}", f"model_{checkpoint['seed']:02d}.npz"))["test"]
    reproduction_differences.append(float(np.abs(predictions["real history"][-1] - cached).mean()))

print(f"\nreproduction: mean absolute difference to the cached E7 models, per model (dt/ha): "
      f"{np.round(reproduction_differences, 3).tolist()}")
rows = []
for name, bank in predictions.items():
    ensemble = np.mean(bank, axis=0)
    metrics = cohort_metrics(ensemble)
    rows.append({"network input": name,
                 "10d RMSE (t/ha)": metrics["headline"]["RMSE"], "10d MAE (t/ha)": metrics["headline"]["MAE"],
                 "10d R2": metrics["headline"]["R2"], "20d RMSE (t/ha)": metrics["full"]["RMSE"],
                 "mean predicted deviation from history (t/ha)": float((ensemble - history_real).mean()) / DT_HA_PER_T_HA,
                 "sd of predicted deviation (t/ha)": float((ensemble - history_real).std()) / DT_HA_PER_T_HA})
table = pd.DataFrame(rows)
output_folder = os.path.join(CACHE, "history_input_probe")
os.makedirs(output_folder, exist_ok=True)
table.to_csv(os.path.join(output_folder, "results.csv"), index=False)
actual_deviation = (headline_split["yield_actual"][is_test_row] - history_real) / DT_HA_PER_T_HA
print(f"observed deviation from history in 2025: mean {actual_deviation.mean():.2f} t/ha, sd {actual_deviation.std():.2f} t/ha")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(table.round(3).to_string(index=False))
print(f"done ({time.time() - start:.0f}s)")
