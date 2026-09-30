"""The 30 headline models: train them (or load them once saved) and evaluate them.

The weights are not part of the repository. This script writes them to cache/e07_weights_width256/: the 30
seeded models of the headline configuration (E7 models
0-29 of 08_experiments.ipynb). Their mean is the reference ensemble of sections 2-4 of the thesis and the 2025
fold of the leave-one-year-out evaluation: RMSE 0.510, MAE 0.395, R2 0.867 on the 282 counties of the 10d test
set. (The headline itself, 01_headline_register.ipynb, is an unseeded ensemble of the same configuration:
0.508 / 0.395 / 0.868.)

A missing weight file is trained with the notebook's own setup and training code (same seed, same protocol)
and saved, about seven minutes for all 30 on a GPU. With all 30 files present the script only loads them and
evaluates the ensemble on the 2025 test season.
Every model is checked against its cached prediction in cache/e07_transformer_bank_width256/.

Needs Claude/data_cache/ (the modelling tables) and runs on CPU or GPU.

    python Claude/final_results_claude/headline_weights.py
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

FINAL_RESULTS = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(FINAL_RESULTS, "08_experiments.ipynb")
N_MODELS = 30

if sys.platform == "win32":
    import ctypes
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
    "coverage_headline", "headline_split", "cohort_metrics", "HEADLINE_WIDTH_TAG"]})

# 2. load every model, training the missing ones
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
    checkpoints.append(torch.load(weights_file, map_location=dev, weights_only=False))

# 3. predict the 2025 season with every model and average
is_test_row = headline_split["is_test_row"]
history_test = headline_split["yield_history"][is_test_row].astype(np.float32)
tokens_test = torch.tensor(lean_tokens_headline[is_test_row], device=dev)
history_input = torch.tensor(history_test[:, None], device=dev)
coverage_test = torch.tensor(coverage_headline[is_test_row], device=dev)
bank = []
largest_difference = 0.0
for checkpoint in checkpoints:
    model = YIELD_PREDICTOR(lean_tokens_headline.shape[2]).to(dev)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        standardized = model(tokens_test, history_input, coverage_test).squeeze(1).cpu().numpy()
    # the network predicts the standardized deviation from the history; undo both (dt/ha)
    bank.append(standardized * checkpoint["target_std"] + checkpoint["target_mean"] + history_test)
    cached_file = os.path.join(CACHE, f"e07_transformer_bank_{HEADLINE_WIDTH_TAG}", f"model_{checkpoint['seed']:02d}.npz")
    if os.path.exists(cached_file):
        largest_difference = max(largest_difference, float(np.abs(bank[-1] - np.load(cached_file)["test"]).max()))

ensemble = np.mean(bank, axis=0)
metrics = cohort_metrics(ensemble)
print(f"\n{len(bank)} models, largest difference to the cached predictions: {largest_difference:.4f} dt/ha")
for name, label in [("headline", "10d test set (282 counties)"), ("full", "20d test set (426 counties)")]:
    values = metrics[name]
    print(f"{label}: RMSE {values['RMSE']:.3f} t/ha, MAE {values['MAE']:.3f} t/ha, R2 {values['R2']:.3f}")
print(f"done ({time.time() - start:.0f}s)")
