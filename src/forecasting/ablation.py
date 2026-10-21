"""Ablation Study — chạy 4 mô hình × 3 kịch bản A/B/C."""
import yaml
import numpy as np
from pathlib import Path

from src.forecasting.features import build_features
from src.forecasting.models import build_model
from src.forecasting.walk_forward import make_folds, run_walk_forward, concatenate_predictions
from src.forecasting.metrics import all_metrics, diebold_mariano


SCENARIOS = {
    "A": "Market only (vol_lag, return_lag)",
    "B": "Market + sentiment (S_t, delta_S_t)",
    "C": "Market + sentiment + RAG (v_ctx)",
}

MODELS = ["linear_regression", "decision_tree", "knn", "ann_mlp"]


def run_full_ablation(df, config_path="config/forecasting.yaml", verbose=True):
    """Chạy toàn bộ ablation: 4 models × 3 scenarios × walk-forward CV."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fc = cfg["forecasting"]
    results = {}

    for scenario in SCENARIOS:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Scenario {scenario}: {SCENARIOS[scenario]}")
            print('='*70)

        try:
            X, y, feat_names, dates = build_features(
                df,
                target=fc["target"],
                scenario=scenario,
            )
        except Exception as e:
            print(f"  ⚠️ Skip {scenario}: {e}")
            continue

        if len(X) < fc["min_train_samples"]:
            print(f"  ⚠️ Skip {scenario}: too few samples ({len(X)})")
            continue

        folds = make_folds(
            len(X),
            fc["window_train_days"],
            fc["window_val_days"],
            fc["window_test_days"],
            fc["step_days"],
        )

        if not folds:
            print(f"  ⚠️ Skip {scenario}: no valid folds")
            continue

        scenario_results = {}

        for model_name in MODELS:
            if verbose:
                print(f"\n  Model: {model_name} ({len(folds)} folds)")

            model_cfg = cfg["models"].get(model_name, {})

            def factory(name=model_name, mc=model_cfg):
                return build_model(name, **mc)

            wf = run_walk_forward(factory, X, y, folds, verbose=False)
            y_true, y_pred = concatenate_predictions(wf)
            metrics = all_metrics(y_true, y_pred)

            if verbose:
                print(f"    RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
                      f"R²={metrics['r2']:.4f}")

            scenario_results[model_name] = {
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
            }

        results[scenario] = scenario_results

    return results


def print_summary(results):
    """In bảng so sánh 4 model × 3 scenario."""
    print(f"\n{'='*80}")
    print("SUMMARY — RMSE / MAE / R²")
    print('='*80)

    header = f"{'Model':<20}" + "".join([f"{s:>18}" for s in ["A", "B", "C"]])
    print(header)
    print('-' * len(header))

    for model in MODELS:
        row = f"{model:<20}"
        for scenario in ["A", "B", "C"]:
            if scenario in results and model in results[scenario]:
                m = results[scenario][model]["metrics"]
                row += f"{m['rmse']:>7.3f}/{m['mae']:>6.3f}/{m['r2']:>6.2f}  "
            else:
                row += f"{'—':>18}"
        print(row)

    # DM test: model tốt nhất scenario A vs B, B vs C (LR làm baseline)
    print(f"\n{'='*80}")
    print("Diebold-Mariano test (Linear Regression baseline)")
    print('='*80)

    for model in MODELS:
        for s1, s2 in [("A", "B"), ("B", "C")]:
            if (s1 in results and s2 in results
                and model in results[s1] and model in results[s2]):
                e1 = results[s1][model]["y_true"] - results[s1][model]["y_pred"]
                e2 = results[s2][model]["y_true"] - results[s2][model]["y_pred"]
                if len(e1) == len(e2):
                    stat, p = diebold_mariano(e1, e2)
                    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
                    print(f"  {model:<20} {s1}→{s2}: DM={stat:>7.3f}, p={p:.4f} {sig}")