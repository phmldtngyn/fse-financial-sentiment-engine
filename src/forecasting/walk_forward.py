"""Walk-forward Cross-Validation cho time series."""
import numpy as np


def make_folds(n_samples, train_window, val_window, test_window, step):
    """Sinh các fold (train_start, train_end, val_start, val_end, test_start, test_end).

    Rolling window: train trượt đều mỗi bước step.
    """
    folds = []
    start = 0
    while start + train_window + val_window + test_window <= n_samples:
        tr_start = start
        tr_end = start + train_window
        va_start = tr_end
        va_end = va_start + val_window
        te_start = va_end
        te_end = te_start + test_window
        folds.append({
            "train": (tr_start, tr_end),
            "val": (va_start, va_end),
            "test": (te_start, te_end),
        })
        start += step
    return folds


def run_walk_forward(model_factory, X, y, folds, verbose=False):
    """Chạy CV, trả về predictions + metrics per fold.

    Args:
        model_factory: callable → model (đã cấu hình)
        X, y: np.ndarray
        folds: list từ make_folds

    Returns:
        {
            "predictions": [(test_indices, y_pred, y_true), ...],
            "fold_models": [model, ...],
        }
    """
    results = {"predictions": [], "fold_models": []}

    for i, fold in enumerate(folds):
        tr_s, tr_e = fold["train"]
        te_s, te_e = fold["test"]

        X_tr, y_tr = X[tr_s:tr_e], y[tr_s:tr_e]
        X_te, y_te = X[te_s:te_e], y[te_s:te_e]

        if len(X_tr) < 10 or len(X_te) < 1:
            continue

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        results["predictions"].append({
            "fold": i,
            "test_start": te_s,
            "test_end": te_e,
            "y_true": y_te,
            "y_pred": y_pred,
        })
        results["fold_models"].append(model)

        if verbose:
            print(f"  Fold {i}: train={len(X_tr)} test={len(X_te)}")

    return results


def concatenate_predictions(results):
    """Gộp y_true và y_pred của tất cả các fold."""
    y_true_all = np.concatenate([p["y_true"] for p in results["predictions"]])
    y_pred_all = np.concatenate([p["y_pred"] for p in results["predictions"]])
    return y_true_all, y_pred_all