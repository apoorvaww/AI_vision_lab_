"""
search.py
---------
The actual "AutoML" loop: for every split candidate, try every selected
model with every hyperparameter combination in the search space, score
each trial, and return a sorted leaderboard.

This is intentionally a simple grid search (see the note in the project
README about swapping this for Optuna/Ray Tune once the search space
grows -- the loop structure stays the same, only how you pick the next
combination changes).

Model dispatch is driven entirely by `MODEL_REGISTRY` in `models.py`, so
adding a new model doesn't require touching this file.
"""

import itertools
from typing import Callable, Dict, List, Optional
import numpy as np

from .models import MODEL_REGISTRY, TrialResult
from .splits import SplitCandidate


class SearchCancelled(Exception):
    """Raised internally when a `should_stop` callback returns True."""


def _param_grid(param_space: Dict[str, list]) -> List[dict]:
    """Turns {'a': [1,2], 'b': [10,20]} into every combination as dicts."""
    if not param_space:
        return [{}]
    keys = list(param_space.keys())
    combos = itertools.product(*param_space.values())
    return [dict(zip(keys, combo)) for combo in combos]


def run_search(
    splits: List[SplitCandidate],
    search_space: Dict[str, Dict[str, list]],
    metric: str = "f1_macro",
    progress_callback: Optional[Callable] = None,
    epoch_callback: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    is_cv: bool = False,
) -> List[TrialResult]:
    """AutoML search supporting both standard holdout splits and K-Fold cross validation.

    When `is_cv=True`, all folds in `splits` are evaluated per model configuration,
    and results are aggregated into a single TrialResult containing fold-wise breakdown
    and mean ± std metrics.
    """
    trials: List[TrialResult] = []

    if is_cv:
        # Cross-validation: group trials by model and hyperparameter combination
        model_combos = []
        for model_key, param_space in search_space.items():
            entry = MODEL_REGISTRY.get(model_key)
            if entry is None or not entry["available"]:
                continue
            for params in _param_grid(param_space):
                model_combos.append((model_key, params))

        total_steps = len(model_combos) * len(splits)
        step = 0

        for model_key, params in model_combos:
            entry = MODEL_REGISTRY[model_key]
            fold_results: List[TrialResult] = []

            for split in splits:
                step += 1
                if should_stop is not None and should_stop():
                    if progress_callback:
                        progress_callback(step - 1, total_steps, "Cancelled")
                    raise SearchCancelled()

                fold_name = split.label
                if progress_callback:
                    progress_callback(
                        step - 1, total_steps, f"Training {entry['label']} on {fold_name}"
                    )

                p_copy = dict(params)
                if epoch_callback:
                    def _make_epoch_cb(step_idx, total_count, label, fold_str):
                        def _cb(info):
                            info["job_step"] = step_idx
                            info["total_jobs"] = total_count
                            info["model_name"] = f"{label} ({fold_str})"
                            epoch_callback(info)
                        return _cb
                    p_copy["_epoch_callback"] = _make_epoch_cb(step, total_steps, entry["label"], fold_name)

                res = entry["fn"](split.train_files, split.test_files, split.label, split.ratio, p_copy)
                if res is not None:
                    fold_results.append(res)
                if progress_callback:
                    progress_callback(step, total_steps, f"Completed {entry['label']} on {fold_name}")

            if fold_results:
                accs = [r.accuracy for r in fold_results]
                f1s = [r.f1_macro for r in fold_results]
                precs = [r.precision_macro for r in fold_results]
                recs = [r.recall_macro for r in fold_results]
                aucs = [r.roc_auc for r in fold_results]
                train_accs = [r.train_accuracy for r in fold_results]
                gaps = [r.generalization_gap for r in fold_results]
                durations = [r.train_seconds for r in fold_results]

                fold_metrics = [
                    {
                        "fold": idx + 1,
                        "split_label": r.split_label,
                        "accuracy": round(r.accuracy, 4),
                        "precision_macro": round(r.precision_macro, 4),
                        "recall_macro": round(r.recall_macro, 4),
                        "f1_macro": round(r.f1_macro, 4),
                        "roc_auc": round(r.roc_auc, 4),
                        "train_accuracy": round(r.train_accuracy, 4),
                        "train_seconds": round(r.train_seconds, 2),
                    }
                    for idx, r in enumerate(fold_results)
                ]

                # Select representative fold for visualization (prefer one with XAI visualizations)
                rep = fold_results[0]
                has_xai = any(bool(r.xai_visualizations) for r in fold_results)
                if has_xai:
                    for r in fold_results:
                        if r.xai_visualizations:
                            rep = r
                            break
                else:
                    mean_acc = float(np.mean(accs))
                    rep = min(fold_results, key=lambda r: abs(r.accuracy - mean_acc))

                aggregated = TrialResult(
                    model_name=rep.model_name,
                    model_family=rep.model_family,
                    params=rep.params,
                    split_label=f"{len(splits)}-Fold CV",
                    split_ratio=rep.split_ratio,
                    accuracy=float(np.mean(accs)),
                    f1_macro=float(np.mean(f1s)),
                    train_seconds=float(sum(durations)),
                    precision_macro=float(np.mean(precs)),
                    recall_macro=float(np.mean(recs)),
                    roc_auc=float(np.mean(aucs)),
                    roc_curve=rep.roc_curve,
                    param_count=rep.param_count,
                    param_count_str=rep.param_count_str,
                    prediction_sample=rep.prediction_sample,
                    confusion_matrix=rep.confusion_matrix,
                    class_names=rep.class_names,
                    train_accuracy=float(np.mean(train_accs)),
                    generalization_gap=float(np.mean(gaps)),
                    training_log=rep.training_log,
                    training_history=rep.training_history,
                    xai_enabled=rep.xai_enabled,
                    xai_method=rep.xai_method,
                    xai_visualizations=rep.xai_visualizations,
                    xai_note=rep.xai_note,
                    is_cv=True,
                    cv_folds=len(splits),
                    fold_metrics=fold_metrics,
                    accuracy_std=float(np.std(accs)),
                    f1_macro_std=float(np.std(f1s)),
                    precision_macro_std=float(np.std(precs)),
                    recall_macro_std=float(np.std(recs)),
                    roc_auc_std=float(np.std(aucs)),
                )
                trials.append(aggregated)

    else:
        # Standard holdout search loop
        jobs = []  # (model_key, split, params)
        for split in splits:
            for model_key, param_space in search_space.items():
                entry = MODEL_REGISTRY.get(model_key)
                if entry is None or not entry["available"]:
                    continue
                for params in _param_grid(param_space):
                    jobs.append((model_key, split, params))

        total = len(jobs)

        for i, (model_key, split, params) in enumerate(jobs, start=1):
            if should_stop is not None and should_stop():
                if progress_callback:
                    progress_callback(i - 1, total, "Cancelled")
                raise SearchCancelled()

            entry = MODEL_REGISTRY[model_key]
            if progress_callback:
                progress_callback(
                    i - 1, total, f"Training {entry['label']} on {split.label} split"
                )

            p_copy = dict(params)
            if epoch_callback:
                def _make_epoch_cb(step_idx, total_jobs_count, label):
                    def _cb(info):
                        info["job_step"] = step_idx
                        info["total_jobs"] = total_jobs_count
                        if "model_name" not in info:
                            info["model_name"] = label
                        epoch_callback(info)
                    return _cb
                p_copy["_epoch_callback"] = _make_epoch_cb(i, total, entry["label"])

            result = entry["fn"](split.train_files, split.test_files, split.label, split.ratio, p_copy)

            if result is not None:
                trials.append(result)
            if progress_callback:
                progress_callback(i, total, f"Completed {entry['label']} on {split.label} split")

    trials.sort(key=lambda t: getattr(t, metric), reverse=True)

    if progress_callback:
        progress_callback(100, 100, "Search complete")

    return trials
