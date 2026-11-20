"""Leakage-safe train/test split generation."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy replacements for sklearn split utilities
# (sklearn 1.9.1 compiled extensions are blocked by Application Control policy
#  on this machine; these pure-Python versions have identical semantics)
# ---------------------------------------------------------------------------

def _stratified_split(paths, labels, train_size: float, seed: int = 42):
    """Stratified train/test split without sklearn."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(labels))
    train_idx, test_idx = [], []
    for cls in unique:
        idx = np.where(np.array(labels) == cls)[0]
        idx = rng.permutation(idx)
        n_train = max(1, int(round(len(idx) * train_size)))
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    # Shuffle within each split to avoid class-ordered output
    train_idx = rng.permutation(train_idx).tolist()
    test_idx = rng.permutation(test_idx).tolist()
    paths = list(paths)
    return (
        [paths[i] for i in train_idx],
        [paths[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


def _stratified_kfold_indices(labels, n_splits: int, seed: int = 42):
    """Return (train_idx, test_idx) pairs for stratified k-fold."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    n = len(labels)
    unique = sorted(set(labels.tolist()))
    # Build per-class shuffled index lists
    class_indices = {cls: rng.permutation(np.where(labels == cls)[0]).tolist() for cls in unique}
    # Assign each sample to a fold
    fold_assignment = np.empty(n, dtype=int)
    for cls in unique:
        idx_list = class_indices[cls]
        for i, sample_idx in enumerate(idx_list):
            fold_assignment[sample_idx] = i % n_splits
    for fold in range(n_splits):
        test_idx = np.where(fold_assignment == fold)[0].tolist()
        train_idx = np.where(fold_assignment != fold)[0].tolist()
        yield train_idx, test_idx


def _kfold_indices(n: int, n_splits: int, seed: int = 42):
    """Return (train_idx, test_idx) pairs for plain k-fold."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    fold_sizes = [(n // n_splits) + (1 if i < n % n_splits else 0) for i in range(n_splits)]
    current = 0
    for size in fold_sizes:
        test_idx = indices[current: current + size]
        train_idx = indices[:current] + indices[current + size:]
        yield train_idx, test_idx
        current += size


@dataclass
class SplitCandidate:
    ratio: float
    train_files: List[Tuple[str, str]]
    test_files: List[Tuple[str, str]]
    fold_index: Optional[int] = None
    total_folds: Optional[int] = None
    is_cv: bool = False

    @property
    def label(self) -> str:
        if self.is_cv and self.fold_index is not None and self.total_folds is not None:
            return f"Fold {self.fold_index + 1}/{self.total_folds}"
        if self.ratio >= 0.999:
            return "Provided test split"
        pct = int(round(self.ratio * 100))
        return f"{pct}/{100 - pct}"


def generate_splits(valid_files, ratios=(0.8,), seed=42):
    """Create stratified holdouts.

    A split is rejected when stratification cannot produce at least one test
    example for every class. This avoids silently producing misleading scores.
    """
    labels = [l for _, l in valid_files]
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required for classification.")
    min_class = min(labels.count(c) for c in set(labels))
    candidates = []
    for ratio in ratios:
        test_size = 1.0 - ratio
        if int(round(len(valid_files) * test_size)) < len(set(labels)) or min_class < 2:
            continue
        train_p, test_p, train_l, test_l = _stratified_split(
            [p for p, _ in valid_files], labels, train_size=ratio, seed=seed
        )
        candidates.append(
            SplitCandidate(ratio, list(zip(train_p, train_l)), list(zip(test_p, test_l)))
        )
    if not candidates:
        raise ValueError("The dataset is too small for the selected train/test split. Use more images per class or a larger test set.")
    return candidates


def generate_kfold_splits(valid_files: List[Tuple[str, str]], n_splits: int = 5, seed: int = 42) -> List[SplitCandidate]:
    """Generate n-fold cross-validation splits (stratified when class sizes permit).

    Each fold returns a SplitCandidate with fold_index, total_folds, and is_cv=True.
    """
    labels = [l for _, l in valid_files]
    unique_classes = set(labels)
    if len(unique_classes) < 2:
        raise ValueError("At least two classes are required for cross-validation.")
    if len(valid_files) < n_splits:
        raise ValueError(f"Dataset has only {len(valid_files)} images, which is fewer than {n_splits} folds.")

    min_class = min(labels.count(c) for c in unique_classes)
    paths = np.array([p for p, _ in valid_files])
    lbls = np.array(labels)

    # Use stratified k-fold if every class has at least n_splits samples, otherwise plain k-fold
    if min_class >= n_splits:
        splits_iter = _stratified_kfold_indices(labels, n_splits=n_splits, seed=seed)
    else:
        splits_iter = _kfold_indices(len(valid_files), n_splits=n_splits, seed=seed)

    candidates = []
    ratio = round(1.0 - (1.0 / n_splits), 2)
    for fold_idx, (train_idx, test_idx) in enumerate(splits_iter):
        train_p = paths[train_idx]
        train_l = lbls[train_idx]
        test_p = paths[test_idx]
        test_l = lbls[test_idx]
        candidates.append(
            SplitCandidate(
                ratio=ratio,
                train_files=list(zip(train_p.tolist(), train_l.tolist())),
                test_files=list(zip(test_p.tolist(), test_l.tolist())),
                fold_index=fold_idx,
                total_folds=n_splits,
                is_cv=True,
            )
        )
    return candidates

