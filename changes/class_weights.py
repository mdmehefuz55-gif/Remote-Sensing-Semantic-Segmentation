"""Median Frequency Class Weighting for LoveDA Semantic Segmentation.

Computes class weights from the official LoveDA dataset pixel frequency
distribution (measured from the full 2,522-pair training set in Phase 2).

Class Frequencies (% of total pixels, excluding ignore index 0):
    1 - Background:   34.72%  (dominant class — was suppressing minorities)
    2 - Building:      9.20%
    3 - Road:          4.86%  (rare thin linear class)
    4 - Water:         8.24%
    5 - Barren:        4.71%  (rarest class — severely under-detected)
    6 - Forest:       12.17%
    7 - Agriculture:  22.78%

Median Frequency Weight formula:
    weight_c = median(all_valid_frequencies) / frequency_c

Result: rare classes (Barren, Road) get high weights; dominant classes
(Background, Agriculture) get low weights, forcing the model to learn all
7 categories equally during backpropagation.
"""

import torch


# ---------------------------------------------------------------------------
# Pixel frequencies from LoveDA Phase 2 Dataset Verification Report
# Index 0 = No-data / Ignore (excluded from weighting)
# Index 1-7 = the 7 semantic land-cover categories
# ---------------------------------------------------------------------------
LOVA_DA_PIXEL_FREQUENCIES = torch.tensor([
    0.0331,   # 0: No-data / Ignore  (3.31%)  — excluded from weighting
    0.3472,   # 1: Background        (34.72%) — most dominant
    0.0920,   # 2: Building          (9.20%)
    0.0486,   # 3: Road              (4.86%)
    0.0824,   # 4: Water             (8.24%)
    0.0471,   # 5: Barren            (4.71%)  — rarest class
    0.1217,   # 6: Forest            (12.17%)
    0.2278,   # 7: Agriculture       (22.78%)
], dtype=torch.float32)


def compute_median_frequency_weights(
    frequencies: torch.Tensor = LOVA_DA_PIXEL_FREQUENCIES,
    ignore_index: int = 0,
    num_classes: int = 8,
) -> torch.Tensor:
    """Compute Median Frequency class weights for semantic segmentation.

    Parameters
    ----------
    frequencies : torch.Tensor
        Per-class pixel frequency tensor of shape [num_classes].
        Index 0 is expected to be the ignore/no-data class.
    ignore_index : int
        Class index to assign zero weight (excluded from loss computation).
    num_classes : int
        Total number of classes including ignore index.

    Returns
    -------
    torch.Tensor
        Class weight tensor of shape [num_classes] where:
        - ignore_index class gets weight = 0.0
        - valid classes get weight = median_freq / class_freq
        - rare classes get HIGH weights, dominant classes get LOW weights
    """
    # Extract frequencies of valid classes (exclude ignore_index)
    valid_indices = [i for i in range(num_classes) if i != ignore_index]
    valid_freqs = frequencies[valid_indices]

    # Compute median of valid class frequencies
    median_freq = torch.median(valid_freqs)

    # Allocate weight tensor — ignore index always gets 0.0
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    for i in valid_indices:
        class_weights[i] = median_freq / frequencies[i]

    return class_weights


def get_lova_da_class_weights() -> torch.Tensor:
    """Return pre-computed Median Frequency class weights for LoveDA dataset.

    Returns
    -------
    torch.Tensor of shape [8]:
        weights[0] = 0.0000  (No-data / Ignore — excluded)
        weights[1] = 0.2650  (Background — dominant, gets LOW weight)
        weights[2] = 1.0000  (Building — reference class = median)
        weights[3] = 1.8930  (Road — rare, gets HIGH weight)
        weights[4] = 1.1165  (Water)
        weights[5] = 1.9568  (Barren — rarest, gets HIGHEST weight)
        weights[6] = 0.7559  (Forest)
        weights[7] = 0.4039  (Agriculture — somewhat dominant)
    """
    return compute_median_frequency_weights(
        frequencies=LOVA_DA_PIXEL_FREQUENCIES,
        ignore_index=0,
        num_classes=8,
    )


def print_class_weights_table(weights: torch.Tensor) -> None:
    """Print a formatted table of class weights for inspection."""
    from src.utils.class_mapping import CLASS_NAMES

    print("\n" + "=" * 65)
    print("  MEDIAN FREQUENCY CLASS WEIGHTS — LoveDA Dataset")
    print("=" * 65)
    print(f"  {'ID':<5} {'Class Name':<20} {'Frequency':>12} {'Weight':>10}")
    print("-" * 65)
    for i, (freq, weight) in enumerate(zip(LOVA_DA_PIXEL_FREQUENCIES, weights)):
        name = CLASS_NAMES.get(i, f"Class {i}")
        flag = "  << ignore" if i == 0 else ""
        print(f"  {i:<5} {name:<20} {freq.item():>11.2%} {weight.item():>10.4f}{flag}")
    print("=" * 65)
    print(f"  Median Frequency: {torch.median(LOVA_DA_PIXEL_FREQUENCIES[1:]).item():.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    weights = get_lova_da_class_weights()
    print_class_weights_table(weights)
