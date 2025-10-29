# SAXS Peak Alignment - Improved Algorithm Guide

## Overview

This improved alignment algorithm is based on our approach and systematically tries normalizing by **each peak** in our observed data (Yd) to find the best match with reference patterns (Yr).

## Algorithm Explanation

### Key Concept

The algorithm recognizes that we don't know which peak in Yd corresponds to which peak in Yr. So it tries **all possibilities**:

1. **For each peak i in Yd**: Assume this peak is the "anchor" and scale Yr accordingly
2. **Match all other peaks**: Find where each peak in Yd best matches the scaled Yr
3. **Refine**: Remove duplicate matches (same Yr peak matched multiple times)
4. **Score**: Evaluate based on number of valid matches and MSE
5. **Select best**: Choose the normalization that gives the best score

### our Results

**Observed peaks**: [0.01517034, 0.02737475, 0.04016032, 0.05352705, 0.06602204]

**Best Match: BCC (Body-Centered Cubic)**
- ✅ Valid matches: **4 out of 5** peaks
- ✅ RMSE: 4.9 × 10⁻⁴
- ✅ Normalization: By first peak (peak #0)
- ✅ Scale factor: 0.015170
- ❌ Outlier: Peak #1 (q = 0.027375)

**Top 3 Rankings:**
1. **BCC** - 4/5 matches, RMSE=4.91e-4
2. **HCP** - 4/5 matches, RMSE=4.91e-4  
3. **FCC** - 4/5 matches, RMSE=7.35e-4

## Quick Start

```python
import numpy as np
from improved_alignment import get_best_alignment, compare_structures, generate_theoretical_peaks

# our observed SAXS peaks
Yd = np.array([0.01517034, 0.02737475, 0.04016032, 0.05352705, 0.06602204])

# Generate normalized reference (starts at 1.0)
Yr_bcc = generate_theoretical_peaks('bcc', normalized=True, max_peaks=20)

# Align
result = get_best_alignment(Yd, Yr_bcc, tolerance=0.03)

# Check results
print(f"Valid matches: {result['num_matches_valid']}/{len(Yd)}")
print(f"Matched Yd indices: {result['yd_indices_valid']}")
print(f"Matched Yr indices: {result['yr_indices_valid']}")
print(f"Outliers: {result['yd_outliers']}")
print(f"RMSE: {result['rmse_valid']:.6e}")
```

## Compare Multiple Structures

```python
# Generate multiple reference structures
structures = {
    'BCC': generate_theoretical_peaks('bcc', normalized=True),
    'FCC': generate_theoretical_peaks('fcc', normalized=True),
    'HCP': generate_theoretical_peaks('hcp', normalized=True),
    'SC': generate_theoretical_peaks('sc', normalized=True),
    'Diamond': generate_theoretical_peaks('diamond', normalized=True)
}

# Compare all at once
results = compare_structures(Yd, structures, tolerance=0.03)

# Results are sorted by score
for name, result in results.items():
    print(f"{name}: {result['num_matches_valid']}/{len(Yd)} matches, "
          f"RMSE={result['rmse_valid']:.4e}")
```

## Key Functions

### 1. get_align_score(Yd, Yr, tolerance=0.03)

The core alignment function from our algorithm.

**Returns:**
- `score`: Dict with {norm_idx: [num_all, mse_all, num_valid, mse_valid]}
- `Align_dict`: Raw alignment for each normalization
- `Align_dict_refine`: Refined alignment (no duplicate Yr indices)

### 2. get_best_alignment(Yd, Yr, tolerance=0.03)

Wrapper that finds the best normalization and returns detailed results.

**Returns dictionary with:**
```python
{
    'norm_peak_idx': 0,                    # Which Yd peak was used for normalization
    'scale_factor': 0.015170,              # The scaling factor
    'yd_indices_valid': [0, 2, 3, 4],     # Valid matched peaks in Yd
    'yr_indices_valid': [0, 6, 11, 18],   # Corresponding peaks in Yr
    'yd_outliers': [1],                    # Outlier peaks in Yd
    'num_matches_valid': 4,                # Number of valid matches
    'rmse_valid': 4.91e-04,                # RMSE for valid matches
    'match_fraction': 0.8,                 # Fraction of Yd matched
    'score': 4.0,                          # Overall quality score
    # ... and more
}
```

### 3. compare_structures(Yd, structures_dict, tolerance=0.03)

Compare against multiple reference structures.

**Parameters:**
- `Yd`: our observed peaks
- `structures_dict`: {name: reference_peaks}
- `tolerance`: Relative tolerance (0.03 = 3%)
- `score_weights`: [weight_matches, weight_mse] (default: [1.0, 1000])

### 4. generate_theoretical_peaks(structure_type, normalized=True)

Generate reference peak positions.

**Parameters:**
- `structure_type`: 'fcc', 'bcc', 'sc', 'diamond', 'hcp'
- `normalized`: If True, first peak = 1.0 (for alignment)
                If False, use absolute q-values
- `q_first`: First peak position (when normalized=False)

**Available structures:**
- **'fcc'**: Face-centered cubic
- **'bcc'**: Body-centered cubic
- **'sc'**: Simple cubic
- **'diamond'**: Diamond structure
- **'hcp'**: Hexagonal close-packed

## Important: Normalized vs Absolute Peaks

The alignment algorithm expects **normalized** reference peaks (first peak = 1.0):

```python
# For alignment - use normalized=True
Yr_norm = generate_theoretical_peaks('bcc', normalized=True)
# Output: [1.0, 1.414, 1.732, 2.0, ...]

# For plotting with actual q-values - use normalized=False
Yr_abs = generate_theoretical_peaks('bcc', normalized=False, q_first=0.015)
# Output: [0.015, 0.0212, 0.0260, 0.030, ...]
```

## Parameters Tuning

### tolerance (default: 0.03)

Relative tolerance for considering a match valid.

- **0.01-0.02**: High-quality data, strict matching
- **0.03-0.05**: Medium quality (recommended)
- **0.05-0.10**: Noisy data, relaxed matching

Example: With tolerance=0.03 and peak at q=0.03, a match is valid if within ±0.0009

### score_weights (default: [1.0, 1000])

Balance between number of matches and error:
- `[weight_matches, weight_mse]`
- Score = `num_valid * weight_matches - mse_valid * weight_mse`

Adjust to prioritize:
- More matches: Increase first weight, e.g., `[2.0, 1000]`
- Lower error: Increase second weight, e.g., `[1.0, 5000]`

## Visualization

```python
from improved_alignment import plot_alignment, plot_comparison

# Plot single alignment
fig = plot_alignment(Yd, Yr, result, title="BCC Match")
plt.show()

# Plot comparison of top structures
fig = plot_comparison(Yd, results, top_n=5)
plt.show()
```

## Custom Reference Patterns

We can provide our own reference patterns:

```python
# our custom reference (must be normalized to first peak = 1.0)
Yr_custom = np.array([1.0, 1.5, 2.1, 2.8, 3.4, 4.0])

# Align
result = get_best_alignment(Yd, Yr_custom, tolerance=0.03)
```

## Understanding the Results

### What does "outlier" mean?

A peak in Yd that either:
1. Doesn't match any peak in Yr within tolerance
2. Matches a Yr peak that's already matched by another Yd peak (and lost in refinement)

In our case, peak #1 (q=0.027375) is an outlier for BCC structure.

### What does "normalization by peak #X" mean?

The algorithm tried using Yd[X] as the scaling reference. For our best match:
- Peak #0 (q=0.015170) was used
- This means Yr was scaled by 0.015170
- So Yr[0] * 0.015170 = 1.0 * 0.015170 = 0.015170 ≈ Yd[0]

### Valid vs All matches

- **All matches**: Total number of Yd peaks matched to some Yr peak
- **Valid matches**: Matches within the specified tolerance
- Some matches may be technically found but outside tolerance threshold

## Algorithm Step-by-Step Example

Let's trace through for our data with BCC:

**Step 1**: Try normalizing by Yd[0] = 0.015170
- Yr_scaled = [1.0, 1.414, ...] * 0.015170 = [0.015170, 0.021445, ...]

**Step 2**: Match each Yd to closest in Yr_scaled
- Yd[0]=0.015170 → closest is Yr[0]=0.015170 ✓
- Yd[1]=0.027375 → closest is Yr[3]=0.03034 (6% error) ❌
- Yd[2]=0.040160 → closest is Yr[6]=0.04013 ✓
- Yd[3]=0.053527 → closest is Yr[11]=0.05345 ✓
- Yd[4]=0.066022 → closest is Yr[18]=0.06591 ✓

**Step 3**: Check tolerance (3%)
- 4 matches are within 3% → Valid!
- 1 match (Yd[1]) is outside → Outlier

**Step 4**: Score = 4 valid matches (excellent!)

## Tips for Best Results

1. **Start with 3% tolerance** and adjust based on data quality

2. **Check multiple top matches** - sometimes scores are similar

3. **Look at outliers** - they might indicate:
   - Missing peaks in reference
   - Extra peaks from impurities
   - Wrong structure entirely

4. **Visualize results** - plots reveal patterns scores miss

5. **Try our own references** if standard structures don't fit well

## Differences from Original Approach

**our algorithm (now implemented):**
- ✅ Tries normalizing by **every** peak in Yd
- ✅ More robust to wrong first peak
- ✅ Handles missing/extra peaks better
- ✅ Simpler scoring based on matches and MSE

**My original approach:**
- Limited normalization strategies
- More complex tolerance handling
- Less systematic exploration

**our approach is better!** It's more exhaustive and finds better alignments.

## Questions?

The algorithm essentially asks: "What if THIS peak is actually THAT theoretical peak?" and tries all possibilities systematically.

For more examples, see the main script output or run:

```bash
python improved_alignment.py
```

This will show our data analyzed against all standard crystal structures.