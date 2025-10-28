# SAXS Peak Alignment - Usage Guide

## Overview
This code aligns 1D peak positions (SAXS dips) to reference patterns, handling:
- Missing peaks in your data
- Outlier/fake peaks
- Uncertain first peak position
- Multiple normalization strategies

## Quick Start

```python
import numpy as np
from saxs_alignment import PeakAligner, generate_theoretical_peaks

# Your observed SAXS peaks (q-values)
Yd_observed = np.array([0.01517034, 0.02737475, 0.04016032, 0.05352705, 0.06602204])

# Create reference pattern
Yr_reference = generate_theoretical_peaks('fcc', q_first=0.015, max_peaks=15)

# Create aligner
aligner = PeakAligner(
    relative_tolerance=0.03,  # 3% tolerance
    min_matches=2,            # Need at least 2 matches
    normalization_modes=['first', 'auto']  # Try both strategies
)

# Align
result = aligner.align(Yd_observed, Yr_reference)

# Check results
print(f"Matched {result['num_matches']} out of {len(Yd_observed)} peaks")
print(f"Scale factor: {result['scale_factor']:.4f}")
print(f"RMSE: {result['rmse']:.4e}")
print(f"Matched indices in Yd: {result['yd_matched']}")
print(f"Outlier indices: {result['yd_outliers']}")
```

## Compare Multiple Structures

```python
# Generate multiple references
structures = {
    'FCC': generate_theoretical_peaks('fcc', q_first=0.015, max_peaks=15),
    'BCC': generate_theoretical_peaks('bcc', q_first=0.015, max_peaks=15),
    'SC': generate_theoretical_peaks('sc', q_first=0.015, max_peaks=15),
}

# Compare all at once
results = aligner.compare_structures(Yd_observed, structures)

# Results are sorted by score (best first)
for structure_name, result in results.items():
    print(f"{structure_name}: score={result['score']:.2f}, "
          f"matches={result['num_matches']}/{len(Yd_observed)}")
```

## Using Your Own Reference Data

```python
# Instead of generate_theoretical_peaks, use your own computed data
Yr_custom = np.array([0.015, 0.021, 0.026, 0.030, 0.034, ...])

result = aligner.align(Yd_observed, Yr_custom)
```

## Key Parameters

### PeakAligner Parameters

- **relative_tolerance** (default: 0.03)
  - Relative tolerance for peak matching
  - 0.03 means 3% of the peak value
  - Example: For a peak at q=0.03, tolerance = 0.03 × 0.03 = 0.0009

- **absolute_tolerance** (default: None)
  - Optional absolute tolerance
  - If set, uses max(relative_tol, absolute_tol)

- **min_matches** (default: 2)
  - Minimum number of peaks that must match
  - Alignments with fewer matches are rejected

- **normalization_modes** (default: ['first', 'auto'])
  - 'first': Normalize by first peak (assumes first peak is correct)
  - 'auto': Try multiple scale factors based on different peak pairs
  - 'none': No normalization (use when data is already scaled)

## Result Dictionary

The `align()` method returns a dictionary with:

```python
{
    'scale_factor': 1.0234,           # Normalization factor applied
    'yd_matched': [1, 2, 3, 4],       # Indices of matched peaks in Yd
    'yr_matched': [2, 5, 9, 14],      # Corresponding indices in Yr
    'yd_outliers': [0],               # Outlier indices in Yd
    'yr_missing': [0, 1, 3, ...],     # Peaks in Yr not observed
    'residuals': [...],               # Residuals for matched peaks
    'mse': 0.000123,                  # Mean squared error
    'rmse': 0.0111,                   # Root mean squared error
    'num_matches': 4,                 # Number of matched peaks
    'match_fraction': 0.8,            # Fraction of Yd peaks matched
    'score': 3.16,                    # Overall quality score
    'normalization_mode': 'auto'      # Which mode was used
}
```

## Visualization

```python
# Plot single alignment
fig = aligner.plot_alignment(Yd_observed, Yr_reference, result, 
                             title="My Structure")

# Plot comparison of multiple structures
fig = aligner.plot_comparison(Yd_observed, results, top_n=4)
```

## Advanced: Custom Structure

```python
def my_custom_structure(q_first, max_peaks=20):
    """Define your own peak pattern."""
    # Example: peaks at q * sqrt(n) for n = 1, 2, 3, ...
    indices = np.arange(1, max_peaks + 1)
    q_values = q_first * np.sqrt(indices)
    return q_values

# Use it
Yr_custom = my_custom_structure(q_first=0.015, max_peaks=15)
result = aligner.align(Yd_observed, Yr_custom)
```

## Tips for Best Results

1. **Adjust tolerance based on your data quality**
   - High-quality data: relative_tolerance=0.01-0.02
   - Medium quality: relative_tolerance=0.03-0.05
   - Noisy data: relative_tolerance=0.05-0.10

2. **Use 'auto' normalization when uncertain**
   - If first peak might be missing or wrong, use ['auto']
   - This tries multiple scale factors

3. **Check the outliers**
   - High outlier count might indicate wrong structure
   - Or need to adjust tolerance

4. **Score interpretation**
   - Higher score = better match
   - Score balances number of matches vs. error
   - Compare scores across different structures

## Available Crystal Structures

- 'fcc': Face-centered cubic
- 'bcc': Body-centered cubic  
- 'sc': Simple cubic
- 'diamond': Diamond structure

Each follows allowed Miller indices for that structure type.

## Example Output from Demo

```
Ranking of structures:
--------------------------------------------------------------------------------
1. SC           | Score:     3.16 | Matches: 4/5 | RMSE: 5.7983e-04 | Scale: 0.9393
   Outlier indices: [0]
2. Diamond      | Score:     3.13 | Matches: 4/5 | RMSE: 1.1580e-03 | Scale: 1.0382
   Outlier indices: [0]
3. BCC          | Score:     1.79 | Matches: 3/5 | RMSE: 3.0254e-04 | Scale: 0.7470
   Outlier indices: [0 1]
4. FCC          | Score:     1.79 | Matches: 3/5 | RMSE: 2.5537e-04 | Scale: 0.6404
   Outlier indices: [0 2]
```

## Questions?

The algorithm:
1. Tries multiple normalization strategies
2. For each strategy, tests different scale factors
3. Matches peaks within tolerance
4. Scores based on: num_matches / (1 + normalized_rmse)
5. Returns the best scoring alignment