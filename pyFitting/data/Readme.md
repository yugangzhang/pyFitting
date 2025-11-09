# Data Smoothing Toolkit

A comprehensive Python toolkit for smoothing noisy scientific data with multiple algorithms optimized for various noise types and signal characteristics.

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Multiple smoothing methods**: Gaussian, Savitzky-Golay, Spline, Moving Average, Median
- **Specialized for SAXS/WAXS**: Log-space smoothing for scattering data
- **Noise-type aware**: Optimized for different noise characteristics
- **Easy to use**: Simple API with sensible defaults
- **Well documented**: Comprehensive examples and tutorials

## Installation
```bash
# Clone the repository
 
# Install dependencies
pip install numpy scipy matplotlib
```

## Quick Start
```python
from smoothing import DataSmoother, quick_smooth
import numpy as np

# Generate noisy data
x = np.linspace(0, 10, 100)
y_noisy = np.sin(x) + 0.2 * np.random.randn(100)

# Quick smoothing with default (Savitzky-Golay)
y_smooth = quick_smooth(x, y_noisy, window_length=11, polyorder=3)

# Or use the class interface
smoother = DataSmoother('gaussian')
y_smooth = smoother.smooth(x, y_noisy, sigma=2.0)
```

## Smoothing Methods

### 1. Savitzky-Golay Filter (Recommended Default)

**Best for**: Preserving sharp features while removing noise
```python
y_smooth = quick_smooth(x, y_noisy, method='savgol',
                       window_length=11,  # Must be odd
                       polyorder=3)       # Polynomial order
```

**Pros**: Preserves peaks and dips, good for oscillatory data
**Cons**: Requires sufficient data points

### 2. Gaussian Filter

**Best for**: Uniform Gaussian noise, simple smoothing
```python
y_smooth = quick_smooth(x, y_noisy, method='gaussian',
                       sigma=2.0)  # Higher = more smoothing
```

**Pros**: Fast, simple, well-understood
**Cons**: Blurs sharp features

### 3. Median Filter

**Best for**: Removing outliers, spikes, salt-and-pepper noise
```python
y_smooth = quick_smooth(x, y_noisy, method='median',
                       window=5)  # Must be odd
```

**Pros**: Excellent for outliers, preserves edges
**Cons**: Can be slow for large datasets

### 4. Spline Smoothing

**Best for**: Smooth underlying trends, SAXS/WAXS data
```python
y_smooth = quick_smooth(x, y_noisy, method='spline',
                       smoothing=len(x) * 0.5,  # Smoothing factor
                       spline_order=3)          # Degree
```

**Pros**: Flexible, handles irregular spacing
**Cons**: Can overshoot at boundaries

### 5. Moving Average

**Best for**: Quick smoothing, real-time data
```python
y_smooth = quick_smooth(x, y_noisy, method='moving_average',
                       window=5)
```

**Pros**: Very fast, simple
**Cons**: Poor feature preservation

## Special Features for SAXS/WAXS Data

For scattering data spanning many orders of magnitude, smooth in **log space**:
```python
# SAXS data: I(q) vs q
q = np.logspace(-2, 0, 100)  # Scattering vector
I_noisy = sphere_form_factor(q) + noise

# Smooth in log-log space (RECOMMENDED)
I_smooth = quick_smooth(q, I_noisy, method='savgol',
                       log_space=True,  # ← Key parameter!
                       window_length=11, polyorder=3)

# Or use the class method
smoother = DataSmoother('savgol')
I_smooth = smoother.smooth_log_space(q, I_noisy, 
                                    window_length=11, polyorder=3)
```

**Why log space?**
- Preserves power-law behavior (Porod region)
- Better handles data spanning orders of magnitude
- More uniform noise treatment across q-range

## Examples

Run the comprehensive examples:
```bash
python smoothing_examples.py
```

This generates 5 example plots demonstrating:

1. **Basic smoothing comparison** - All methods on the same data
2. **Noise-type specific** - Best methods for different noise types
3. **SAXS data** - Log-space vs linear-space smoothing
4. **Parameter tuning** - Effects of different parameters
5. **Sharp features** - Preserving edges and steps

### Example Outputs

#### Example 1: Comparing Methods
![Basic Smoothing](docs/example1_basic_smoothing.png)

#### Example 3: SAXS Data
![SAXS Smoothing](docs/example3_saxs_smoothing.png)

## API Reference

### `DataSmoother` Class
```python
DataSmoother(method='savgol')
```

**Methods:**
- `smooth(x, y, **kwargs)` - Smooth data with selected method
- `smooth_log_space(x, y, **kwargs)` - Smooth in log-log space

**Parameters depend on method:**

| Method | Parameters | Typical Values |
|--------|-----------|----------------|
| `gaussian` | `sigma` | 1.0 - 5.0 |
| `savgol` | `window_length`, `polyorder` | 11, 3 |
| `spline` | `smoothing`, `spline_order` | len(x)*0.5, 3 |
| `moving_average` | `window` | 5 - 10 |
| `median` | `window` | 3 - 7 (odd) |

### Helper Functions
```python
# Quick smoothing
y_smooth = quick_smooth(x, y, method='savgol', log_space=False, **kwargs)

# Compare multiple methods
results = compare_smoothers(x, y, methods=['gaussian', 'savgol', 'median'])
# Returns: dict of {method: smoothed_data}
```

## Choosing the Right Method

| Your Data Has... | Use This Method | Why |
|------------------|-----------------|-----|
| Gaussian noise | Savitzky-Golay or Gaussian | Good balance or speed |
| Outliers/spikes | Median | Robust to outliers |
| Sharp edges | Median or Savitzky-Golay | Preserve edges |
| Smooth trends | Spline | Flexible fitting |
| SAXS/power-law | Spline or Savitzky-Golay (log space) | Handles wide range |
| Real-time data | Moving Average | Fastest |

## Parameter Selection Guide

### Window Size / Sigma

- **Too small**: Noise remains, undersmoothed
- **Too large**: Features lost, oversmoothed
- **Rule of thumb**: Start with 5-10% of data length

### For Noisy Data
```python
# High noise → larger window/sigma
y_smooth = quick_smooth(x, y_noisy, method='savgol',
                       window_length=21,  # Larger window
                       polyorder=3)

# OR smooth twice
y_smooth = quick_smooth(x, y_noisy, method='median', window=5)
y_smooth = quick_smooth(x, y_smooth, method='savgol', window_length=11)
```

### For Clean Data with Features
```python
# Low noise → smaller window/sigma
y_smooth = quick_smooth(x, y, method='savgol',
                       window_length=7,   # Smaller window
                       polyorder=3)
```

## Advanced Usage

### Pre-processing Pipeline for SAXS
```python
def preprocess_saxs(q, I_raw):
    """Complete pre-processing for SAXS data."""
    
    # Step 1: Remove outliers with median filter
    I_clean = quick_smooth(q, I_raw, method='median', 
                          log_space=True, window=5)
    
    # Step 2: Smooth with Savitzky-Golay in log space
    I_smooth = quick_smooth(q, I_clean, method='savgol',
                           log_space=True, 
                           window_length=11, polyorder=3)
    
    return I_smooth

# Use it
I_processed = preprocess_saxs(q, I_raw)
```

### Custom Smoothing Workflow
```python
# Create smoother instance
smoother = DataSmoother('savgol')

# Smooth multiple datasets with same parameters
for dataset in datasets:
    x, y = dataset
    y_smooth = smoother.smooth(x, y, window_length=11, polyorder=3)
    # ... process y_smooth ...
```

## Performance Tips

1. **For large datasets**: Use `gaussian` or `moving_average` (faster than `savgol`)
2. **For real-time**: Use `moving_average` with small window
3. **For best quality**: Use `savgol` or `spline` (slower but better)
4. **Vectorize**: Process multiple traces in parallel with NumPy arrays

## Common Issues and Solutions

### Issue: "Data too short for Savitzky-Golay filter"
**Solution**: Reduce `window_length` or use a different method
```python
# Auto-adjust window
n = len(y)
window = min(11, n if n % 2 == 1 else n-1)
y_smooth = quick_smooth(x, y, method='savgol', window_length=window)
```

### Issue: Smoothing removes important features
**Solution**: Reduce smoothing strength
```python
# Reduce window/sigma
y_smooth = quick_smooth(x, y, method='savgol', window_length=7)  # Instead of 15
```

### Issue: Noise still present after smoothing
**Solution**: Increase smoothing or use two-stage smoothing
```python
# Two-stage: median + savgol
y_temp = quick_smooth(x, y_noisy, method='median', window=5)
y_smooth = quick_smooth(x, y_temp, method='savgol', window_length=11)
```

## Citation

If you use this toolkit in your research, please cite:
```bibtex
@software{data_smoothing_toolkit,
  author = {Zhang, Yugang},
  title = {Data Smoothing Toolkit for Scientific Data},
  year = {2025},
  url = {https://github.com/yourusername/data-smoothing}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Savitzky-Golay filter: A. Savitzky and M. J. E. Golay, Analytical Chemistry, 1964
- Scipy for robust implementations
- SAXS community for feedback and use cases

## Contact

Yugang Zhang - Brookhaven National Laboratory

Project Link: [https://github.com/yugangzhang/data-smoothing](https://github.com/yugangzhang/pyFitting/blob/main/pyFitting/data/smoothing.py)

---

**Happy Smoothing! 🎯**