# pyFitting - Quick Start Guide

## 🚀 Installation

```bash
cd /mnt/user-data/outputs/pyFitting
pip install -e .
```

## 💡 3-Line Usage

```python
from pyFitting import Fitter, ArrayData, GaussianModel

data = ArrayData(x, y)
result = Fitter(data, GaussianModel()).fit()
result.summary()
```

## 📚 Common Patterns

### Pattern 1: Simple Fit
```python
from pyFitting import Fitter, ArrayData, ExponentialModel

data = ArrayData(x, y)
model = ExponentialModel()
result = Fitter(data, model).fit()
```

### Pattern 2: With Bounds
```python
model = ExponentialModel()
model.set_bounds(A=(0, 10), k=(0, 5), c=(-1, 1))
result = Fitter(data, model).fit()
```

### Pattern 3: Custom Loss
```python
from pyFitting import CorrelationLoss

loss = CorrelationLoss(use_log=True)
result = Fitter(data, model, loss=loss).fit()
```

### Pattern 4: Choose Optimizer
```python
result = Fitter(data, model, optimizer='SLSQP').fit()
result = Fitter(data, model, optimizer='trust-constr').fit()
```

### Pattern 5: Custom Model
```python
from pyFitting import BaseModel

class MyModel(BaseModel):
    def evaluate(self, x, a, b):
        return a * x**b
    
    def get_initial_guess(self, x, y):
        return {'a': 1.0, 'b': 1.0}

result = Fitter(data, MyModel()).fit()
```

## 🎯 Available Components

### Models
- `GaussianModel()` - Gaussian peak
- `ExponentialModel()` - Exponential decay
- `LinearModel()` - Linear function
- `PowerLawModel()` - Power law
- `PolynomialModel(degree=2)` - Polynomial

### Loss Functions
- `MSELoss(use_log=True)` - Mean squared error
- `Chi2Loss(use_log=False)` - Chi-squared
- `CorrelationLoss(use_log=True)` - Correlation
- `HybridLoss(alpha=0.7)` - Hybrid

### Optimizers
- `'SLSQP'` - Recommended (robust and fast)
- `'trust-constr'` - Most robust
- `'Powell'` - Derivative-free
- `'L-BFGS-B'` - Fast but less robust

## 📊 Result Methods

```python
result = fitter.fit()

# Print summary
result.summary()

# Get parameters
params = result.parameters.values
print(params['A'], params['k'])

# Get metrics
print(result.metrics['r2'])
print(result.metrics['chi2_reduced'])

# Get residuals
residuals = result.get_residuals()
rel_residuals_pct = result.get_relative_residuals_percent()

# Export
df = result.to_dataframe()
df.to_csv('results.csv')

dict_result = result.to_dict()
```

## 🔧 Advanced Usage

### Compare Optimizers
```python
from pyFitting import compare_optimizers

# (Need to extract objective function from fitter)
results = compare_optimizers(
    objective, x0, bounds,
    methods=['SLSQP', 'Powell', 'trust-constr']
)
```

### Transform Data
```python
data = ArrayData(x, y)
data_log = data.transform('log')  # Log(y)
data_loglog = data.transform('log_log')  # Log(x), Log(y)
```

### Apply Range Mask
```python
data.apply_range_mask(x_min=1.0, x_max=10.0)
```

### Fix Parameters
```python
model = GaussianModel()
model.fix_parameter('sigma', value=1.0)  # Fix sigma = 1.0
result = Fitter(data, model).fit()  # Only fit A, mu, c
```

## 🎓 Complete Example

```python
import numpy as np
from pyFitting import (
    Fitter,
    ArrayData,
    ExponentialModel,
    MSELoss,
    LocalOptimizer
)

# Generate data
x = np.linspace(0, 5, 100)
y = 3.0 * np.exp(-0.8 * x) + 0.2 + np.random.normal(0, 0.05, 100)

# Setup
data = ArrayData(x, y)
model = ExponentialModel()
model.set_bounds(A=(0, 10), k=(0, 5), c=(-1, 1))

loss = MSELoss(use_log=False)
optimizer = LocalOptimizer('SLSQP')

# Fit
fitter = Fitter(data, model, loss, optimizer)
result = fitter.fit(verbose=True)

# Results
result.summary()
print(f"R² = {result.metrics['r2']:.4f}")
print(f"Fitted: A={result.parameters.values['A']:.3f}, k={result.parameters.values['k']:.3f}")
```

## 🆘 Troubleshooting

### Import Error
```bash
# Make sure you installed the package
cd /mnt/user-data/outputs/pyFitting
pip install -e .
```

### Fit Fails
```python
# Try different optimizer
result = Fitter(data, model, optimizer='trust-constr').fit()

# Try with verbose
result = Fitter(data, model).fit(verbose=True)

# Check initial guess
guess = model.get_initial_guess(data.get_x(), data.get_y())
print(guess)
```

### Poor Fit Quality
```python
# Try different loss
from pyFitting import CorrelationLoss
loss = CorrelationLoss(use_log=True)
result = Fitter(data, model, loss=loss).fit()

# Try log space
data_log = data.transform('log')
result = Fitter(data_log, model).fit()

# Check bounds
model.set_bounds(param=(lower, upper))
```

## 📞 Next Steps

1. ✅ Install package
2. ✅ Run examples: `python examples/simple_examples.py`
3. ✅ Try with your data
4. 🔜 Phase 2: Add SAXS support!

---
 