"""
pyFitting.workflow.fitter - Main Fitter Class

This module provides the main Fitter class that orchestrates all components.
"""

import numpy as np
from typing import Optional, Dict, Any

from pyFitting.core.interfaces import IData, IModel, ILoss, IOptimizer, IEvaluator
from pyFitting.core.result import FitResult
from pyFitting.loss import MSELoss
from pyFitting.optimizers import LocalOptimizer,GlobalOptimizer, HybridOptimizer, MultiStartOptimizer 
from pyFitting.evaluation import StandardEvaluator


__all__ = ['Fitter']


class Fitter:
    """
    Main fitter class that orchestrates all components.
    
    This is the user-facing API that ties everything together:
    - Data
    - Model
    - Loss function
    - Optimizer
    - Evaluator
    
    Parameters:
    -----------
    data : IData
        Data to fit
    model : IModel
        Model to fit
    loss : ILoss, optional
        Loss function (default: MSELoss)
    optimizer : IOptimizer or str, optional
        LocalOptimizer: can be one of ['SLSQP', 'L-BFGS-B', 'Powell', 'TNC', 'trust-constr', 'Nelder-Mead']

    Global optimization methods that avoid local minima, can be one of ['differential_evolution',
    'dual_annealing', 'basinhopping','shgo']
    
    evaluator : IEvaluator, optional
        Evaluator for metrics (default: StandardEvaluator)
    
    Examples:
    ---------
    >>> # Simple fit
    >>> data = ArrayData(x, y)
    >>> model = GaussianModel()
    >>> fitter = Fitter(data, model)
    >>> result = fitter.fit()
    >>> result.summary()
    >>> 
    >>> # Custom components
    >>> fitter = Fitter(
    ...     data,
    ...     model,
    ...     loss=CorrelationLoss(use_log=True),
    ...     optimizer=LocalOptimizer('trust-constr')
    ... )
    >>> result = fitter.fit()
    """
    
    def __init__(self,
                 data: IData,
                 model: IModel,
                 loss: Optional[ILoss] = None,
                 optimizer: Optional[IOptimizer] = None,
                 evaluator: Optional[IEvaluator] = None,
                 optimizer_global = False, 
                ):
        """Initialize fitter with components."""
        self.data = data
        self.model = model
        self.loss = loss if loss is not None else MSELoss(use_log=False)
        if optimizer_global:
            # Handle optimizer
            if optimizer is None:
                self.optimizer = GlobalOptimizer('differential_evolution')
            elif isinstance(optimizer, str):
                self.optimizer = GlobalOptimizer(optimizer)
            else:
                self.optimizer = optimizer
        else:
            # Handle optimizer
            if optimizer is None:
                self.optimizer = LocalOptimizer('SLSQP')
            elif isinstance(optimizer, str):
                self.optimizer = LocalOptimizer(optimizer)
            else:
                self.optimizer = optimizer
        
        self.evaluator = evaluator if evaluator is not None else StandardEvaluator()
    
    def _create_objective(self):
        """
        Create objective function for optimization.
        
        Returns:
        --------
        callable : objective function that takes parameter array and returns loss
        """
        # Get data
        x = self.data.get_x()
        y_data = self.data.get_y()
        weights = self.data.get_weights()
        
        # Store for closure
        params = self.model.get_parameters()
        
        def objective(params_array: np.ndarray) -> float:
            """Objective function to minimize."""
            # Convert array to parameter dict
            params.from_array(params_array)            
            # Evaluate model
            try:
                y_model = self.model.evaluate(x, **params.values)
            except Exception as e:
                # If model evaluation fails, return large value
                return 1e12            
            # Check for invalid values
            if not np.all(np.isfinite(y_model)):
                return 1e12            
            # Compute loss
            try:
                loss = self.loss.compute(y_data, y_model, weights)
            except Exception as e:
                return 1e12            
            if not np.isfinite(loss):
                return 1e12            
            return loss        
        return objective
    
    def fit(self, 
            initial_guess: Optional[Dict[str, float]] = None,
            bounds: Optional[Dict[str, tuple]] = None,
            verbose: bool = False,
            **options) -> FitResult:
        """
        Perform fit.
        
        Parameters:
        -----------
        initial_guess : Dict[str, float], optional
            Initial parameter guess (if None, uses model's get_initial_guess)
        bounds : Dict[str, tuple], optional
            Parameter bounds (if None, uses model's bounds)
        verbose : bool
            Print progress information
        **options : dict
            Additional options passed to optimizer
        
        Returns:
        --------
        FitResult : fit results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Fitting with {self.optimizer}")
            print(f"Model: {self.model}")
            print(f"Loss: {self.loss}")
            print(f"Data: {len(self.data)} points in '{self.data.space}' space")
            print(f"{'='*70}\n")
        
        # Get initial guess
        if initial_guess is None:
            x = self.data.get_x()
            y = self.data.get_y()
            initial_guess = self.model.get_initial_guess(x, y)
            if verbose:
                print(f"Initial guess: {initial_guess}")
        
        # Set parameters and bounds
        self.model.set_parameters(**initial_guess)
        if bounds is not None:
            self.model.set_bounds(**bounds)
        
        # Get parameters
        params = self.model.get_parameters()
        x0 = params.to_array()
        bounds_array = params.get_bounds_array()
        
        if verbose:
            free_params = params.get_free_names()
            print(f"Free parameters: {free_params}")
            print(f"Bounds: {dict(zip(free_params, bounds_array))}")
            print(f"\nOptimizing...")
        
        # Create objective
        objective = self._create_objective()
        
        # Optimize
        opt_result = self.optimizer.optimize(objective, x0, bounds_array, **options)
        
        if verbose:
            status = "✓ SUCCESS" if opt_result.success else "✗ FAILED"
            print(f"\n{status}: {opt_result.message}")
            print(f"Function evaluations: {opt_result.nfev}")
            print(f"Final loss: {opt_result.fun:.6e}")
        
        # Update parameters with optimized values
        params.from_array(opt_result.x)
        
        # Compute final fit
        x = self.data.get_x()
        y_fit = self.model.evaluate(x, **params.values)
        
        # Create result
        result = FitResult(
            data=self.data,
            parameters=params.copy(),
            y_fit=y_fit,
            optimizer=str(self.optimizer),
            success=opt_result.success,
            message=opt_result.message,
            nfev=opt_result.nfev,
            nit=opt_result.nit
        )
        
        # Evaluate
        result.metrics = self.evaluator.evaluate(result)
        
        if verbose:
            print(f"\nFit quality:")
            print(f"  R² = {result.metrics['r2']:.4f}")
            print(f"  R² (log) = {result.metrics['r2_log']:.4f}")
            print(f"  χ²_red = {result.metrics['chi2_reduced']:.4f}")
            print(f"{'='*70}\n")
        
        return result
    
    def __repr__(self) -> str:
        return (f"Fitter(model={self.model.__class__.__name__}, "
                f"loss={self.loss.__class__.__name__}, "
                f"optimizer={self.optimizer.__class__.__name__})")