from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
#from scipy.optimize import minimize
from scipy.optimize import differential_evolution, basinhopping, dual_annealing, shgo

from pyFitting.core.interfaces import IOptimizer
from pyFitting.core.types import OptimizeResult
from pyFitting.optimizers.local import LocalOptimizer, compare_optimizers

__all__ = ['GlobalOptimizer', 'HybridOptimizer', 'MultiStartOptimizer']



class GlobalOptimizer(IOptimizer):
    """
    Global optimization methods that avoid local minima.
    
    Methods:
    - 'differential_evolution': Robust, good default (recommended)
    - 'dual_annealing': Fast, good for moderate dimensions
    - 'basinhopping': Good for rough landscapes
    - 'shgo': Simplicial homology (for smooth problems)
    """
    
    def __init__(self, method: str = 'differential_evolution'):
        self.method = method
    
    def optimize(self,
                 objective: Callable,
                 x0: np.ndarray,
                 bounds: List[Tuple[float, float]],
                 **options) -> OptimizeResult:
        """Global optimization - x0 is optional for some methods."""
        
        if self.method == 'differential_evolution':
            # Population-based, very robust
            result = differential_evolution(
                objective,
                bounds,
                maxiter=options.get('maxiter', 1000),
                popsize=options.get('popsize', 15),
                atol=options.get('ftol', 1e-9),
                seed=options.get('seed', None),
                workers=options.get('workers', 1),  # parallel!
                polish=True  # local refinement at end
            )
        
        elif self.method == 'dual_annealing':
            # Simulated annealing, faster
            result = dual_annealing(
                objective,
                bounds,
                maxiter=options.get('maxiter', 1000),
                initial_temp=options.get('initial_temp', 5230.0),
                seed=options.get('seed', None)
            )
        
        elif self.method == 'basinhopping':
            # Random hopping + local minimization
            if x0 is None:
                x0 = np.array([(b[0] + b[1])/2 for b in bounds])
            
            result = basinhopping(
                objective,
                x0,
                niter=options.get('niter', 100),
                T=options.get('T', 1.0),
                stepsize=options.get('stepsize', 0.5),
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds}
            )
        
        elif self.method == 'shgo':
            # Simplicial homology
            result = shgo(
                objective,
                bounds,
                n=options.get('n', 100),
                iters=options.get('iters', 1)
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return OptimizeResult(
            x=result.x,
            fun=float(result.fun),
            success=bool(result.success),
            message=str(getattr(result, 'message', 'success')),
            nfev=int(result.nfev),
            nit=int(getattr(result, 'nit', 0))
        )


class MultiStartOptimizer(IOptimizer):
    """
    Run local optimizer from multiple random starting points.
    Simple but effective for avoiding local minima.
    """
    
    def __init__(self, method: str = 'SLSQP', n_starts: int = 20):
        self.method = method
        self.n_starts = n_starts
        self.local_opt = LocalOptimizer(method)
    
    def optimize(self,
                 objective: Callable,
                 x0: np.ndarray,
                 bounds: List[Tuple[float, float]],
                 **options) -> OptimizeResult:
        """Try multiple random starting points, return best."""
        
        best_result = None
        best_fun = np.inf
        
        # Always try user's initial guess first
        results = [self.local_opt.optimize(objective, x0, bounds, **options)]
        
        # Generate random starting points
        rng = np.random.RandomState(options.get('seed', None))
        bounds_array = np.array(bounds)
        
        for i in range(self.n_starts - 1):
            # Random point in bounds
            x_random = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
            
            try:
                result = self.local_opt.optimize(objective, x_random, bounds, **options)
                results.append(result)
                
                if result.success and result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except:
                continue
        
        # Return best result found
        if best_result is None:
            best_result = min(results, key=lambda r: r.fun)
        
        return best_result

class HybridOptimizer(IOptimizer):
    """
    Two-stage: global search + local refinement.
    Best of both worlds - finds basin, then refines precisely.
    """
    
    def __init__(self, 
                 global_method: str = 'differential_evolution',
                 local_method: str = 'SLSQP'):
        self.global_opt = GlobalOptimizer(global_method)
        self.local_opt = LocalOptimizer(local_method)
    
    def optimize(self,
                 objective: Callable,
                 x0: np.ndarray,
                 bounds: List[Tuple[float, float]],
                 **options) -> OptimizeResult:
        """Global search followed by local refinement."""
        
        # Stage 1: Global search
        global_result = self.global_opt.optimize(
            objective, x0, bounds, 
            maxiter=options.get('global_maxiter', 500)
        )
        
        # Stage 2: Local refinement from best global point
        local_result = self.local_opt.optimize(
            objective, 
            global_result.x,  # start from global optimum
            bounds,
            **options
        )
        
        return local_result                
        