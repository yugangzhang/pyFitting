import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class PeakAligner:
    """
    Aligns 1D peak positions (e.g., SAXS dips) to reference patterns.
    Handles missing peaks, outliers, and normalization uncertainty.
    """
    
    def __init__(self, 
                 relative_tolerance: float = 0.03,  # 3% relative tolerance
                 absolute_tolerance: Optional[float] = None,
                 min_matches: int = 2,
                 normalization_modes: List[str] = ['first', 'auto']):
        """
        Parameters:
        -----------
        relative_tolerance : float
            Relative tolerance for matching (e.g., 0.03 = 3%)
        absolute_tolerance : float or None
            Absolute tolerance, if None uses relative only
        min_matches : int
            Minimum number of matches required
        normalization_modes : list
            Which normalization strategies to try ['first', 'auto', 'none']
        """
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.min_matches = min_matches
        self.normalization_modes = normalization_modes
    
    def align(self, Yd: np.ndarray, Yr: np.ndarray) -> Dict:
        """
        Find best alignment between data peaks Yd and reference peaks Yr.
        
        Parameters:
        -----------
        Yd : array (Nd,) - observed peak positions
        Yr : array (Nr,) - reference peak positions (Nr >= Nd typically)
        
        Returns:
        --------
        result : dict with keys:
            - 'scale_factor': best normalization factor
            - 'yd_matched': indices in Yd that matched
            - 'yr_matched': corresponding indices in Yr
            - 'yd_outliers': indices in Yd that didn't match
            - 'yr_missing': indices in Yr that weren't observed
            - 'residuals': residuals for matched peaks
            - 'mse': mean squared error
            - 'rmse': root mean squared error
            - 'num_matches': number of matched peaks
            - 'match_fraction': fraction of Yd peaks matched
            - 'score': overall quality score
            - 'normalization_mode': which normalization was used
        """
        Yd = np.asarray(Yd).flatten()
        Yr = np.asarray(Yr).flatten()
        
        best_result = None
        best_score = -np.inf
        
        # Try different normalization strategies
        for norm_mode in self.normalization_modes:
            result = self._align_with_normalization(Yd, Yr, norm_mode)
            
            if result is not None and result['score'] > best_score:
                best_score = result['score']
                best_result = result
        
        return best_result
    
    def _align_with_normalization(self, Yd: np.ndarray, Yr: np.ndarray, 
                                   norm_mode: str) -> Optional[Dict]:
        """Try alignment with a specific normalization strategy."""
        
        if norm_mode == 'first':
            # Normalize by first peak
            if Yd[0] == 0 or Yr[0] == 0:
                return None
            scale_factors = [Yr[0] / Yd[0]]
            
        elif norm_mode == 'auto':
            # Try multiple scale factors based on different peak pairs
            scale_factors = []
            for i in range(min(3, len(Yd))):  # Try first 3 peaks
                for j in range(min(5, len(Yr))):  # Against first 5 ref peaks
                    if Yd[i] > 0:
                        scale_factors.append(Yr[j] / Yd[i])
            
            # Add some variations around each scale factor
            expanded_factors = []
            for sf in scale_factors:
                expanded_factors.extend([sf * 0.95, sf, sf * 1.05])
            scale_factors = expanded_factors
            
        elif norm_mode == 'none':
            scale_factors = [1.0]
        else:
            raise ValueError(f"Unknown normalization mode: {norm_mode}")
        
        # Try each scale factor
        best_match = None
        best_score = -np.inf
        
        for scale_factor in scale_factors:
            match = self._match_peaks(Yd, Yr, scale_factor)
            
            if match['num_matches'] >= self.min_matches:
                score = self._compute_score(match)
                
                if score > best_score:
                    best_score = score
                    match['scale_factor'] = scale_factor
                    match['normalization_mode'] = norm_mode
                    match['score'] = score
                    best_match = match
        
        return best_match
    
    def _match_peaks(self, Yd: np.ndarray, Yr: np.ndarray, 
                     scale_factor: float) -> Dict:
        """Match peaks between Yd and Yr using given scale factor."""
        
        # Scale Yd to match Yr
        Yd_scaled = Yd * scale_factor
        
        yd_matched = []
        yr_matched = []
        residuals = []
        
        # For each peak in Yd, find closest peak in Yr
        for i, yd_val in enumerate(Yd_scaled):
            distances = np.abs(Yr - yd_val)
            min_dist = np.min(distances)
            closest_idx = np.argmin(distances)
            
            # Check if within tolerance
            tolerance = self.relative_tolerance * yd_val
            if self.absolute_tolerance is not None:
                tolerance = max(tolerance, self.absolute_tolerance)
            
            if min_dist <= tolerance:
                # Check if this Yr peak was already matched
                if closest_idx not in yr_matched:
                    yd_matched.append(i)
                    yr_matched.append(closest_idx)
                    residuals.append(yd_val - Yr[closest_idx])
        
        yd_matched = np.array(yd_matched)
        yr_matched = np.array(yr_matched)
        residuals = np.array(residuals)
        
        # Identify outliers and missing peaks
        yd_outliers = np.setdiff1d(np.arange(len(Yd)), yd_matched)
        yr_missing = np.setdiff1d(np.arange(len(Yr)), yr_matched)
        
        # Compute error metrics
        if len(residuals) > 0:
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            # Normalized RMSE (relative to peak positions)
            nrmse = rmse / np.mean(Yd_scaled[yd_matched]) if len(yd_matched) > 0 else np.inf
        else:
            mse = np.inf
            rmse = np.inf
            nrmse = np.inf
        
        return {
            'yd_matched': yd_matched,
            'yr_matched': yr_matched,
            'yd_outliers': yd_outliers,
            'yr_missing': yr_missing,
            'residuals': residuals,
            'mse': mse,
            'rmse': rmse,
            'nrmse': nrmse,
            'num_matches': len(yd_matched),
            'match_fraction': len(yd_matched) / len(Yd) if len(Yd) > 0 else 0
        }
    
    def _compute_score(self, match: Dict) -> float:
        """
        Compute quality score for a match.
        Higher is better. Balances number of matches and alignment quality.
        """
        num_matches = match['num_matches']
        match_fraction = match['match_fraction']
        nrmse = match['nrmse']
        
        # Score formula: prioritize matches but penalize error
        # Using normalized RMSE to make it scale-independent
        if nrmse == 0:
            score = num_matches * 1000  # Perfect match
        else:
            score = (num_matches * match_fraction) / (1 + nrmse)
        
        return score
    
    def compare_structures(self, Yd: np.ndarray, 
                          reference_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Compare observed peaks against multiple reference structures.
        
        Parameters:
        -----------
        Yd : array - observed peak positions
        reference_dict : dict - {structure_name: reference_peaks}
        
        Returns:
        --------
        results : dict - {structure_name: alignment_result}
        """
        results = {}
        
        for structure_name, Yr in reference_dict.items():
            result = self.align(Yd, Yr)
            results[structure_name] = result
        
        # Sort by score
        results = dict(sorted(results.items(), 
                            key=lambda x: x[1]['score'] if x[1] is not None else -np.inf, 
                            reverse=True))
        
        return results
    
    def plot_alignment(self, Yd: np.ndarray, Yr: np.ndarray, 
                       result: Dict, title: str = "Peak Alignment"):
        """Visualize the alignment result."""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top plot: Original positions
        ax1 = axes[0]
        ax1.stem(Yd, np.ones_like(Yd), linefmt='b-', markerfmt='bo', 
                basefmt='gray', label='Observed peaks (Yd)')
        markerline, stemlines, baseline = ax1.stem(Yr, np.ones_like(Yr) * 0.5, linefmt='r-', markerfmt='rs', 
                basefmt='gray', label='Reference peaks (Yr)')
        plt.setp(markerline, alpha=0.6)
        plt.setp(stemlines, alpha=0.6)
        ax1.set_ylabel('Intensity (arb.)')
        ax1.set_title(f'{title} - Original Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Aligned positions
        ax2 = axes[1]
        
        Yd_scaled = Yd * result['scale_factor']
        
        # Plot reference peaks
        markerline, stemlines, baseline = ax2.stem(Yr, np.ones_like(Yr) * 0.5, linefmt='gray', markerfmt='s', 
                basefmt='gray', label='Reference')
        plt.setp(markerline, alpha=0.3)
        plt.setp(stemlines, alpha=0.3)
        
        # Plot matched peaks
        if len(result['yd_matched']) > 0:
            yd_matched_vals = Yd_scaled[result['yd_matched']]
            ax2.stem(yd_matched_vals, np.ones_like(yd_matched_vals), 
                    linefmt='g-', markerfmt='go', basefmt='gray',
                    label=f'Matched ({len(result["yd_matched"])})')
            
            # Draw connecting lines to show residuals
            for i, yd_idx in enumerate(result['yd_matched']):
                yr_idx = result['yr_matched'][i]
                ax2.plot([Yd_scaled[yd_idx], Yr[yr_idx]], [1.0, 0.5], 
                        'g--', alpha=0.3, linewidth=1)
        
        # Plot outliers
        if len(result['yd_outliers']) > 0:
            yd_outlier_vals = Yd_scaled[result['yd_outliers']]
            markerline, stemlines, baseline = ax2.stem(yd_outlier_vals, np.ones_like(yd_outlier_vals), 
                    linefmt='r-', markerfmt='rx', basefmt='gray',
                    label=f'Outliers ({len(result["yd_outliers"])})')
            plt.setp(markerline, markersize=8, markeredgewidth=2)
        
        ax2.set_xlabel('Position (q or index)')
        ax2.set_ylabel('Intensity (arb.)')
        ax2.set_title(f'Aligned (scale={result["scale_factor"]:.4f}, '
                     f'matches={result["num_matches"]}/{len(Yd)}, '
                     f'RMSE={result["rmse"]:.4e}, '
                     f'score={result["score"]:.2f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, Yd: np.ndarray, results: Dict[str, Dict], 
                       top_n: int = 5):
        """Plot comparison across multiple structures."""
        
        # Filter out None results and get top N
        valid_results = {k: v for k, v in results.items() if v is not None}
        sorted_names = list(valid_results.keys())[:top_n]
        
        fig, axes = plt.subplots(len(sorted_names), 1, 
                                figsize=(12, 3*len(sorted_names)))
        
        if len(sorted_names) == 1:
            axes = [axes]
        
        for idx, structure_name in enumerate(sorted_names):
            result = results[structure_name]
            ax = axes[idx]
            
            # Get reference from result (need to pass it separately in practice)
            Yd_scaled = Yd * result['scale_factor']
            
            # Plot matched peaks
            if len(result['yd_matched']) > 0:
                yd_matched_vals = Yd_scaled[result['yd_matched']]
                ax.stem(yd_matched_vals, np.ones_like(yd_matched_vals), 
                       linefmt='g-', markerfmt='go', basefmt='gray',
                       label=f'Matched ({len(result["yd_matched"])})')
            
            # Plot outliers
            if len(result['yd_outliers']) > 0:
                yd_outlier_vals = Yd_scaled[result['yd_outliers']]
                ax.stem(yd_outlier_vals, np.ones_like(yd_outlier_vals), 
                       linefmt='r-', markerfmt='rx', basefmt='gray',
                       label=f'Outliers ({len(result["yd_outliers"])})')
            
            ax.set_title(f'{structure_name}: score={result["score"]:.2f}, '
                        f'matches={result["num_matches"]}/{len(Yd)}, '
                        f'RMSE={result["rmse"]:.4e}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Position (q)')
        plt.tight_layout()
        return fig

