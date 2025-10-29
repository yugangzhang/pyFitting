import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def find_index(array, value):
    """Find the index of the closest value in array."""
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return idx


def get_align_score(Yd, Yr, tolerance=0.03):
    """
    Align Yd to Yr by trying ALL combinations of:
    - Which Yr peak is the reference (normalized to 1.0)
    - Which Yd peak scales to that reference
    
    Parameters:
    -----------
    Yd : array - observed peaks
    Yr : array - reference peaks
    tolerance : float - relative tolerance for valid matches
    
    Returns:
    --------
    score : dict - {(yr_ref_idx, yd_norm_idx): [num_matches, mse, valid_matches, mse_valid]}
    Align_dict : dict - raw alignment for each normalization
    Align_dict_refine : dict - refined alignment (no duplicate Yr indices)
    """
    score = {}
    Align_dict = {}
    
    # Try normalizing by each peak in Yr as reference
    for m, _ in enumerate(Yr):
        yrm = Yr / Yr[m]  # Normalize Yr so that Yr[m] = 1.0
        align_dict_yr = {}
        
        # Try scaling by each peak in Yd
        for i, ydi in enumerate(Yd):    
            Yri = yrm * ydi  # Scale normalized Yr by this Yd peak
            align_dict = {}
            
            # Match all peaks in Yd to scaled Yr
            for j, ydj in enumerate(Yd):
                ind = find_index(Yri, ydj)
                align_dict[j] = [ind, ydj, Yri[ind]]
            
            align_dict_yr[i] = align_dict
        
        Align_dict[m] = align_dict_yr 
    
    # Refine: ensure each Yr index is matched only once (keep best match)
    Align_dict_refine = {}
    
    for m in Align_dict:  # For each Yr reference peak
        Align_dict_refine[m] = {}
        
        for i in Align_dict[m]:  # For each Yd scaling peak
            Align_dict_refine[m][i] = {}
            
            for j in Align_dict[m][i]:  # For each Yd peak being matched
                yr_idx = Align_dict[m][i][j][0]  # Yr index
                
                if yr_idx not in Align_dict_refine[m][i]:
                    Align_dict_refine[m][i][yr_idx] = Align_dict[m][i][j]
                    diff = abs(Align_dict[m][i][j][1] - Align_dict[m][i][j][2])
                else:
                    diff_new = abs(Align_dict[m][i][j][1] - Align_dict[m][i][j][2])
                    if diff_new < diff:
                        Align_dict_refine[m][i][yr_idx] = Align_dict[m][i][j]
                        diff = diff_new
    
    # Compute score for each (m, i) combination
    for m in Align_dict_refine:
        for i in Align_dict_refine[m]:
            y1 = np.array([Align_dict_refine[m][i][idx][1] for idx in Align_dict_refine[m][i].keys()])
            y2 = np.array([Align_dict_refine[m][i][idx][2] for idx in Align_dict_refine[m][i].keys()])
            
            # Check which matches are within tolerance
            relative_errors = np.abs((y1 - y2) / y1)
            valid_mask = relative_errors < tolerance
            num_valid = np.sum(valid_mask)
            
            if num_valid > 0:
                mse_valid = np.mean((y1[valid_mask] - y2[valid_mask])**2)
            else:
                mse_valid = np.inf
            
            score[(m, i)] = [len(y1), np.mean((y1 - y2)**2), num_valid, mse_valid]
    
    return score, Align_dict, Align_dict_refine


def get_best_alignment(Yd, Yr, tolerance=0.03, score_weights=[1.0, 0.1]):
    """
    Find best alignment between Yd and Yr.
    
    Parameters:
    -----------
    Yd : array - observed peaks
    Yr : array - reference peaks
    tolerance : float - relative tolerance
    score_weights : list - [weight_for_num_matches, weight_for_mse]
    
    Returns:
    --------
    best_result : dict with alignment details
    """
    score, Align_dict, Align_dict_refine = get_align_score(Yd, Yr, tolerance)
    
    # Find best (m, i) combination based on: maximize valid matches, minimize MSE
    best_key = None
    best_composite_score = -np.inf
    
    for key in score:  # key is now (m, i)
        num_matches, mse_all, num_valid, mse_valid = score[key]
        
        # Composite score: prioritize valid matches, then penalize MSE
        if mse_valid < np.inf:
            composite_score = num_valid * score_weights[0] - mse_valid * score_weights[1]
        else:
            composite_score = -np.inf
        
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_key = key
    
    if best_key is None:
        return None
    
    m, i = best_key  # Unpack the best (yr_ref, yd_norm) indices
    
    # Extract alignment details for best normalization
    align_refine = Align_dict_refine[m][i]
    
    # Get scale factor: Yr[m] normalized to 1.0, then scaled by Yd[i]
    scale_factor = Yd[i] / Yr[m]
    
    # Get matched indices
    yr_indices = list(align_refine.keys())
    yd_indices = []
    yd_values = []
    yr_values = []
    residuals = []
    
    for yr_idx in yr_indices:
        # Need to find which Yd index maps to this
        for yd_idx in range(len(Yd)):
            if Align_dict[m][i][yd_idx][0] == yr_idx:
                # Check if this survived the refinement
                if Align_dict[m][i][yd_idx] == align_refine[yr_idx]:
                    yd_indices.append(yd_idx)
                    yd_values.append(Align_dict[m][i][yd_idx][1])
                    yr_values.append(Align_dict[m][i][yd_idx][2])
                    residuals.append(Align_dict[m][i][yd_idx][1] - Align_dict[m][i][yd_idx][2])
                    break
    
    yd_indices = np.array(yd_indices)
    yr_indices = np.array(yr_indices)
    yd_values = np.array(yd_values)
    yr_values = np.array(yr_values)
    residuals = np.array(residuals)
    
    # Check which matches are valid (within tolerance)
    relative_errors = np.abs(residuals / yd_values)
    valid_mask = relative_errors < tolerance
    
    # Get outliers
    yd_outliers = np.setdiff1d(np.arange(len(Yd)), yd_indices[valid_mask])
    
    # Compute metrics
    num_matches_all = len(yd_indices)
    num_matches_valid = np.sum(valid_mask)
    mse_all = np.mean(residuals**2)
    rmse_all = np.sqrt(mse_all)
    
    if num_matches_valid > 0:
        mse_valid = np.mean(residuals[valid_mask]**2)
        rmse_valid = np.sqrt(mse_valid)
    else:
        mse_valid = np.inf
        rmse_valid = np.inf
    
    result = {
        'yr_ref_idx': m,                       # Which Yr peak was reference
        'yd_norm_idx': i,                      # Which Yd peak was used for scaling
        'scale_factor': scale_factor,
        'yd_indices_all': yd_indices,
        'yr_indices_all': yr_indices,
        'yd_indices_valid': yd_indices[valid_mask],
        'yr_indices_valid': yr_indices[valid_mask],
        'yd_outliers': yd_outliers,
        'residuals': residuals,
        'valid_mask': valid_mask,
        'num_matches_all': num_matches_all,
        'num_matches_valid': num_matches_valid,
        'mse_all': mse_all,
        'rmse_all': rmse_all,
        'mse_valid': mse_valid,
        'rmse_valid': rmse_valid,
        'match_fraction': num_matches_valid / len(Yd),
        'score': best_composite_score,
        'tolerance': tolerance,
        'Yr_scaled': (Yr / Yr[m]) * Yd[i]      # The scaled reference
    }
    
    return result


def compare_structures(Yd, structures_dict, tolerance=0.03, score_weights=[1.0, 0.1]):
    """
    Compare Yd against multiple reference structures.
    
    Parameters:
    -----------
    Yd : array - observed peaks
    structures_dict : dict - {name: reference_peaks}
    tolerance : float - relative tolerance
    score_weights : list - scoring weights
    
    Returns:
    --------
    results : dict - {name: alignment_result}, sorted by score
    """
    results = {}
    
    for name, Yr in structures_dict.items():
        result = get_best_alignment(Yd, Yr, tolerance, score_weights)
        results[name] = result
    
    # Sort by score (best first)
    results = dict(sorted(results.items(), 
                         key=lambda x: x[1]['score'] if x[1] is not None else -np.inf,
                         reverse=True))
    
    return results


def plot_alignment(Yd, Yr, result, title="Peak Alignment"):
    """Visualize alignment result."""
    
    if result is None:
        print("No valid alignment found")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top plot: Original positions
    ax1 = axes[0]
    ax1.stem(Yd, np.ones_like(Yd), linefmt='b-', markerfmt='bo', 
            basefmt='gray', label=f'Observed peaks (Yd) N={len(Yd)}')
    markerline, stemlines, baseline = ax1.stem(Yr, np.ones_like(Yr) * 0.5, 
            linefmt='r-', markerfmt='rs', basefmt='gray', 
            label=f'Reference peaks (Yr) N={len(Yr)}')
    plt.setp(markerline, alpha=0.5)
    plt.setp(stemlines, alpha=0.5)
    ax1.set_ylabel('Intensity (arb.)')
    ax1.set_title(f'{title} - Original Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Aligned positions
    ax2 = axes[1]
    
    Yr_scaled = result['Yr_scaled']
    
    # Plot all reference peaks (faint)
    markerline, stemlines, baseline = ax2.stem(Yr_scaled, np.ones_like(Yr_scaled) * 0.3,
            linefmt='gray', markerfmt='s', basefmt='gray', label='Reference (scaled)')
    plt.setp(markerline, alpha=0.2)
    plt.setp(stemlines, alpha=0.2)
    
    # Plot all matched peaks (including invalid)
    if len(result['yd_indices_all']) > 0:
        yd_matched_all = Yd[result['yd_indices_all']]
        yr_matched_all = Yr_scaled[result['yr_indices_all']]
        
        # Valid matches in green
        valid_mask = result['valid_mask']
        if np.sum(valid_mask) > 0:
            ax2.stem(yd_matched_all[valid_mask], np.ones(np.sum(valid_mask)),
                    linefmt='g-', markerfmt='go', basefmt='gray',
                    label=f'Valid matches ({np.sum(valid_mask)})')
            
            # Draw connection lines for valid matches
            for i in np.where(valid_mask)[0]:
                ax2.plot([yd_matched_all[i], yr_matched_all[i]], [1.0, 0.3],
                        'g--', alpha=0.4, linewidth=1)
        
        # Invalid matches in orange
        if np.sum(~valid_mask) > 0:
            ax2.stem(yd_matched_all[~valid_mask], np.ones(np.sum(~valid_mask)),
                    linefmt='orange', markerfmt='o', basefmt='gray',
                    label=f'Out of tolerance ({np.sum(~valid_mask)})')
    
    # Plot outliers (peaks in Yd that didn't match anything)
    if len(result['yd_outliers']) > 0:
        yd_outlier_vals = Yd[result['yd_outliers']]
        markerline, stemlines, baseline = ax2.stem(yd_outlier_vals, 
                np.ones_like(yd_outlier_vals),
                linefmt='r-', markerfmt='rx', basefmt='gray',
                label=f'Outliers ({len(result["yd_outliers"])})')
        plt.setp(markerline, markersize=10, markeredgewidth=2)
    
    ax2.set_xlabel('Peak Position (q-value)')
    ax2.set_ylabel('Intensity (arb.)')
    title_str = (f'Aligned: Yr[{result["yr_ref_idx"]}] as ref, scaled by Yd[{result["yd_norm_idx"]}]\n'
                f'Scale factor={result["scale_factor"]:.5f}, '
                f'Valid: {result["num_matches_valid"]}/{len(Yd)}, '
                f'RMSE={result["rmse_valid"]:.4e}, '
                f'tolerance={result["tolerance"]*100:.1f}%, '
                f'score={result["score"]:.2f}')
    ax2.set_title(title_str)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison(Yd, results, top_n=4):
    """Plot comparison of top N structure matches."""
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    names = list(valid_results.keys())[:top_n]
    
    if len(names) == 0:
        print("No valid results to plot")
        return None
    
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3*len(names)))
    
    if len(names) == 1:
        axes = [axes]
    
    for idx, name in enumerate(names):
        result = results[name]
        ax = axes[idx]
        
        Yr_scaled = result['Yr_scaled']
        
        # Plot reference (faint)
        markerline, stemlines, baseline = ax.stem(Yr_scaled, np.ones_like(Yr_scaled) * 0.5,
                linefmt='gray', markerfmt='s', basefmt='gray')
        plt.setp(markerline, alpha=0.2)
        plt.setp(stemlines, alpha=0.2)
        
        # Plot valid matches
        if len(result['yd_indices_valid']) > 0:
            yd_valid = Yd[result['yd_indices_valid']]
            ax.stem(yd_valid, np.ones_like(yd_valid),
                   linefmt='g-', markerfmt='go', basefmt='gray',
                   label=f'Valid ({len(result["yd_indices_valid"])})')
        
        # Plot invalid/outliers
        invalid_and_outliers = np.concatenate([
            result['yd_indices_all'][~result['valid_mask']],
            result['yd_outliers']
        ])
        if len(invalid_and_outliers) > 0:
            yd_invalid = Yd[invalid_and_outliers]
            markerline, stemlines, baseline = ax.stem(yd_invalid, np.ones_like(yd_invalid),
                   linefmt='r-', markerfmt='rx', basefmt='gray',
                   label=f'Invalid ({len(invalid_and_outliers)})')
            plt.setp(markerline, markersize=8, markeredgewidth=2)
        
        title_str = (f'{idx+1}. {name}: score={result["score"]:.6f}, '
                    f'valid={result["num_matches_valid"]}/{len(Yd)}, '
                    f'RMSE={result["rmse_valid"]:.4e}')
        ax.set_title(title_str)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Peak Position (q-value)')
    plt.tight_layout()
    return fig


def generate_theoretical_peaks(structure_type, normalized=True, q_first=1.0, max_peaks=20):
    """
    Generate theoretical peak positions for crystal structures.
    
    Parameters:
    -----------
    structure_type : str - 'fcc', 'bcc', 'sc', 'diamond', 'hcp'
    normalized : bool - if True, normalize to first peak = 1.0 (for alignment algorithm)
                       if False, use absolute q-values with given q_first
    q_first : float - position of first peak (used when normalized=False)
    max_peaks : int - number of peaks to generate
    """
    
    allowed_reflections = {
        'fcc': [3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32, 35, 36, 40, 43, 44, 48, 51, 56, 59],
        'bcc': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
        'sc': [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22],
        'diamond': [3, 8, 11, 16, 19, 24, 27, 32, 35, 40, 43, 48, 51, 56, 59, 64, 67, 72],
        'hcp': [1, 3, 4, 7, 8, 9, 11, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36, 37, 39, 43],
    }
    
    if structure_type not in allowed_reflections:
        raise ValueError(f"Unknown structure: {structure_type}")
    
    hkl_squared = allowed_reflections[structure_type][:max_peaks]
    
    if normalized:
        # Normalize to first peak = 1.0
        q_values = np.sqrt(np.array(hkl_squared) / hkl_squared[0])
    else:
        # Absolute q-values
        q_values = q_first * np.sqrt(np.array(hkl_squared) / hkl_squared[0])
    
    return q_values


# ============================================================================
# MAIN DEMO
# ============================================================================
test = False
#if __name__ == "__main__":
if test:    
    # Your SAXS data
    Yd_observed = np.array([0.01517034, 0.02737475, 0.04016032, 0.05352705, 0.06602204])
    
    print("="*80)
    print("SAXS Peak Alignment - Improved Algorithm")
    print("="*80)
    print(f"\nObserved peaks (Yd): {Yd_observed}")
    print(f"Number of peaks: {len(Yd_observed)}")
    
    # Generate reference structures (normalized to 1.0 for alignment algorithm)
    structures = {}
    structures['FCC'] = generate_theoretical_peaks('fcc', normalized=True, max_peaks=20)
    structures['BCC'] = generate_theoretical_peaks('bcc', normalized=True, max_peaks=20)
    structures['SC'] = generate_theoretical_peaks('sc', normalized=True, max_peaks=20)
    structures['Diamond'] = generate_theoretical_peaks('diamond', normalized=True, max_peaks=20)
    structures['HCP'] = generate_theoretical_peaks('hcp', normalized=True, max_peaks=20)
    
    print("\nReference structures (normalized, first 8 peaks):")
    for name, peaks in structures.items():
        print(f"  {name:8s}: {peaks[:8]}")
    
    print("\n" + "="*80)
    print("Comparing against reference structures...")
    print("="*80)
    
    # Compare with tolerance of 3%
    results = compare_structures(Yd_observed, structures, tolerance=0.03, score_weights=[1.0, 1000])
    
    # Print ranking
    print("\nRanking of structures:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Structure':<12} {'Score':>8} {'Valid':>7} {'All':>7} "
          f"{'RMSE':>12} {'Yr_ref':>8} {'Yd_norm':>8} {'Scale':>10}")
    print("-" * 100)
    
    for rank, (name, result) in enumerate(results.items(), 1):
        if result is not None:
            print(f"{rank:<6} {name:<12} {result['score']:>8.2f} "
                  f"{result['num_matches_valid']:>3}/{len(Yd_observed):<3} "
                  f"{result['num_matches_all']:>3}/{len(Yd_observed):<3} "
                  f"{result['rmse_valid']:>12.4e} "
                  f"Yr[{result['yr_ref_idx']:<3}] "
                  f"Yd[{result['yd_norm_idx']:<3}] "
                  f"{result['scale_factor']:>10.6f}")
            
            if len(result['yd_outliers']) > 0:
                print(f"       └─ Outliers at indices: {result['yd_outliers']}")
    
    # Get best match
    best_name = list(results.keys())[0]
    best_result = results[best_name]
    
    print("\n" + "="*80)
    print(f"BEST MATCH: {best_name}")
    print("="*80)
    print(f"Yr reference: Yr[{best_result['yr_ref_idx']}] normalized to 1.0")
    print(f"Yd scaling: Scaled by Yd[{best_result['yd_norm_idx']}] "
          f"(q = {Yd_observed[best_result['yd_norm_idx']]:.6f})")
    print(f"Scale factor: {best_result['scale_factor']:.6f}")
    print(f"Valid matches: {best_result['num_matches_valid']}/{len(Yd_observed)}")
    print(f"RMSE (valid): {best_result['rmse_valid']:.6e}")
    print(f"Tolerance: {best_result['tolerance']*100:.1f}%")
    print(f"\nMatched Yd indices: {best_result['yd_indices_valid']}")
    print(f"Matched Yr indices: {best_result['yr_indices_valid']}")
    if len(best_result['yd_outliers']) > 0:
        print(f"Outlier Yd indices: {best_result['yd_outliers']}")
    
    # Visualization
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    fig1 = plot_alignment(Yd_observed, structures[best_name], best_result,
                         title=f"Best Match: {best_name}")
    plt.savefig('/mnt/user-data/outputs/improved_alignment_best.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: improved_alignment_best.png")
    
    fig2 = plot_comparison(Yd_observed, results, top_n=5)
    plt.savefig('/mnt/user-data/outputs/improved_alignment_comparison.png',
                dpi=150, bbox_inches='tight')
    print("✓ Saved: improved_alignment_comparison.png")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80)
    
    plt.show()