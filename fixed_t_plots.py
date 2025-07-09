import zipfile
import os
import pandas as pd
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

#tprime = t-tmin (whatever that means)

# CONFIGURATION
#Path to main zip file containing the data
top_zip_path = r"C:\Users\Sweetie Pie\Desktop\CUA REU\meson-strcutrue-2025-06-05-csv-20250623T143248Z-1-001.zip"
extract_root = 'meson_temp_extract'

# Analysis configuration
SAVE_PLOTS = False
SHOW_PLOTS = True
DEVIATION_THRESHOLDS = [15, 10, 5]  # Percentage thresholds for acceptance
T_BINS = 40  # Number of bins for -t axis

# PARTICLE MASSES (GeV)
M_LAMBDA = 1.115683  # Lambda mass
M_PROTON = 0.938272  # Proton mass (for target)
M_ELECTRON = 0.000511  # Electron mass

# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

def parse_filename(filename):
    #Parse filename to extract metadata
    name = filename.replace('.csv.zip', '')
    parts = name.split('.')
    
    if len(parts) != 2:
        return None, None, None, None
    
    base_name, file_type = parts
    base_parts = base_name.split('_')
    
    if len(base_parts) != 5:
        return None, None, None, None
    
    # k_lambda_5x41_5000evt_001 -> kinematics=5x41, event_number=001
    process_name = '_'.join(base_parts[:2])  # k_lambda
    kinematics = base_parts[2]  # 5x41, 10x100, or 18x275
    event_number = base_parts[4]  # 001-200
    
    #Map file types
    if file_type == 'mc_dis':
        data_type = 'mc_dis'
    elif file_type == 'mcpart_lambda':
        data_type = 'mc_part'
    elif file_type == 'reco_dis':  
        data_type = 'recon'
    else:
        data_type = None
    
    return process_name, kinematics, event_number, data_type

def load_csv_from_zip(zip_path):
    #Load CSV from zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = next((name for name in z.namelist() if name.endswith('.csv')), None)
            if csv_name is None:
                return None
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                z.extract(csv_name, tmpdirname)
                csv_full_path = os.path.join(tmpdirname, csv_name)
                return pd.read_csv(csv_full_path)
                
    except Exception as e:
        print(f"Error loading {zip_path}: {e}")
        return None

def load_and_process_data():
    """Load and process all the data files"""
    print("Loading data...")
    
    # Extract main zip file
    with zipfile.ZipFile(top_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_root)
    
    inner_zip_folder = os.path.join(extract_root, 'meson-strcutrue-2025-06-05-csv')
    inner_zip_files = [os.path.join(inner_zip_folder, f) 
                      for f in os.listdir(inner_zip_folder) if f.endswith('.zip')]
    
    print(f"Found {len(inner_zip_files)} zip files")
    
    
    # Organize data by kinematics and event_number
    data_inventory = defaultdict(lambda: defaultdict(dict))
    file_type_counts = defaultdict(int)
    
    # Process files
    for zip_path in inner_zip_files:
        zip_filename = os.path.basename(zip_path)
        process_name, kinematics, event_number, data_type = parse_filename(zip_filename)
        
        if data_type:
            file_type_counts[data_type] += 1
        else:
            file_type_counts['unparsed'] += 1
            
        if not all([process_name, kinematics, event_number, data_type]):
            continue
        
        df = load_csv_from_zip(zip_path)
        if df is not None:
            data_inventory[kinematics][event_number][data_type] = df
    
    print(f"\nFile type counts: {dict(file_type_counts)}")
    print(f"Kinematics found: {list(data_inventory.keys())}")
    
    # Debug: Show data inventory structure
    for kinematics in data_inventory:
        print(f"\nKinematics {kinematics}:")
        event_count = len(data_inventory[kinematics])
        print(f"  Total events: {event_count}")
        
        # Sample a few events to see their structure
        sample_events = list(data_inventory[kinematics].keys())[:3]
        for event in sample_events:
            file_types = list(data_inventory[kinematics][event].keys())
            print(f"    Event {event}: {file_types}")
    
    # Combine data ensuring proper three-way alignment
    print("\nCombining data with proper event alignment (MC_DIS + MC_PART + RECON)...")
    combined_data = {}
    
    for kinematics in data_inventory:
        print(f"Processing kinematics: {kinematics}")
        
        # Find events that have all three file types
        complete_events = []
        for event_number in data_inventory[kinematics]:
            event_data = data_inventory[kinematics][event_number]
            has_all = all(data_type in event_data for data_type in ['mc_dis', 'mc_part', 'recon'])
            if has_all:
                complete_events.append(event_number)
        
        print(f"  Events with all three file types: {len(complete_events)}")
        
        if len(complete_events) == 0:
            print(f"  WARNING: No complete event sets found for {kinematics}")
            # Debug: Show what file types we do have
            all_file_types = set()
            for event_number in data_inventory[kinematics]:
                all_file_types.update(data_inventory[kinematics][event_number].keys())
            print(f"  Available file types in this kinematics: {sorted(all_file_types)}")
            continue
        
        # Process complete events
        all_data = {'mc_dis': [], 'mc_part': [], 'recon': []}
        
        for event_number in sorted(complete_events):
            event_data = data_inventory[kinematics][event_number]
            
            for data_type in ['mc_dis', 'mc_part', 'recon']:
                df_copy = event_data[data_type].copy()
                df_copy['event_file'] = event_number
                df_copy['row_index'] = range(len(df_copy))
                all_data[data_type].append(df_copy)
        
        # Concatenate and merge
        mc_dis_combined = pd.concat(all_data['mc_dis'], ignore_index=True)
        mc_part_combined = pd.concat(all_data['mc_part'], ignore_index=True)
        recon_combined = pd.concat(all_data['recon'], ignore_index=True)
        
        # Create alignment keys and merge
        for df in [mc_dis_combined, mc_part_combined, recon_combined]:
            df['alignment_key'] = df['event_file'].astype(str) + '_' + df['row_index'].astype(str)
        
        # Merge all three datasets
        mc_merged = pd.merge(mc_dis_combined, mc_part_combined, on='alignment_key', suffixes=('_dis', '_part'))
        full_merged = pd.merge(mc_merged, recon_combined, on='alignment_key', suffixes=('', '_recon'))
        
        # Clean up redundant columns
        cols_to_drop = [col for col in full_merged.columns 
                       if col.endswith(('_part', '_recon')) and col.startswith(('event_file', 'row_index'))]
        full_merged = full_merged.drop(cols_to_drop, axis=1)
        
        # Rename remaining columns for clarity
        full_merged = full_merged.rename(columns={
            'event_file_dis': 'event_file',
            'row_index_dis': 'row_index'
        })
        
        if len(full_merged) > 0:
            combined_data[kinematics] = {'merged': full_merged}
            print(f"  Successfully aligned {len(full_merged)} events across all three file types")
            
            # Check for lambda momentum columns
            lambda_cols = ['lam_px', 'lam_py', 'lam_pz']
            lambda_cols_found = [col for col in lambda_cols if col in full_merged.columns]
            print(f"    ✓ Lambda momentum columns found: {len(lambda_cols_found)}/{len(lambda_cols)}")
            
            # Check for tprime column
            if 'tprime' in full_merged.columns:
                print(f"    ✓ tprime column found with {(~pd.isna(full_merged['tprime'])).sum()} valid values")
            else:
                print(f"    ⚠️  tprime column not found")
    
    print(f"Found properly aligned data for kinematics: {list(combined_data.keys())}")
    return combined_data

# =============================================================================
# ACCEPTANCE ANALYSIS FUNCTIONS
# =============================================================================

def get_beam_energy_from_kinematics(kinematics):
    """Extract beam energy from kinematics name"""
    if kinematics.startswith('10x'):
        return 10.0
    elif kinematics.startswith('18x'):
        return 18.0
    elif kinematics.startswith('5x'):
        return 5.0
    else:
        print(f"Warning: Unknown kinematics {kinematics}, defaulting to 5.0 GeV")
        return 5.0

def calculate_virtual_photon_4momentum(Q2, nu, beam_energy):
    """
    Calculate virtual photon 4-momentum from Q2 and nu
    
    Parameters:
    Q2 : float or array
        Virtual photon Q² (positive)
    nu : float or array  
        Energy transfer ν
    beam_energy : float
        Beam energy in GeV
        
    Returns:
    tuple : (E_gamma, px, py, pz) - virtual photon 4-momentum
    """
    # Virtual photon energy
    E_gamma = nu
    
    # Virtual photon momentum magnitude
    # For virtual photon: q² = E² - |q|² = -Q² (timelike)
    # So |q|² = E² + Q² = ν² + Q²
    q_mag = np.sqrt(nu**2 + Q2)
    
    # Assume beam along z-axis, scattered electron in x-z plane
    # Virtual photon momentum is approximately in -z direction for small angles
    # For more precise calculation, would need scattered electron angles
    px_gamma = 0.0  # Simplified assumption
    py_gamma = 0.0  # Simplified assumption  
    pz_gamma = -q_mag  # Momentum transfer is roughly along beam direction
    
    return E_gamma, px_gamma, py_gamma, pz_gamma

def calculate_t_from_lambda_4vectors(data, mask, kinematics):
    """
    Calculate momentum transfer squared t from lambda 4-vectors and virtual photon kinematics.
    
    t = (q - p_lambda)² where q is virtual photon and p_lambda is lambda 4-momentum
    
    Parameters:
    data : DataFrame
        Merged data containing MC_DIS and MC_PART data
    mask : array-like
        Boolean mask for valid events
    kinematics : str
        Kinematics identifier to get beam energy
        
    Returns:
    array-like
        -t values calculated from 4-vectors (absolute values)
    """
    # Get beam energy
    beam_energy = get_beam_energy_from_kinematics(kinematics)
    
    #Check required columns
    required_lambda_cols = ['lam_px', 'lam_py', 'lam_pz']
    required_dis_cols = ['q2', 'nu']
    
    missing_lambda = [col for col in required_lambda_cols if col not in data.columns]
    missing_dis = [col for col in required_dis_cols if col not in data.columns]
    
    if missing_lambda:
        raise ValueError(f"Missing lambda momentum columns: {missing_lambda}")
    if missing_dis:
        raise ValueError(f"Missing DIS columns: {missing_dis}")
    
    # Extract data for masked events
    n_events = mask.sum()
    print(f"Calculating t for {n_events} events")
    
    # Lambda 4-momentum components
    lam_px = data['lam_px'].values[mask]
    lam_py = data['lam_py'].values[mask]  
    lam_pz = data['lam_pz'].values[mask]
    
    # Calculate lambda energy
    lam_E = np.sqrt(lam_px**2 + lam_py**2 + lam_pz**2 + M_LAMBDA**2)
    
    # Virtual photon kinematics
    Q2 = data['q2'].values[mask]
    nu = data['nu'].values[mask]
    
    # Calculate virtual photon 4-momentum
    gamma_E, gamma_px, gamma_py, gamma_pz = calculate_virtual_photon_4momentum(Q2, nu, beam_energy)
    
    # Calculate t = (q - p_lambda)²
    # t = (E_gamma - E_lambda)² - (px_gamma - px_lambda)² - (py_gamma - py_lambda)² - (pz_gamma - pz_lambda)²
    delta_E = gamma_E - lam_E
    delta_px = gamma_px - lam_px  # gamma_px = 0 in our approximation
    delta_py = gamma_py - lam_py  # gamma_py = 0 in our approximation  
    delta_pz = gamma_pz - lam_pz
    
    t = delta_E**2 - delta_px**2 - delta_py**2 - delta_pz**2
    
    # Return -t (positive values)
    minus_t = -t
    
    # Filter out unphysical values
    valid_t = np.isfinite(minus_t) & (minus_t > 0)
    print(f"Valid -t values: {valid_t.sum()}/{len(minus_t)}")
    
    if valid_t.sum() > 0:
        print(f"-t range: {minus_t[valid_t].min():.6f} to {minus_t[valid_t].max():.6f} GeV²")
        print(f"-t mean: {minus_t[valid_t].mean():.6f} GeV²")
    
    return minus_t

def calculate_t_from_kinematics(data, mask, kinematics=None):
    """
    Calculate -t using both methods: from lambda 4-vectors and from tprime column.
    Compare the results for validation.
    
    Parameters:
    data : DataFrame
        Merged data containing MC_DIS, MC_PART, and reconstruction data
    mask : array-like
        Boolean mask for valid events
    kinematics : str
        Kinematics identifier
        
    Returns:
    array-like
        -t values calculated from lambda 4-vectors
    """
    print(f"\nCalculating -t for {kinematics} using lambda 4-vectors...")
    
    # Method 1: Calculate from lambda 4-vectors
    try:
        t_from_4vectors = calculate_t_from_lambda_4vectors(data, mask, kinematics)
    except ValueError as e:
        print(f"Error calculating t from 4-vectors: {e}")
        # Fallback to tprime method
        if 'tprime' in data.columns:
            print("Falling back to tprime column...")
            tprime_values = data['tprime'].values[mask]
            return np.abs(tprime_values)
        else:
            raise e
    
    # Method 2: Compare with tprime if available
    if 'tprime' in data.columns:
        tprime_values = np.abs(data['tprime'].values[mask])
        
        # Compare the two methods
        valid_both = np.isfinite(t_from_4vectors) & np.isfinite(tprime_values) & (t_from_4vectors > 0) & (tprime_values > 0)
        
        if valid_both.sum() > 10:
            correlation = np.corrcoef(t_from_4vectors[valid_both], tprime_values[valid_both])[0, 1]
            relative_diff = np.abs(t_from_4vectors[valid_both] - tprime_values[valid_both]) / tprime_values[valid_both]
            mean_rel_diff = np.mean(relative_diff)
            
            print(f"Validation against tprime:")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  Mean relative difference: {mean_rel_diff:.4f} ({mean_rel_diff*100:.2f}%)")
            print(f"  Valid comparison points: {valid_both.sum()}")
            
            if correlation < 0.8 or mean_rel_diff > 0.5:
                print("  ⚠️  WARNING: Large discrepancy between methods!")
            else:
                print("  ✓ Good agreement between methods")
    
    return t_from_4vectors

def define_reconstruction_success_by_deviation(mc_values, recon_values, deviation_percent, variable='X'):
    """Define reconstruction success based on percent deviation from true value"""
    finite_mask = np.isfinite(mc_values) & np.isfinite(recon_values)
    
    # Calculate relative error
    min_threshold = 0.001 if variable in ['X', 'Y'] else 1.0
    denominator = np.maximum(np.abs(mc_values), min_threshold)
    relative_error = np.abs(recon_values - mc_values) / denominator
    
    # Convert to percentage and check against threshold
    relative_error_percent = relative_error * 100
    within_tolerance = relative_error_percent <= deviation_percent
    
    return finite_mask & within_tolerance

def create_acceptance_deviation_plots_electron_only(combined_data, variable='X', deviations=[15, 10, 5], t_bins=40, figsize=(16, 12)):
    """
    Create acceptance vs -t plots for different deviation thresholds using ONLY Electron Method.
    Now calculates -t from lambda 4-vectors instead of just reading tprime.
    """
    
    # Electron method column mapping
    electron_col_map = {
        'X': 'electron_x', 'Q2': 'electron_q2', 'Y': 'electron_y',
        'W': 'electron_w', 'NU': 'electron_nu'
    }
    
    # MC columns might be in MC_PART data
    mc_col_map = {
        'X': 'mc_x', 'Q2': 'mc_q2', 'Y': 'mc_y', 
        'W': 'mc_w', 'NU': 'mc_nu'
    }
    
    if variable not in electron_col_map:
        print(f"Variable {variable} not supported")
        return None, None
    
    mc_col = mc_col_map[variable]
    electron_col = electron_col_map[variable]
    
    # Setup subplots
    kinematics_list = list(combined_data.keys())
    n_kinematics = len(kinematics_list)
    n_deviations = len(deviations)
    
    fig, axes = plt.subplots(n_kinematics, n_deviations, figsize=figsize, squeeze=False)
    fig.suptitle(f'Acceptance vs -t: {variable} (MCLambda 4-vectors)', fontsize=16, y=0.95)
    
    colors = ['red', 'blue', 'green']
    all_results = {}
    
    for row, kinematics in enumerate(kinematics_list):
        data = combined_data[kinematics]['merged']
        
        # Check for required columns
        if mc_col not in data.columns:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, f'Missing MC {variable} data', 
                                   transform=axes[row, col].transAxes, ha='center', va='center')
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
            
        if electron_col not in data.columns:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, f'Missing Electron Method\n{variable} data', 
                                   transform=axes[row, col].transAxes, ha='center', va='center',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
        
        # Check for required lambda and DIS columns
        required_cols = ['lam_px', 'lam_py', 'lam_pz', 'q2', 'nu']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, f'Missing columns for\n-t calculation:\n{missing_cols}', 
                                   transform=axes[row, col].transAxes, ha='center', va='center',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
        
        # Calculate -t values from lambda 4-vectors
        # We use electron method kinematics for valid event selection
        Q2, x, y = data['electron_q2'], data['electron_x'], data['electron_y']
        valid_kinematics = ((Q2 > 0) & (x > 0) & (x < 1) & (y > 0) & (y < 1))
        
        # Also check for valid lambda momentum
        valid_lambda = (np.isfinite(data['lam_px']) & np.isfinite(data['lam_py']) & 
                       np.isfinite(data['lam_pz']) & np.isfinite(data['q2']) & 
                       np.isfinite(data['nu']))
        
        valid_events = valid_kinematics & valid_lambda
        
        if valid_events.sum() < 100:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, f'Insufficient data\n({valid_events.sum()} events)', 
                                   transform=axes[row, col].transAxes, ha='center', va='center')
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
        
        # Calculate -t from lambda 4-vectors
        t_values = np.full(len(data), np.nan)
        try:
            t_calculated = calculate_t_from_kinematics(data, valid_events, kinematics)
            t_values[valid_events] = t_calculated
        except ValueError as e:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, f'Error calculating -t:\n{str(e)}', 
                                   transform=axes[row, col].transAxes, ha='center', va='center',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
        
        # Check that -t extraction was successful
        valid_t = np.isfinite(t_values) & (t_values > 0)
        print(f"Successfully calculated -t for {valid_t.sum()}/{len(data)} events")
        
        mc_values = data[mc_col].values
        electron_values = data[electron_col].values  # Use ONLY electron method
        
        # Base selection
        base_valid = (valid_events & valid_t & np.isfinite(mc_values))
        
        if base_valid.sum() < 100:
            for col in range(n_deviations):
                axes[row, col].text(0.5, 0.5, 'No valid events\nafter -t calculation', 
                                   transform=axes[row, col].transAxes, ha='center', va='center')
                axes[row, col].set_title(f'{kinematics}: {deviations[col]}%')
            continue
        
        # Get -t range
        t_valid = t_values[base_valid]
        t_min, t_max = np.percentile(t_valid, [5, 95])
        t_bin_edges = np.linspace(t_min, t_max, t_bins + 1)
        t_bin_centers = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2
        
        print(f"-t range: {t_min:.4f} to {t_max:.4f} GeV²")
        
        all_results[kinematics] = {}
        
        # Plot for each deviation threshold
        for col, deviation in enumerate(deviations):
            ax = axes[row, col]
            
            acceptances = []
            acceptance_errors = []
            valid_t_centers = []
            
            for i in range(len(t_bin_edges) - 1):
                in_bin_mc = (base_valid & (t_values >= t_bin_edges[i]) & (t_values < t_bin_edges[i + 1]))
                n_mc = in_bin_mc.sum()
                
                if n_mc < 5:
                    continue
                
                success = define_reconstruction_success_by_deviation(
                    mc_values[in_bin_mc], electron_values[in_bin_mc], deviation, variable
                )
                
                n_success = success.sum()
                acceptance = n_success / n_mc
                error = np.sqrt(acceptance * (1 - acceptance) / n_mc)
                
                acceptances.append(acceptance)
                acceptance_errors.append(error)
                valid_t_centers.append(t_bin_centers[i])
            
            if len(acceptances) == 0:
                ax.text(0.5, 0.5, 'No valid bins', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{kinematics}: {deviation}%')
                continue
            
            acceptances = np.array(acceptances)
            acceptance_errors = np.array(acceptance_errors)
            valid_t_centers = np.array(valid_t_centers)
            
            # Plot
            color = colors[col % len(colors)]
            ax.errorbar(valid_t_centers, acceptances, yerr=acceptance_errors,
                       fmt='o-', color=color, capsize=3, markersize=4, alpha=0.8, linewidth=2)
            
            ax.set_xlabel('-t (GeV²) [from λ 4-vectors]')
            ax.set_ylabel('Acceptance (Electron Method)')
            ax.set_title(f'{kinematics}: ±{deviation}% tolerance')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            # Add statistics
            mean_acceptance = np.mean(acceptances)
            total_mc = base_valid.sum()
            overall_success = define_reconstruction_success_by_deviation(
                mc_values[base_valid], electron_values[base_valid], deviation, variable
            )
            overall_acceptance = overall_success.sum() / total_mc
            
            
            all_results[kinematics][f'{deviation}%'] = {
                't_centers': valid_t_centers,
                'acceptances': acceptances,
                'errors': acceptance_errors,
                'mean_acceptance': mean_acceptance,
                'overall_acceptance': overall_acceptance,
                'total_events': total_mc,
                'successful_events': overall_success.sum(),
                'method': 'Electron Method',
                't_from_lambda_4vectors': True  # Flag to indicate -t was calculated from lambda 4-vectors
            }
    
    plt.tight_layout()
    return fig, all_results

def print_deviation_summary(all_results, variable='X'):
    """Print numerical summary of results"""
    print("\n" + "="*80)
    print(f"ACCEPTANCE ANALYSIS SUMMARY: {variable} RECONSTRUCTION QUALITY (ELECTRON METHOD)")
    print("NOTE: -t values are calculated from lambda 4-vectors and virtual photon kinematics")
    print("="*80)
    
    deviations = [f'{d}%' for d in DEVIATION_THRESHOLDS]
    
    for kinematics in all_results:
        print(f"\nKinematics: {kinematics}")
        print("-" * 50)
        print(f"{'Tolerance':<12} {'Acceptance':<12} {'Events Used':<15} {'Success Rate'}")
        print("-" * 50)
        
        for deviation in deviations:
            if deviation in all_results[kinematics]:
                data = all_results[kinematics][deviation]
                acceptance = data['overall_acceptance']
                total = data['total_events']
                successful = data['successful_events']
                
                print(f"±{deviation:<11} {acceptance:<12.3f} {successful:,}/{total:,:<10} {successful/total*100:.1f}%")
            else:
                print(f"±{deviation:<11} {'No data':<12} {'N/A':<15} {'N/A'}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*80)
    print("ELECTRON METHOD ACCEPTANCE vs -t ANALYSIS")
    print("="*80)
    print(f"Data file: {top_zip_path}")
    print(f"Deviation thresholds: {DEVIATION_THRESHOLDS}%")
    print(f"Number of -t bins: {T_BINS}")
    
    try:
        # Load and process data
        combined_data = load_and_process_data()
        
        available_kinematics = sorted(combined_data.keys())
        print(f"\n✓ Available kinematics: {available_kinematics}")
        
        # Print data statistics
        for kinematics in available_kinematics:
            merged_data = combined_data[kinematics]['merged']
            print(f"\n{kinematics} statistics:")
            print(f"  Total aligned events: {len(merged_data)}")
            print(f"  Event files used: {merged_data['event_file'].nunique()}")
            
            # Check for electron method data
            electron_cols = ['electron_x', 'electron_q2', 'electron_y', 'electron_w', 'electron_nu']
            
            # Check for lambda momentum columns
            lambda_cols = ['lam_px', 'lam_py', 'lam_pz']
            available_lambda_cols = [col for col in lambda_cols if col in merged_data.columns]
            print(f"  Available lambda momentum columns: {len(available_lambda_cols)}/{len(lambda_cols)}")
            
            # Check for DIS kinematic columns
            dis_cols = ['q2', 'nu']
            available_dis_cols = [col for col in dis_cols if col in merged_data.columns]
            print(f"  Available DIS columns: {len(available_dis_cols)}/{len(dis_cols)}")
            
            if len(available_lambda_cols) == len(lambda_cols) and len(available_dis_cols) == len(dis_cols):
                print("    ✓ All required columns found for -t calculation from 4-vectors")
                
                # Show some statistics
                valid_lambda = (np.isfinite(merged_data['lam_px']) & 
                               np.isfinite(merged_data['lam_py']) & 
                               np.isfinite(merged_data['lam_pz']))
                print(f"    Valid lambda events: {valid_lambda.sum()}/{len(merged_data)}")
            else:
                print("    ❌ Missing required columns for -t calculation")
        
        # Run acceptance analysis for different variables
        variables_to_analyze = ['X', 'Q2', 'Y']  # Can add 'W', 'NU' if needed
        
        for variable in variables_to_analyze:
            print(f"\n{'='*60}")
            print(f"ANALYZING {variable} RECONSTRUCTION QUALITY (ELECTRON METHOD)")
            print(f"(-t calculated from lambda 4-vectors and virtual photon kinematics)")
            print(f"{'='*60}")
            
            # Create acceptance vs -t plots
            fig, results = create_acceptance_deviation_plots_electron_only(
                combined_data, 
                variable=variable, 
                deviations=DEVIATION_THRESHOLDS, 
                t_bins=T_BINS
            )
            
            if fig is not None and SAVE_PLOTS:
                filename = f'acceptance_vs_t_{variable.lower()}_electron_method_lambda_4vectors.png'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {filename}")
            
            if SHOW_PLOTS and fig is not None:
                plt.show()
            elif fig is not None:
                plt.close(fig)
            
            # Print numerical summary
            if results:
                print_deviation_summary(results, variable=variable)
            else:
                print(f"❌ No results generated for {variable} - check your data")
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"✓ Generated acceptance vs -t plots for variables: {variables_to_analyze}")
        print(f"✓ Used deviation thresholds: ±{DEVIATION_THRESHOLDS}%")
        print("✓ All -t values were calculated from lambda 4-vectors and virtual photon")
        print("✓ Data properly aligned across MC_DIS, MC_PART, and RECON files")
        if SAVE_PLOTS:
            print("✓ Plots saved as PNG files in the current directory")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        if os.path.exists(extract_root):
            shutil.rmtree(extract_root)
            print("✓ Cleaned up temporary files")

if __name__ == "__main__":
    main()
