import zipfile
import os
import pandas as pd
import tempfile
import shutil
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
# Path to main zip file containing the data
top_zip_path = r"C:\Users\Sweetie Pie\Desktop\CUA REU\meson-strcutrue-2025-06-05-csv-20250623T143248Z-1-001.zip"
extract_root = 'meson_temp_extract'

# PARTICLE MASSES (GeV)
M_LAMBDA = 1.115683  # Lambda mass

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def parse_filename(filename):
    """Parse filename to extract metadata"""
    name = filename.replace('.csv.zip', '')
    parts = name.split('.')
    
    if len(parts) != 2:
        return None, None, None, None
    
    base_name, file_type = parts
    base_parts = base_name.split('_')
    
    if len(base_parts) != 5:
        return None, None, None, None
    
    process_name = '_'.join(base_parts[:2])  # k_lambda
    kinematics = base_parts[2]  # 5x41, 10x100, or 18x275
    event_number = base_parts[4]  # 001-200
    
    # Map file types
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
    """Load CSV from zip file"""
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
    


def load_mc_data():
    #Loading MC_dis, MC_lamdba
    print("Loading MC data...")
    
    # Extract main zip file
    with zipfile.ZipFile(top_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_root)
    
    inner_zip_folder = os.path.join(extract_root, 'meson-strcutrue-2025-06-05-csv')
    inner_zip_files = [os.path.join(inner_zip_folder, f) 
                      for f in os.listdir(inner_zip_folder) if f.endswith('.zip')]
    
    print(f"Found {len(inner_zip_files)} zip files")
    
    # Organize data by kinematics and type
    mc_part_data = defaultdict(list)
    mc_dis_data = defaultdict(list)

    
    # Process files
    for zip_path in inner_zip_files:
        zip_filename = os.path.basename(zip_path)
        process_name, kinematics, event_number, data_type = parse_filename(zip_filename)
        
        if data_type == 'mc_part':
            df = load_csv_from_zip(zip_path)
            if df is not None:
                df['event_file'] = event_number
                mc_part_data[kinematics].append(df)
        elif data_type == 'mc_dis':
            df = load_csv_from_zip(zip_path)
            if df is not None:
                df['event_file'] = event_number
                mc_dis_data[kinematics].append(df)
    
    # Combine data by kinematics
    combined_data_MC = {}
    for kinematics in mc_part_data:
        if mc_part_data[kinematics] and mc_dis_data[kinematics]:
            mc_part_combined = pd.concat(mc_part_data[kinematics], ignore_index=True)
            mc_dis_combined = pd.concat(mc_dis_data[kinematics], ignore_index=True)
            
            combined_data_MC[kinematics] = {
                'mc_part': mc_part_combined,
                'mc_dis': mc_dis_combined
            }
            print(f"Kinematics {kinematics}: {len(mc_part_combined)} MC_PART, {len(mc_dis_combined)} MC_DIS events")
    
    return combined_data_MC

# =============================================================================
# T-VALUE CALCULATION FUNCTIONS
# =============================================================================

def get_beam_energy_from_kinematics(kinematics):
    """Extract beam energy from kinematics name"""
    if kinematics.startswith('10x'):
        return 10.0
    elif kinematics.startswith('18x'):
        return 18.0
    elif kinematics.startswith('5x'):
        return 5.0


def calculate_t_from_lambda_4vectors(mcpart_lambda, mc_dis, kinematics):
    
    #Take one DIS entry per event_file
    dis_per_event = mc_dis.groupby('event_file')[['q2', 'nu']].first().reset_index()
    
    #Merge lambda data with DIS kinematics
    merged = mcpart_lambda.merge(
        right = dis_per_event, 
        on='event_file', 
    )
    
    #Drop rows with missing values
    merged = merged.dropna(subset=['lam_px', 'lam_py', 'lam_pz', 'q2', 'nu'])
    
    
    #Extract momentum and DIS variables
    px = merged['lam_px'].values
    py = merged['lam_py'].values
    pz = merged['lam_pz'].values
    Q2 = merged['q2'].values 
    nu = merged['nu'].values
    
    #Calculate lambda energy/virtual photon properties
    lam_E = np.sqrt(px**2 + py**2 + pz**2 + M_LAMBDA**2)
    gamma_E = nu
    qmag = np.sqrt(nu**2 + Q2)
    
    #t = (q - p_lambda)²
    #Virtual photon 4-momentum
    dE = gamma_E - lam_E
    dpx = -px  #gamma_px = 0
    dpy = -py  #gamma_py = 0  
    dpz = -qmag - pz  #gamma_pz = -qmag
    
    t = dE**2 - dpx**2 - dpy**2 - dpz**2
    minus_t = -t
    
    # Keep only finite positive values
    valid_t = minus_t[np.isfinite(minus_t) & (minus_t > 0)]
    
    return valid_t

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_t_distribution(all_t_values, kinematics_labels):
    
    if not all_t_values:
        print("No data to plot")
        return None
    
    # Combine all t-values
    combined_t = np.concatenate([t for t in all_t_values if t is not None])
    
    if len(combined_t) == 0:
        print("No valid t-values to plot")
        return None
    
    # Calculate statistics
    mean_t = np.mean(combined_t)
    std_t = np.std(combined_t)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    

    n_bins = 50
    bin_range = None
    
    # Create histogram
    counts, bins, patches = plt.hist(combined_t, bins=n_bins, range=bin_range,
                                   alpha=0.7, color='black', 
                                   linewidth=0.5, label='Truth')
    
    #Formatting
    plt.xlabel('t [GeV²]', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title('MC -t Distribution (MC Truth)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    #statistics box
    stats_text = f'Truth Mean = {mean_t:.3f}\nTruth Std Dev = {std_t:.3f}'
    plt.text(0.65, 0.85, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    print(f"\nPlot Statistics:")
    print(f"Total events: {len(combined_t)}")
    print(f"Mean t: {mean_t:.6f} GeV²")
    print(f"Std Dev t: {std_t:.6f} GeV²")
    print(f"Min t: {np.min(combined_t):.6f} GeV²")
    print(f"Max t: {np.max(combined_t):.6f} GeV²")
    print(f"Range: {np.max(combined_t) - np.min(combined_t):.6f} GeV²")
    
    # Check if distribution looks like the original
    if mean_t > 0 and np.abs(mean_t) > 1000:
        print("\n⚠️  WARNING: t-values are large and positive.")
        print("   Original shows small negative values around -150.")
        print("   This suggests a units or calculation issue.")
    elif mean_t < 0 and np.abs(mean_t) < 1000:
        print("\n✓ t-values look similar to original distribution")
    
    return plt.gcf()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*60)
    print("MC TRUTH T-VALUES HISTOGRAM PLOTTER")
    print("="*60)
    
    try:
        # Load MC data (both MC_PART and MC_DIS)
        mc_data = load_mc_data()
        
        if not mc_data:
            print("No MC data found!")
            return
        
        # Calculate t-values for each kinematics
        all_t_values = []
        kinematics_labels = []
        
        for kinematics in sorted(mc_data.keys()):
            print(f"\nProcessing kinematics: {kinematics}")
            mc_part = mc_data[kinematics]['mc_part']
            mc_dis = mc_data[kinematics]['mc_dis']
            
            # Debug: Show available columns
            print(f"MC_PART columns: {list(mc_part.columns)}")
            print(f"MC_DIS columns: {list(mc_dis.columns)}")
            
            # Check for t-related columns in MC_PART
            t_cols = [col for col in mc_part.columns if 't' in col.lower()]
            if t_cols:
                print(f"Columns containing 't' in MC_PART: {t_cols}")
            
            t_values = calculate_t_from_lambda_4vectors(mc_part, mc_dis, kinematics)
            
            if t_values is not None:
                all_t_values.append(t_values)
                kinematics_labels.append(kinematics)
                print(f"✓ Successfully extracted {len(t_values)} t-values for {kinematics}")
            else:
                print("Could not calculate t-values for {kinematics}")
        
        print(f"\nSummary: Successfully processed {len(all_t_values)} out of {len(mc_data)} kinematics")
        
        # Create and display plot
        if all_t_values:
            fig = plot_t_distribution(all_t_values, kinematics_labels)
            
            if fig is not None:
                # Save plot
                output_filename = 'mc_truth_t_distribution.png'
                fig.savefig(output_filename, dpi=300, bbox_inches='tight')
                print(f"\n✓ Plot saved as: {output_filename}")
                
                # Show plot
                plt.show()
            else:
                print("Could not create plot")
        else:
            print("No valid t-values calculated from any kinematics")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup temporary files
        if os.path.exists(extract_root):
            shutil.rmtree(extract_root)
            print("✓ Cleaned up temporary files")

if __name__ == "__main__":
    main()