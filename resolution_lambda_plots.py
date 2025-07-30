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
warnings.filterwarnings('ignore')

# Configuration - Update this path to your data location
top_zip_path = r"C:\Users\Sweetie Pie\Desktop\CUA REU\meson-strcutrue-2025-06-05-csv-20250623T143248Z-1-001.zip"
extract_root = 'meson_temp_extract'

print("Loading data for resolution plot recreation...")

# Extract and process data (same as original code)
with zipfile.ZipFile(top_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_root)

inner_zip_folder = os.path.join(extract_root, 'meson-strcutrue-2025-06-05-csv')
inner_zip_files = [os.path.join(inner_zip_folder, f) for f in os.listdir(inner_zip_folder) if f.endswith('.zip')]

print(f"Found {len(inner_zip_files)} zip files")

# Data organization (same as original)
data_inventory = defaultdict(lambda: defaultdict(dict))

def parse_filename(filename, extensions_info=None):
    """Parse filename to extract data - same as original"""
    if extensions_info and extensions_info.get('debug_mode', False):
        print(f"DEBUG: Parsing filename: {filename}")
    
    name = filename.replace('.zip', '')
    parts = name.split('.')
    
    if len(parts) < 2:
        return None, None, None, None
    
    base_name = parts[0]
    file_type = parts[1]
    
    base_parts = base_name.split('_')
    if len(base_parts) < 5:
        return None, None, None, None
    
    process_name = '_'.join(base_parts[:2])
    kinematics = base_parts[2]
    event_number = base_parts[4]
    
    if extensions_info and 'mc_extension' in extensions_info:
        mc_ext = extensions_info['mc_extension']
        recon_ext = extensions_info['recon_extension']
    else:
        mc_ext = 'mc'
        recon_ext = 'reco_dis'
    
    if file_type == mc_ext:
        data_type = 'mc'
    elif file_type == recon_ext:
        data_type = 'recon'
    else:
        data_type = None
    
    return process_name, kinematics, event_number, data_type

def auto_detect_extensions(zip_files, sample_size=50):
    """Auto-detect file extensions - looking specifically for mc_dis and reco_dis"""
    print(f"\nAuto-detecting file extensions from {min(sample_size, len(zip_files))} sample files...")
    
    extensions = defaultdict(list)
    
    for zip_path in zip_files[:sample_size]:
        filename = os.path.basename(zip_path)
        name_without_zip = filename.replace('.zip', '')
        parts = name_without_zip.split('.')
        
        if len(parts) >= 2:
            ext = parts[1]
            extensions[ext].append(filename)
        else:
            extensions['no_extension'].append(filename)
    
    print(f"Found extensions: {list(extensions.keys())}")
    
    # Look specifically for mc_dis and reco_dis
    mc_ext = 'mc_dis' if 'mc_dis' in extensions else None
    recon_ext = 'reco_dis' if 'reco_dis' in extensions else None
    
    if not mc_ext:
        # Fallback: look for other MC-like extensions
        for ext in extensions.keys():
            if 'mc' in ext.lower() and 'dis' in ext.lower():
                mc_ext = ext
                break
    
    if not recon_ext:
        # Fallback: look for other reconstruction-like extensions
        for ext in extensions.keys():
            if any(word in ext.lower() for word in ['reco', 'recon']) and 'dis' in ext.lower():
                recon_ext = ext
                break
    
    if not mc_ext or not recon_ext:
        print(f"Warning: Could not find expected extensions. Found: {list(extensions.keys())}")
        print(f"Looking for: mc_dis and reco_dis")
        return 'mc_dis', 'reco_dis', list(extensions.keys())
    
    return mc_ext, recon_ext, None

def load_csv_from_zip(zip_path):
    """Load CSV from zip file - same as original"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = None
            for name in z.namelist():
                if name.endswith('.csv'):
                    csv_name = name
                    break
            
            if csv_name is None:
                return None
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                z.extract(csv_name, tmpdirname)
                csv_full_path = os.path.join(tmpdirname, csv_name)
                df = pd.read_csv(csv_full_path)
                return df
                
    except Exception as e:
        print(f"Error loading {zip_path}: {e}")
        return None

# Process files and organize by event pairing
print("Building data inventory...")

mc_ext, recon_ext, all_exts = auto_detect_extensions(inner_zip_files)

if mc_ext and recon_ext:
    print(f"✓ Auto-detected extensions: MC='{mc_ext}', Reconstruction='{recon_ext}'")
    extensions_info = {
        'mc_extension': mc_ext,
        'recon_extension': recon_ext,
        'debug_mode': False
    }
else:
    print("Using default extensions: MC='mc_dis', Reconstruction='reco_dis'")
    extensions_info = {
        'mc_extension': 'mc_dis',  # Updated default
        'recon_extension': 'reco_dis',
        'debug_mode': False
    }

file_type_counts = defaultdict(int)

for i, zip_path in enumerate(inner_zip_files):
    if i % 50 == 0:
        print(f"Processing file {i+1}/{len(inner_zip_files)}")
    
    zip_filename = os.path.basename(zip_path)
    process_name, kinematics, event_number, data_type = parse_filename(zip_filename, extensions_info)
    
    if data_type:
        file_type_counts[data_type] += 1
    else:
        file_type_counts['unknown'] += 1
    
    if not all([process_name, kinematics, event_number, data_type]):
        continue
    
    df = load_csv_from_zip(zip_path)
    if df is not None:
        # Debug: Show column info for first few files
        if len(data_inventory) == 0 and data_type in ['mc', 'recon']:
            print(f"Sample {data_type} file columns: {list(df.columns)[:10]}...")
        data_inventory[kinematics][event_number][data_type] = df

print(f"\nFile type summary:")
for file_type, count in file_type_counts.items():
    print(f"  {file_type}: {count} files")

if file_type_counts['mc'] == 0:
    print("\n⚠️  CRITICAL: No MC files detected!")
    print("Check that mc_dis files are being properly identified.")
    print(f"Current settings: MC extension='{extensions_info['mc_extension']}', Recon extension='{extensions_info['recon_extension']}'")
else:
    print(f"\n✓ Successfully detected {file_type_counts['mc']} MC files and {file_type_counts['recon']} reconstruction files")

# Combine data ensuring proper MC-reconstruction pairing
print("Combining data with proper event alignment...")
combined_data = {}

for kinematics in data_inventory:
    print(f"Processing kinematics: {kinematics}")
    
    events_with_both = []
    for event_number in data_inventory[kinematics]:
        has_mc = 'mc' in data_inventory[kinematics][event_number]
        has_recon = 'recon' in data_inventory[kinematics][event_number]
        if has_mc and has_recon:
            events_with_both.append(event_number)
    
    print(f"  Events with both MC and recon: {len(events_with_both)}")
    
    if len(events_with_both) == 0:
        continue
    
    mc_events = []
    recon_events = []
    
    for event_number in sorted(events_with_both):
        mc_df = data_inventory[kinematics][event_number]['mc']
        recon_df = data_inventory[kinematics][event_number]['recon']
        
        mc_df_copy = mc_df.copy()
        recon_df_copy = recon_df.copy()
        mc_df_copy['event_file'] = event_number
        recon_df_copy['event_file'] = event_number
        mc_df_copy['row_index'] = range(len(mc_df_copy))
        recon_df_copy['row_index'] = range(len(recon_df_copy))
        
        mc_events.append(mc_df_copy)
        recon_events.append(recon_df_copy)
    
    if mc_events and recon_events:
        mc_combined = pd.concat(mc_events, ignore_index=False)
        recon_combined = pd.concat(recon_events, ignore_index=False)
        
        mc_combined['alignment_key'] = mc_combined['event_file'].astype(str) + '_' + mc_combined['row_index'].astype(str)
        recon_combined['alignment_key'] = recon_combined['event_file'].astype(str) + '_' + recon_combined['row_index'].astype(str)
        
        merged_data = pd.merge(mc_combined, recon_combined, on='alignment_key', suffixes=('', '_reco'))
        
        if 'event_file_reco' in merged_data.columns:
            if merged_data['event_file'].equals(merged_data['event_file_reco']):
                merged_data = merged_data.drop('event_file_reco', axis=1)
        
        if 'row_index_reco' in merged_data.columns:
            if merged_data['row_index'].equals(merged_data['row_index_reco']):
                merged_data = merged_data.drop('row_index_reco', axis=1)
        
        if len(merged_data) > 0:
            combined_data[kinematics] = {'merged': merged_data}
            print(f"  Successfully paired {len(merged_data)} events")

print(f"Found properly aligned data for kinematics: {list(combined_data.keys())}")

def create_resolution_plots_by_kinematics(kinematics, figsize=(16, 12), bins=100):
    """
    Create resolution plots matching pages 18-20 of the presentation
    Shows (Reco - Truth)/Truth vs Truth for Q² and X_bj with y-cuts
    """
    if kinematics not in combined_data or 'merged' not in combined_data[kinematics]:
        print(f"No merged data for {kinematics}")
        return None
    
    data = combined_data[kinematics]['merged']
    
    # Check for required columns - updated to use correct column names
    required_cols = ['q2', 'electron_q2', 'xbj', 'electron_x']  # Fixed: electron_x not electron_xbj
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Missing columns for {kinematics}: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        return None
    
    # Check for y column (for cuts) - try different possible names
    y_col = None
    possible_y_cols = ['y_d', 'y', 'mc_y', 'truth_y']  # Put y_d first as most likely candidate
    for col in possible_y_cols:
        if col in data.columns:
            y_col = col
            break
    
    if y_col is None:
        print(f"Warning: No y column found for {kinematics}. Available columns: {list(data.columns)}")
        # Create plots without y-cuts
        y_cuts = [('all events', pd.Series([True] * len(data), index=data.index))]
    else:
        print(f"Using y column: '{y_col}' for inelasticity cuts")
        # Define y-cuts using the found y column
        y_cuts = [
            ('y > 0.1', data[y_col] > 0.1),
            ('y <= 0.1', data[y_col] <= 0.1)
        ]
    
    # Create figure with 2x2 subplot layout (or 2x1 if no y-cuts)
    if len(y_cuts) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2))
        axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    
    # Variables to plot - updated column names
    variables = [
        ('Q²', 'q2', 'electron_q2'),           # MC: q2, Recon: electron_q2
        ('X_bj', 'xbj', 'electron_x')          # MC: xbj, Recon: electron_x (not electron_xbj!)
    ]
    
    for var_idx, (var_name, mc_col, recon_col) in enumerate(variables):
        for cut_idx, (cut_name, cut_mask) in enumerate(y_cuts):
            
            # Handle different subplot layouts
            if len(y_cuts) == 1:
                # Only one row needed
                row = 0
                col = var_idx
                if var_idx >= axes.shape[1]:
                    continue  # Skip if we don't have enough columns
            else:
                # Standard 2x2 layout
                row = var_idx
                col = cut_idx
            
            ax = axes[row, col]
            
            # Apply cut and filter valid data
            cut_data = data[cut_mask]
            mc_valid = cut_data[mc_col].notna()
            recon_valid = cut_data[recon_col].notna()
            both_valid = mc_valid & recon_valid
            
            if both_valid.sum() == 0:
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{kinematics}: {var_name} Resolution vs Truth\n(Cut: {cut_name}, Method: electron)')
                continue
            
            # Get valid data
            mc_values = cut_data[mc_col][both_valid].values
            recon_values = cut_data[recon_col][both_valid].values
            
            # Calculate resolution: (Reco - Truth) / Truth
            resolution = (recon_values - mc_values) / mc_values
            
            # Filter out extreme outliers for better visualization
            valid_resolution = np.isfinite(resolution)
            mc_plot = mc_values[valid_resolution]
            res_plot = resolution[valid_resolution]
            
            if len(mc_plot) == 0:
                ax.text(0.5, 0.5, 'No valid resolution data', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{kinematics}: {var_name} Resolution vs Truth\n(Cut: {cut_name}, Method: electron)')
                continue
            
            # Set appropriate axis limits based on the variable
            if var_name == 'Q²':
                # Q² typically ranges from 0 to several hundred
                x_min, x_max = 0, np.percentile(mc_plot, 99)
                y_min, y_max = np.percentile(res_plot, 1), np.percentile(res_plot, 99)
            else:  # X_bj
                # X_bj ranges from 0 to 1
                x_min, x_max = 0, min(1, np.percentile(mc_plot, 99))
                y_min, y_max = np.percentile(res_plot, 1), np.percentile(res_plot, 99)
            
            # Create 2D histogram
            try:
                hist, xedges, yedges = np.histogram2d(mc_plot, res_plot, bins=bins, 
                                                    range=[[x_min, x_max], [y_min, y_max]])
                
                # Plot with log scale (like in the presentation)
                im = ax.imshow(hist.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                              cmap='viridis', aspect='auto', interpolation='nearest',
                              norm=colors.LogNorm(vmin=max(1, hist.min()), vmax=hist.max()))
                
                # Add horizontal line at y=0 (perfect reconstruction)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Counts')
                
                # Set limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                # Add entry count
                ax.text(0.05, 0.95, f'Entries: {len(mc_plot):,}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')
                
            except Exception as e:
                print(f"Error creating histogram for {var_name} with cut {cut_name}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, ha='center', va='center')
            
            # Labels and title
            ax.set_xlabel(f'Truth {var_name}')
            ax.set_ylabel(f'({var_name} Reco - Truth) / Truth')
            ax.set_title(f'{kinematics}: {var_name} Resolution vs Truth\n(Cut: {cut_name}, Method: electron)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main execution for creating plots like pages 18-20
if __name__ == "__main__":
    # Configuration - Control what happens with the plots
    SAVE_TO_PDF = True  # Set to True if you want to save plots to PDF file
    SHOW_PLOTS = True    # Set to True to display plots on screen (recommended!)
    PDF_FILENAME = f"Resolution_Plots_Recreation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Note: You can set both SAVE_TO_PDF=True and SHOW_PLOTS=True to both save and display
    
    # Find available kinematics
    available_kinematics = list(combined_data.keys())
    
    if not available_kinematics:
        print("No properly aligned data found!")
    else:
        print(f"Available kinematics: {available_kinematics}")
        
        # Sort to match presentation order: 5x41, 10x100, 18x275
        target_kinematics = ['5x41', '10x100', '18x275']
        plot_kinematics = [k for k in target_kinematics if k in available_kinematics]
        
        if not plot_kinematics:
            print("Target kinematics not found, using available ones:")
            plot_kinematics = sorted(available_kinematics)
        
        print(f"Creating plots for kinematics: {plot_kinematics}")
        
        # Debug: Show sample data structure
        if plot_kinematics:
            sample_kinematics = plot_kinematics[0]
            sample_data = combined_data[sample_kinematics]['merged']
            print(f"\nSample data columns for {sample_kinematics}:")
            print(f"  Total columns: {len(sample_data.columns)}")
            print(f"  Sample columns: {list(sample_data.columns)[:20]}...")  # Show first 20
            
            # Look for Q2 and x-related columns specifically
            q2_cols = [col for col in sample_data.columns if 'q2' in col.lower()]
            x_cols = [col for col in sample_data.columns if 'x' in col.lower()]
            y_cols = [col for col in sample_data.columns if 'y' in col.lower()]
            
            print(f"  Q2-related columns: {q2_cols}")
            print(f"  X-related columns: {x_cols}")
            print(f"  Y-related columns: {y_cols}")
        
        # Initialize PDF if saving
        if SAVE_TO_PDF:
            pdf_pages = PdfPages(PDF_FILENAME)
            print(f"Saving plots to: {PDF_FILENAME}")
        
        if SHOW_PLOTS:
            print("Displaying plots on screen...")
            print("Close each plot window to continue to the next plot.")
        
        try:
            # Create title page (only if saving to PDF)
            if SAVE_TO_PDF:
                fig_title = plt.figure(figsize=(8.5, 11))
                fig_title.suptitle('Resolution Plots Recreation - Pages 18-20', fontsize=20, y=0.85)
                
                ax_title = fig_title.add_subplot(111)
                ax_title.axis('off')
                
                summary_text = f"""
                Recreation of Resolution Plots from Presentation Pages 18-20
                
                Plots show: (Reconstruction - Truth) / Truth vs Truth
                
                Variables: Q² and X_bj (Bjorken x)
                Method: Electron reconstruction
                Y-parameter cuts: y > 0.1 and y <= 0.1
                
                Data Sources:
                - MC Truth: mc_dis files (columns: q2, xbj)
                - Reconstruction: reco_dis files (columns: electron_q2, electron_xbj)
                
                Kinematics analyzed: {', '.join(plot_kinematics)}
                
                These plots demonstrate the resolution performance of the electron
                reconstruction method across different kinematic regions and
                inelasticity ranges.
                
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                ax_title.text(0.1, 0.75, summary_text, fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                pdf_pages.savefig(fig_title, bbox_inches='tight')
                plt.close(fig_title)
            
            plot_count = 0
            
            # Create resolution plots for each kinematics
            for kinematics in plot_kinematics:
                print(f"\nCreating resolution plots for {kinematics}")
                
                fig = create_resolution_plots_by_kinematics(kinematics)
                if fig:
                    if SAVE_TO_PDF:
                        pdf_pages.savefig(fig, bbox_inches='tight')
                        print(f"  ✓ Saved plot for {kinematics} to PDF")
                    
                    if SHOW_PLOTS:
                        plt.show()  # This will display the plot and wait for user to close it
                        print(f"  ✓ Displayed plot for {kinematics}")
                    elif not SAVE_TO_PDF:
                        # If we're neither showing nor saving, close the figure to free memory
                        plt.close(fig)
                    
                    plot_count += 1
                else:
                    print(f"  ✗ Failed to create plot for {kinematics}")
        
        finally:
            # Close PDF
            if SAVE_TO_PDF:
                pdf_pages.close()
                print(f"\n✓ Successfully saved {plot_count} plots to {PDF_FILENAME}")
                print(f"  File size: {os.path.getsize(PDF_FILENAME) / (1024*1024):.1f} MB")
                print(f"  You can now open {PDF_FILENAME} to view the recreated resolution plots!")
            
            if SHOW_PLOTS:
                print(f"\n✓ Successfully displayed {plot_count} plots on screen.")
            
            if not SAVE_TO_PDF and not SHOW_PLOTS:
                print(f"\n✓ Successfully created {plot_count} plots (but did not save or display them).")

# Cleanup
if os.path.exists(extract_root):
    shutil.rmtree(extract_root)

print("\nDone! Created resolution plots matching presentation pages 18-20.")
if SHOW_PLOTS:
    print("Plots have been displayed on screen.")
if SAVE_TO_PDF:
    print("Plots have been saved to PDF.")