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

top_zip_path = r"C:\Users\Sweetie Pie\Desktop\CUA REU\meson-strcutrue-2025-06-05-csv-20250623T143248Z-1-001.zip"
extract_root = 'meson_temp_extract'

print("Loading data...")

# Extract and process data
with zipfile.ZipFile(top_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_root)

inner_zip_folder = os.path.join(extract_root, 'meson-strcutrue-2025-06-05-csv')
inner_zip_files = [os.path.join(inner_zip_folder, f) for f in os.listdir(inner_zip_folder) if f.endswith('.zip')]

data_by_kinematics = defaultdict(lambda: {'mc': [], 'recon': []})

def parse_filename(filename):
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
    data_type = 'mc' if file_type == 'mc' else 'recon' if file_type == 'reco_dis' else None
    
    return process_name, kinematics, event_number, data_type

# Process files
for i, zip_path in enumerate(inner_zip_files):
    if i % 50 == 0:
        print(f"Processing file {i+1}/{len(inner_zip_files)}")
    
    zip_filename = os.path.basename(zip_path)
    process_name, kinematics, event_number, data_type = parse_filename(zip_filename)
    
    if not all([process_name, kinematics, event_number, data_type]):
        continue
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = None
            for name in z.namelist():
                if name.endswith('.csv'):
                    csv_name = name
                    break
            
            if csv_name is None:
                continue
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                z.extract(csv_name, tmpdirname)
                csv_full_path = os.path.join(tmpdirname, csv_name)
                df = pd.read_csv(csv_full_path)
                data_by_kinematics[kinematics][data_type].append(df)
                
    except Exception as e:
        continue

# Combine data
combined_data = {}
for kinematics in data_by_kinematics:
    combined_data[kinematics] = {}
    for data_type in ['mc', 'recon']:
        if data_by_kinematics[kinematics][data_type]:
            combined_df = pd.concat(data_by_kinematics[kinematics][data_type], ignore_index=True)
            combined_data[kinematics][data_type] = combined_df

print(f"Found data for kinematics: {list(combined_data.keys())}")

# Define variables to compare
VARIABLES = {
    'X': {'mc_col': 'mc_x', 'recon_cols': ['da_x', 'electron_x', 'jb_x', 'ml_x', 'sigma_x'], 'label': 'x'},
    'Q2': {'mc_col': 'mc_q2', 'recon_cols': ['da_q2', 'electron_q2', 'jb_q2', 'ml_q2', 'sigma_q2'], 'label': 'Q²'},
    'Y': {'mc_col': 'mc_y', 'recon_cols': ['da_y', 'electron_y', 'jb_y', 'ml_y', 'sigma_y'], 'label': 'y'},
    'W': {'mc_col': 'mc_w', 'recon_cols': ['da_w', 'electron_w', 'jb_w', 'ml_w', 'sigma_w'], 'label': 'W'},
    'NU': {'mc_col': 'mc_nu', 'recon_cols': ['da_nu', 'electron_nu', 'jb_nu', 'ml_nu', 'sigma_nu'], 'label': 'ν'}
}

METHOD_NAMES = {
    'da': 'Double Angle',
    'electron': 'Electron', 
    'jb': 'Jacquet-Blondel',
    'ml': 'Machine Learning',
    'sigma': 'Sigma'
}

def remove_outliers(data, percentile=95):
    """Remove outliers based on percentiles to improve visualization"""
    if len(data) == 0:
        return data
    lower = np.percentile(data, 100 - percentile)
    upper = np.percentile(data, percentile)
    return data[(data >= lower) & (data <= upper)]

def get_robust_limits(x_data, y_data, percentile=95):
    """Get robust axis limits based on percentiles"""
    all_data = np.concatenate([x_data, y_data])
    all_data = all_data[np.isfinite(all_data)]
    if len(all_data) == 0:
        return 0, 1, 0, 1
    
    lower = np.percentile(all_data, 100 - percentile)
    upper = np.percentile(all_data, percentile)
    margin = (upper - lower) * 0.1
    
    return lower - margin, upper + margin, lower - margin, upper + margin

def create_2d_histogram_plots(kinematics, variable, figsize=(20, 12), bins=100, max_points=50000):
    """
    Create 2D histogram plots showing event density distribution
    """
    if kinematics not in combined_data or 'recon' not in combined_data[kinematics]:
        print(f"No data for {kinematics}")
        return None
    
    data = combined_data[kinematics]['recon']
    var_info = VARIABLES[variable]
    mc_col = var_info['mc_col']
    
    if mc_col not in data.columns:
        print(f"MC column {mc_col} not found")
        return None
    
    mc_values = data[mc_col].dropna()
    
    # Get available methods
    available_methods = []
    method_data = {}
    for recon_col in var_info['recon_cols']:
        if recon_col in data.columns:
            method_name = recon_col.split('_')[0]
            available_methods.append(method_name)
            method_data[method_name] = data[recon_col].dropna()
    
    if not available_methods:
        return None
    
    # Create subplot grid
    n_methods = len(available_methods)
    cols = 3
    rows = (n_methods + 2) // cols  # +1 for MC reference
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'2D Histogram: {variable} Reconstruction vs Monte Carlo\nKinematics: {kinematics}', 
                 fontsize=16, y=0.95)
    
    plot_idx = 0
    
    # Plot Monte Carlo reference first (diagonal line for perfect correlation)
    row = plot_idx // cols
    col = plot_idx % cols
    ax = axes[row, col]
    
    # Sample MC data for reference
    mc_sample = mc_values.values
    if len(mc_sample) > max_points:
        idx = np.random.choice(len(mc_sample), max_points, replace=False)
        mc_sample = mc_sample[idx]
    
    # Remove outliers for better visualization
    mc_clean = remove_outliers(pd.Series(mc_sample))
    
    # Create a 2D histogram for MC vs MC (should be diagonal)
    if len(mc_clean) > 0:
        xlim_min, xlim_max, _, _ = get_robust_limits(mc_clean, mc_clean)
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(mc_clean, mc_clean, bins=bins, 
                                            range=[[xlim_min, xlim_max], [xlim_min, xlim_max]])
        
        # Plot histogram
        im = ax.imshow(hist.T, origin='lower', extent=[xlim_min, xlim_max, xlim_min, xlim_max], 
                      cmap='Blues', aspect='auto', interpolation='nearest')
        
        # Add perfect correlation line
        ax.plot([xlim_min, xlim_max], [xlim_min, xlim_max], 'r-', 
               alpha=0.8, linewidth=2, label='Perfect correlation')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Event Count')
        
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(xlim_min, xlim_max)
    
    ax.set_xlabel(f'True {var_info["label"]}')
    ax.set_ylabel(f'Reconstructed {var_info["label"]}')
    ax.set_title('Monte Carlo (Perfect Correlation)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    # Plot reconstruction methods
    for method in available_methods:
        if plot_idx >= rows * cols:
            break
            
        row = plot_idx // cols
        col = plot_idx % cols
        ax = axes[row, col]
        
        recon_values = method_data[method]
        
        # Align data lengths
        min_len = min(len(mc_values), len(recon_values))
        if min_len == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{METHOD_NAMES[method]} vs Monte Carlo', fontsize=12)
            plot_idx += 1
            continue
            
        x_data = mc_values.iloc[:min_len].values
        y_data = recon_values.iloc[:min_len].values
        
        # Remove NaN values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        
        if len(x_data) == 0:
            ax.text(0.5, 0.5, 'No valid data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{METHOD_NAMES[method]} vs Monte Carlo', fontsize=12)
            plot_idx += 1
            continue
        
        # Sample data if too many points
        if len(x_data) > max_points:
            idx = np.random.choice(len(x_data), max_points, replace=False)
            x_data = x_data[idx]
            y_data = y_data[idx]
        
        # Get robust limits for better visualization
        xlim_min, xlim_max, ylim_min, ylim_max = get_robust_limits(x_data, y_data)
        
        # Filter data to robust limits
        mask = ((x_data >= xlim_min) & (x_data <= xlim_max) & 
                (y_data >= ylim_min) & (y_data <= ylim_max))
        x_plot = x_data[mask]
        y_plot = y_data[mask]
        
        if len(x_plot) > 10:  # Need minimum points for histogram
            # Create 2D histogram
            hist, xedges, yedges = np.histogram2d(x_plot, y_plot, bins=bins, 
                                                range=[[xlim_min, xlim_max], [ylim_min, ylim_max]])
            
            # Plot histogram with logarithmic color scale for better contrast
            im = ax.imshow(hist.T, origin='lower', extent=[xlim_min, xlim_max, ylim_min, ylim_max], 
                          cmap='plasma', aspect='auto', interpolation='nearest',
                          norm=colors.LogNorm(vmin=max(1, hist.min()), vmax=hist.max()))
            
            # Add perfect reconstruction line (y=x)
            ax.plot([xlim_min, xlim_max], [xlim_min, xlim_max], 'r--', 
                   alpha=0.8, linewidth=2, label='y=x')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Event Count (log scale)')
            
            # Set limits
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            
            # Calculate correlation coefficient
            if len(x_plot) > 1:
                corr_coef = np.corrcoef(x_plot, y_plot)[0, 1]
                if np.isfinite(corr_coef):
                    ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
                else:
                    ax.text(0.05, 0.95, 'r = N/A', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top')
        
        ax.set_xlabel(f'True {var_info["label"]}')
        ax.set_ylabel(f'Reconstructed {var_info["label"]}')
        ax.set_title(f'{METHOD_NAMES[method]} vs Monte Carlo', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_combined_2d_comparison(kinematics, variable, figsize=(24, 16), bins=100):
    """
    Create a comprehensive comparison showing both scatter and 2D histogram for best method
    """
    if kinematics not in combined_data or 'recon' not in combined_data[kinematics]:
        return None
    
    data = combined_data[kinematics]['recon']
    var_info = VARIABLES[variable]
    mc_col = var_info['mc_col']
    
    if mc_col not in data.columns:
        return None
    
    mc_values = data[mc_col].dropna()
    
    # Find the best method (highest correlation)
    best_method = None
    best_corr = -1
    best_data = None
    
    for recon_col in var_info['recon_cols']:
        if recon_col in data.columns:
            method_name = recon_col.split('_')[0]
            recon_values = data[recon_col].dropna()
            
            # Align data
            min_len = min(len(mc_values), len(recon_values))
            if min_len > 1:
                x_data = mc_values.iloc[:min_len].values
                y_data = recon_values.iloc[:min_len].values
                
                valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]
                
                if len(x_clean) > 1:
                    corr_coef = np.corrcoef(x_clean, y_clean)[0, 1]
                    if np.isfinite(corr_coef) and abs(corr_coef) > abs(best_corr):
                        best_corr = corr_coef
                        best_method = method_name
                        best_data = (x_clean, y_clean)
    
    if best_method is None or best_data is None:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Detailed Analysis: {variable} - Best Method: {METHOD_NAMES[best_method]}\n'
                 f'Kinematics: {kinematics}, Correlation: {best_corr:.3f}', 
                 fontsize=16, y=0.95)
    
    x_data, y_data = best_data
    
    # Get robust limits
    xlim_min, xlim_max, ylim_min, ylim_max = get_robust_limits(x_data, y_data)
    
    # Filter data
    mask = ((x_data >= xlim_min) & (x_data <= xlim_max) & 
            (y_data >= ylim_min) & (y_data <= ylim_max))
    x_plot = x_data[mask]
    y_plot = y_data[mask]
    
    # 1. Scatter plot (subsampled)
    if len(x_plot) > 5000:
        idx = np.random.choice(len(x_plot), 5000, replace=False)
        x_scatter = x_plot[idx]
        y_scatter = y_plot[idx]
    else:
        x_scatter = x_plot
        y_scatter = y_plot
    
    ax1.scatter(x_scatter, y_scatter, alpha=0.6, s=1, c='blue', rasterized=True)
    ax1.plot([xlim_min, xlim_max], [xlim_min, xlim_max], 'r--', alpha=0.8, linewidth=2)
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.set_xlabel(f'True {var_info["label"]}')
    ax1.set_ylabel(f'Reconstructed {var_info["label"]}')
    ax1.set_title('Scatter Plot')
    ax1.grid(True, alpha=0.3)
    
    # 2. 2D Histogram (log scale)
    hist, xedges, yedges = np.histogram2d(x_plot, y_plot, bins=bins, 
                                        range=[[xlim_min, xlim_max], [ylim_min, ylim_max]])
    
    im2 = ax2.imshow(hist.T, origin='lower', extent=[xlim_min, xlim_max, ylim_min, ylim_max], 
                    cmap='plasma', aspect='auto', interpolation='nearest',
                    norm=colors.LogNorm(vmin=max(1, hist.min()), vmax=hist.max()))
    ax2.plot([xlim_min, xlim_max], [xlim_min, xlim_max], 'r--', alpha=0.8, linewidth=2)
    ax2.set_xlim(xlim_min, xlim_max)
    ax2.set_ylim(ylim_min, ylim_max)
    ax2.set_xlabel(f'True {var_info["label"]}')
    ax2.set_ylabel(f'Reconstructed {var_info["label"]}')
    ax2.set_title('2D Histogram (Log Scale)')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Event Count (log)')
    
    # 3. Residuals (y - x)
    residuals = y_plot - x_plot
    ax3.hist(residuals, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(np.mean(residuals), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(residuals):.3f}')
    ax3.axvline(np.median(residuals), color='purple', linestyle='-', linewidth=2, 
               label=f'Median: {np.median(residuals):.3f}')
    ax3.set_xlabel(f'Residuals (Reconstructed - True {var_info["label"]})')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Residual Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relative residuals ((y - x) / x) for variables where x > 0
    if np.all(x_plot > 0):
        rel_residuals = (y_plot - x_plot) / x_plot * 100
        ax4.hist(rel_residuals, bins=100, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(np.mean(rel_residuals), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(rel_residuals):.2f}%')
        ax4.axvline(np.median(rel_residuals), color='purple', linestyle='-', linewidth=2, 
                   label=f'Median: {np.median(rel_residuals):.2f}%')
        ax4.set_xlabel(f'Relative Residuals (% of True {var_info["label"]})')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Relative Residual Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # For variables that can be negative or zero, show fractional residuals differently
        # Use fractional residuals relative to the range
        data_range = xlim_max - xlim_min
        frac_residuals = (y_plot - x_plot) / data_range * 100
        ax4.hist(frac_residuals, bins=100, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(np.mean(frac_residuals), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(frac_residuals):.2f}%')
        ax4.set_xlabel(f'Fractional Residuals (% of {var_info["label"]} range)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Fractional Residual Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# MAIN EXECUTION
if __name__ == "__main__":
    # Configuration
    SAVE_TO_PDF = True
    SHOW_PLOTS = False
    PDF_FILENAME = f"DIS_2D_Histograms_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Find available kinematics with reconstruction data
    available_kinematics = []
    for kinematics in combined_data:
        if 'recon' in combined_data[kinematics]:
            available_kinematics.append(kinematics)
    
    if not available_kinematics:
        print("No reconstruction data found!")
    else:
        print(f"Available kinematics: {available_kinematics}")
        
        # Sort kinematics for consistent ordering
        available_kinematics.sort()
        
        # Initialize PDF if saving
        if SAVE_TO_PDF:
            pdf_pages = PdfPages(PDF_FILENAME)
            print(f"Saving plots to: {PDF_FILENAME}")
        
        try:
            # Create title page
            if SAVE_TO_PDF:
                fig_title = plt.figure(figsize=(8.5, 11))
                fig_title.suptitle('Deep Inelastic Scattering 2D Histogram Analysis Report', 
                                  fontsize=20, y=0.85)
                
                ax_title = fig_title.add_subplot(111)
                ax_title.axis('off')
                
                summary_text = f"""
                Available Kinematics: {', '.join(available_kinematics)}
                
                Analysis includes:
                • 2D histograms showing event density distributions
                • Logarithmic color scaling for better contrast
                • Robust outlier handling and axis limits
                • Detailed analysis of best-performing methods
                • Residual and relative residual distributions
                
                Variables analyzed:
                • X (Bjorken scaling variable)
                • Q² (momentum transfer squared)  
                • Y (inelasticity)
                • ν (energy transfer)
                • W (invariant mass)
                
                Reconstruction methods:
                • Double Angle (DA)
                • Electron Method
                • Jacquet-Blondel (JB) 
                • Machine Learning (ML)
                • Sigma Method
                
                Improvements in 2D histogram version:
                • Event density visualization with log color scale
                • Clear identification of high-concentration regions
                • Systematic bias identification through residual analysis
                • Better handling of overlapping data points
                • Quantitative bias and spread measurements
                
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                ax_title.text(0.1, 0.75, summary_text, fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
                
                pdf_pages.savefig(fig_title, bbox_inches='tight')
                plt.close(fig_title)
            
            plot_count = 0
            
            # Create plots for each kinematics
            for kinematics in available_kinematics:
                print(f"\nProcessing kinematics: {kinematics}")
                
                # Create section divider
                if SAVE_TO_PDF:
                    fig_divider = plt.figure(figsize=(8.5, 2))
                    fig_divider.suptitle(f'Kinematics: {kinematics}', fontsize=24, y=0.6)
                    ax_divider = fig_divider.add_subplot(111)
                    ax_divider.axis('off')
                    
                    # Add kinematics info
                    data_info = ""
                    if kinematics in combined_data and 'recon' in combined_data[kinematics]:
                        n_events = len(combined_data[kinematics]['recon'])
                        data_info = f"Number of events: {n_events:,}"
                    
                    ax_divider.text(0.5, 0.3, data_info, fontsize=14, ha='center',
                                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
                    
                    pdf_pages.savefig(fig_divider, bbox_inches='tight')
                    plt.close(fig_divider)
                    plot_count += 1
                
                # Create 2D histogram plots for each variable
                for variable in ['X', 'Q2', 'Y', 'NU', 'W']:
                    print(f"  - Creating 2D histograms for {variable}")
                    
                    # Create main 2D histogram comparison
                    fig = create_2d_histogram_plots(kinematics, variable)
                    if fig:
                        if SAVE_TO_PDF:
                            pdf_pages.savefig(fig, bbox_inches='tight')
                        if SHOW_PLOTS:
                            plt.show()
                        else:
                            plt.close(fig)
                        plot_count += 1
                    
                    # Create detailed analysis for best method
                    print(f"  - Creating detailed analysis for {variable}")
                    fig_detailed = create_combined_2d_comparison(kinematics, variable)
                    if fig_detailed:
                        if SAVE_TO_PDF:
                            pdf_pages.savefig(fig_detailed, bbox_inches='tight')
                        if SHOW_PLOTS:
                            plt.show()
                        else:
                            plt.close(fig_detailed)
                        plot_count += 1
                
        finally:
            # Close PDF
            if SAVE_TO_PDF:
                pdf_pages.close()
                print(f"\n✓ Successfully saved {plot_count} plots to {PDF_FILENAME}")
                print(f"  File size: {os.path.getsize(PDF_FILENAME) / (1024*1024):.1f} MB")
                print(f"  You can now open {PDF_FILENAME} to view all 2D histogram plots!")

# Cleanup
if os.path.exists(extract_root):
    shutil.rmtree(extract_root)

print("\nDone! Created 2D histogram analysis plots showing event concentration patterns.")