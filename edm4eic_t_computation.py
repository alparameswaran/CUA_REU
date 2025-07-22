# EDM4eic ROOT File Analysis for Mandelstam -t Calculation
# 
# This script processes .edm4eic.root files containing k_lambda events
# and calculates Mandelstam -t values from proton and lambda 4-vectors.
#
# Key features:
# - Extracts beam protons using MCBeamProtons_objIdx indices
# - Extracts lambdas from MCParticles collection using PDG filtering
# - Calculates -t = -(p_beam_proton - p_lambda)^2 for MC particles
# - Processes all files from k_lambda_18x275_5000evt_001.edm4eic.root to _110.edm4eic.root
# - Outputs combined CSV and diagnostic plots

import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# === Updated Paths for EDM4eic files ===
# WSL paths - update this to your actual path structure
edm4eic_dir = "/mnt/c/Users/Sweetie Pie/Desktop/CUA REU/edm4eic"

# Physical constants and PDG codes
mass_lambda = 1.1156  # Lambda mass in GeV
mass_proton = 0.93827203  # Proton mass in GeV
PDG_PROTON = 2212  # Proton PDG code
PDG_LAMBDA = 3122  # Lambda PDG code

def mandelstam_t(p_proton, p_lambda):
    q = p_proton - p_lambda
    return q[0]**2 - np.sum(q[1:]**2)

def get_mc_particles_4vectors(tree, n_events, pdg_filter=None):
    """
    Extract 4-vectors from MCParticles collection
    
    Args:
        tree: uproot tree object
        n_events: number of events to process (from main function)
        pdg_filter: PDG code to filter particles (e.g., 2212 for proton, 3122 for lambda)
    
    Returns:
        four_vectors: array of 4-vectors [E, px, py, pz] for each event
        selected_indices: list of particle indices used for each event
    """
    try:
        # Read momentum components
        px = tree["MCParticles.momentum.x"].array(library="np")
        py = tree["MCParticles.momentum.y"].array(library="np")
        pz = tree["MCParticles.momentum.z"].array(library="np")
        
        # Read mass and PDG codes
        mass = tree["MCParticles.mass"].array(library="np")
        pdg = tree["MCParticles.PDG"].array(library="np")
        
        # Ensure we don't exceed available data
        actual_events = min(n_events, len(px))
        

        
        # Handle jagged arrays (multiple particles per event)
        four_vectors = []
        selected_indices = []
        
        for event_idx in range(n_events):
            if event_idx < actual_events:
                event_px = px[event_idx]
                event_py = py[event_idx]
                event_pz = pz[event_idx]
                event_mass = mass[event_idx]
                event_pdg = pdg[event_idx]
                
                if pdg_filter is not None:
                    # Filter by PDG code
                    mask = event_pdg == pdg_filter
                    if np.any(mask):
                        idx = np.where(mask)[0][0]  # Take first matching particle
                        
                        # Calculate energy: E = sqrt(p^2 + m^2)
                        p_squared = event_px[idx]**2 + event_py[idx]**2 + event_pz[idx]**2
                        energy = np.sqrt(p_squared + event_mass[idx]**2)
                        
                        four_vectors.append(np.array([energy, event_px[idx], event_py[idx], event_pz[idx]]))
                        selected_indices.append(idx)
                    else:
                        four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        selected_indices.append(-1)
                else:
                    # Take first particle in each event
                    if len(event_px) > 0:
                        # Calculate energy: E = sqrt(p^2 + m^2)
                        p_squared = event_px[0]**2 + event_py[0]**2 + event_pz[0]**2
                        energy = np.sqrt(p_squared + event_mass[0]**2)
                        
                        four_vectors.append(np.array([energy, event_px[0], event_py[0], event_pz[0]]))
                        selected_indices.append(0)
                    else:
                        four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        selected_indices.append(-1)
            else:
                # Pad with NaN for missing events
                four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                selected_indices.append(-1)
        
        return np.array(four_vectors), selected_indices
        
    except Exception as e:
        print(f"Error extracting MC particles: {e}")
        # Return arrays of correct length filled with NaN
        four_vectors = [np.array([np.nan, np.nan, np.nan, np.nan]) for _ in range(n_events)]
        selected_indices = [-1] * n_events
        return np.array(four_vectors), selected_indices

def get_beam_protons_4vectors(tree, n_events):
    """
    Get beam protons using MCBeamProtons_objIdx indices
    
    Args:
        tree: uproot tree object
        n_events: number of events to process (from main function)
    
    Returns:
        four_vectors: array of 4-vectors [E, px, py, pz] for beam protons
        selected_indices: list of MC particle indices used for each event
    """
    try:
        # Check if the branch exists
        branch_name = "MCBeamProtons_objIdx.index"
        if branch_name not in tree:
            print(f"  Warning: Branch '{branch_name}' not found in tree")
            print(f"  Available branches related to 'Beam' or 'Proton':")
            for branch in tree.keys():
                if 'beam' in branch.lower() or 'proton' in branch.lower():
                    print(f"    {branch}")
            # Return arrays of correct length filled with NaN
            four_vectors = [np.array([np.nan, np.nan, np.nan, np.nan]) for _ in range(n_events)]
            selected_indices = [-1] * n_events
            return np.array(four_vectors), selected_indices
        
        # Read beam proton indices
        proton_indices = tree[branch_name].array(library="np")
        
        # Debug: Check proton indices
        non_empty_indices = sum(1 for idx_array in proton_indices if len(idx_array) > 0)
        print(f"  Debug: Events with beam proton indices: {non_empty_indices}/{len(proton_indices)}")
        if non_empty_indices > 0:
            # Show first few non-empty index arrays
            for i, idx_array in enumerate(proton_indices[:10]):
                if len(idx_array) > 0:
                    print(f"    Event {i}: indices = {idx_array}")
                    break
        
        # Read MCParticles data
        px = tree["MCParticles.momentum.x"].array(library="np")
        py = tree["MCParticles.momentum.y"].array(library="np")
        pz = tree["MCParticles.momentum.z"].array(library="np")
        mass = tree["MCParticles.mass"].array(library="np")
        
        # Ensure we don't exceed available data
        actual_events = min(n_events, len(proton_indices), len(px))
        if actual_events < n_events:
            print(f"  Warning: Limited data available - using {actual_events} events out of {n_events}")
        
        four_vectors = []
        selected_indices = []
        
        for event_idx in range(n_events):
            if event_idx < actual_events:
                event_proton_indices = proton_indices[event_idx]
                event_px = px[event_idx]
                event_py = py[event_idx]
                event_pz = pz[event_idx]
                event_mass = mass[event_idx]
                
                if len(event_proton_indices) > 0 and len(event_px) > 0:
                    # Take first beam proton
                    proton_idx = event_proton_indices[0]
                    
                    # Check if index is valid
                    if 0 <= proton_idx < len(event_px):
                        # Calculate energy: E = sqrt(p^2 + m^2)
                        p_squared = event_px[proton_idx]**2 + event_py[proton_idx]**2 + event_pz[proton_idx]**2
                        energy = np.sqrt(p_squared + event_mass[proton_idx]**2)
                        
                        four_vectors.append(np.array([energy, event_px[proton_idx], event_py[proton_idx], event_pz[proton_idx]]))
                        selected_indices.append(proton_idx)
                    else:
                        four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        selected_indices.append(-1)
                else:
                    four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    selected_indices.append(-1)
            else:
                # Pad with NaN for missing events
                four_vectors.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                selected_indices.append(-1)
        
        return np.array(four_vectors), selected_indices
        
    except Exception as e:
        print(f"Error extracting beam protons: {e}")
        # Return arrays of correct length filled with NaN
        four_vectors = [np.array([np.nan, np.nan, np.nan, np.nan]) for _ in range(n_events)]
        selected_indices = [-1] * n_events
        return np.array(four_vectors), selected_indices

def process_edm4eic_files(max_files=None):
    """Process all EDM4eic files for 18x275 setting
    
    Args:
        max_files: Maximum number of files to process (None = all files)
    """
    
    # Find all available files
    file_pattern = os.path.join(edm4eic_dir, "k_lambda_18x275_5000evt_*.edm4eic.root")
    edm4eic_files = sorted(glob(file_pattern))
    
    if not edm4eic_files:
        print(f"No EDM4eic files found in {edm4eic_dir}")
        return None
    
    print(f"\nFound {len(edm4eic_files)} EDM4eic files")
    print("Files range:", os.path.basename(edm4eic_files[0]), "to", os.path.basename(edm4eic_files[-1]))
    
    # Limit number of files if specified
    if max_files is not None:
        edm4eic_files = edm4eic_files[:max_files]
        print(f"Processing first {len(edm4eic_files)} files...")
    else:
        print(f"Processing all {len(edm4eic_files)} files...")
    
    # Initialize lists to collect data
    all_data = []
    
    for i, filepath in enumerate(edm4eic_files):
        print(f"Processing file {i+1}/{len(edm4eic_files)}: {os.path.basename(filepath)}")
        
        try:
            with uproot.open(filepath) as file:
                # Use "events" tree (standard for EDM4eic)
                tree_name = "events"
                
                if tree_name not in file:
                    # Try alternative tree names
                    available_keys = [k for k in file.keys() if not k.startswith('_')]
                    if available_keys:
                        tree_name = available_keys[0].replace(';1', '')
                        print(f"  Using tree: {tree_name}")
                    else:
                        print(f"  No suitable tree found in {filepath}")
                        continue
                
                tree = file[tree_name]
                n_events = len(tree)
                
                # Extract segment number from filename
                import re
                match = re.search(r"5000evt_(\d+)\.edm4eic\.root", os.path.basename(filepath))
                if match:
                    segment = int(match.group(1))
                else:
                    segment = i + 1  # fallback
                
                # Calculate global event numbers
                global_event_start = (segment - 1) * 5000
                global_events = np.arange(global_event_start, global_event_start + n_events)
                
                print(f"  Events: {n_events}, Segment: {segment}")
                
                # Extract particle 4-vectors - pass n_events to ensure consistent array lengths
                print("  Extracting beam protons...")
                proton_4vec, proton_indices = get_beam_protons_4vectors(tree, n_events)
                
                print("  Extracting MC lambdas...")
                lambda_mc_4vec, lambda_mc_indices = get_mc_particles_4vectors(tree, n_events, pdg_filter=PDG_LAMBDA)
                
                # Calculate t values if we have the required vectors
                t_mc = np.full(n_events, np.nan)
                
                if proton_4vec is not None and lambda_mc_4vec is not None:
                    print("  Computing MC t values...")
                    
                    # Debug: Check how many valid protons and lambdas we have
                    valid_protons = np.sum(~np.isnan(proton_4vec[:, 0]))  # Check energy component
                    valid_lambdas = np.sum(~np.isnan(lambda_mc_4vec[:, 0]))  # Check energy component
                    print(f"    Valid beam protons: {valid_protons}/{n_events}")
                    print(f"    Valid lambdas: {valid_lambdas}/{n_events}")
                    
                    # Debug: Show a few sample values
                    if valid_protons > 0:
                        first_valid_proton_idx = np.where(~np.isnan(proton_4vec[:, 0]))[0][0]
                        print(f"    Sample beam proton 4-vec (event {first_valid_proton_idx}): {proton_4vec[first_valid_proton_idx]}")
                    
                    if valid_lambdas > 0:
                        first_valid_lambda_idx = np.where(~np.isnan(lambda_mc_4vec[:, 0]))[0][0]
                        print(f"    Sample lambda 4-vec (event {first_valid_lambda_idx}): {lambda_mc_4vec[first_valid_lambda_idx]}")
                    
                    valid_pairs = 0
                    for j in range(n_events):
                        if not (np.isnan(proton_4vec[j]).any() or np.isnan(lambda_mc_4vec[j]).any()):
                            # Compute -t (negative t as requested)
                            t_val = -mandelstam_t(proton_4vec[j], lambda_mc_4vec[j])
                            t_mc[j] = t_val
                            valid_pairs += 1
                    
                    print(f"    Events with both valid beam proton and lambda: {valid_pairs}/{n_events}")
                    
                    # Additional debugging for the first few valid events
                    if valid_pairs > 0:
                        print(f"    First few valid t values:")
                        valid_t_vals = t_mc[~np.isnan(t_mc)][:5]  # Show first 5 valid values
                        for k, t_val in enumerate(valid_t_vals):
                            print(f"      t[{k}] = {t_val:.6f} GeV²")
                else:
                    print("    No valid 4-vectors found - cannot compute t values")
                
                # Create DataFrame for this file - ensure all arrays have length n_events
                file_data = {
                    "global_event": global_events,
                    "event": np.arange(n_events),
                    "segment": [segment] * n_events,
                    "setting": ["18x275"] * n_events,
                    "t_mc": t_mc,
                }
                
                # Add 4-vector components if available
                if proton_4vec is not None:
                    file_data.update({
                        "E_beam_proton": proton_4vec[:, 0],
                        "px_beam_proton": proton_4vec[:, 1],
                        "py_beam_proton": proton_4vec[:, 2],
                        "pz_beam_proton": proton_4vec[:, 3],
                        "beam_proton_mc_index": proton_indices,
                    })
                
                if lambda_mc_4vec is not None:
                    file_data.update({
                        "E_lambda_mc": lambda_mc_4vec[:, 0],
                        "px_lambda_mc": lambda_mc_4vec[:, 1],
                        "py_lambda_mc": lambda_mc_4vec[:, 2],
                        "pz_lambda_mc": lambda_mc_4vec[:, 3],
                        "lambda_mc_index": lambda_mc_indices,
                    })
                
                # Debug: Print array lengths before creating DataFrame
                print(f"  Array lengths check:")
                for key, value in file_data.items():
                    if hasattr(value, '__len__'):
                        print(f"    {key}: {len(value)}")
                    else:
                        print(f"    {key}: scalar")
                
                # Print summary
                n_valid_mc = np.sum(~np.isnan(t_mc))
                print(f"  Valid MC t values: {n_valid_mc}/{n_events}")
                if n_valid_mc > 0:
                    print(f"  t range: [{np.nanmin(t_mc):.3f}, {np.nanmax(t_mc):.3f}] GeV²")
                
                all_data.append(pd.DataFrame(file_data))
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_data:
        # Combine all data
        result_df = pd.concat(all_data, ignore_index=True)
        
        # Save results
        output_file = "mandelstam_minus_t_analysis_edm4eic_18x275_beam_protons.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved combined data: {len(result_df)} events to {output_file}")
        
        # Create diagnostic plots
        create_diagnostic_plots(result_df)
        
        return result_df
    else:
        print("No data was successfully processed")
        return None

def create_diagnostic_plots(df):
    plt.figure(figsize=(15, 10))
    
    # Plot t distributions
    plt.subplot(2, 3, 1)
    t_mc_vals = df["t_mc"].dropna()
    
    if len(t_mc_vals) > 0:
        plt.hist(t_mc_vals, bins=50, alpha=0.7, label="MC Lambda vs Beam Proton", color='blue')
        plt.xlabel("-t [GeV²]")
        plt.ylabel("Counts")
        plt.title(f"-t distribution (18x275)\nN = {len(t_mc_vals)} events")
        plt.legend()
        plt.grid(True)
        
        # Add statistics
        mean_t = t_mc_vals.mean()
        std_t = t_mc_vals.std()
        plt.text(0.05, 0.95, f'Mean: {mean_t:.3f} GeV²\nStd: {std_t:.3f} GeV²', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot beam proton energy distribution
    plt.subplot(2, 3, 2)
    if "E_beam_proton" in df.columns:
        proton_E = df["E_beam_proton"].dropna()
        if len(proton_E) > 0:
            plt.hist(proton_E, bins=50, alpha=0.7, color='red')
            plt.xlabel("Beam Proton Energy [GeV]")
            plt.ylabel("Counts")
            plt.title("Beam Proton Energy")
            plt.grid(True)
        # Plot lambda energy distribution
        
    plt.subplot(2, 3, 3)
    if "E_lambda_mc" in df.columns:
        lambda_E = df["E_lambda_mc"].dropna()
        if len(lambda_E) > 0:
            plt.hist(lambda_E, bins=50, alpha=0.7, color='green')
            plt.xlabel("Lambda Energy [GeV]")
            plt.ylabel("Counts")
            plt.title("MC Lambda Energy")
            plt.grid(True)
    
    
    plt.tight_layout()
    plt.savefig("edm4eic_analysis_minus_t_diagnostics_18x275_beam_protons.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Configuration
    TEST_MODE = True  # Set to True to process only first 2 files for testing
    
    print("=== EDM4eic ROOT File Analysis ===")
    print("Calculating Mandelstam -t values from MC beam protons and lambdas")
    
    # Run the analysis
    max_files = 20 if TEST_MODE else None
    result_df = process_edm4eic_files(max_files=max_files)
    
    if result_df is not None:
        print("\nAnalysis completed successfully!")
        print(f"Total events processed: {len(result_df)}")
        print(f"Events with MC -t: {(~result_df['t_mc'].isna()).sum()}")
        
        # Show some statistics
        if (~result_df['t_mc'].isna()).sum() > 0:
            mc_t_vals = result_df['t_mc'].dropna()
            print(f"MC -t range: [{mc_t_vals.min():.3f}, {mc_t_vals.max():.3f}] GeV²")
            print(f"MC -t mean: {mc_t_vals.mean():.3f} ± {mc_t_vals.std():.3f} GeV²")
            
            # Show first few valid events
            valid_events = result_df[~result_df['t_mc'].isna()].head()
            print("\nFirst few valid events:")
            print(valid_events[['global_event', 't_mc', 'E_beam_proton', 'E_lambda_mc']].to_string())
    else:
        print("Analysis failed - please check the file paths and ROOT file structure")