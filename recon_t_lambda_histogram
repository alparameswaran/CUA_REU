# EDM4eic Hybrid Analysis: MC Beam Protons + Reconstructed Lambdas
# 
# BRILLIANT SOLUTION: Use MC beam protons (we know these work) with 
# reconstructed lambdas to calculate the EXACT SAME -t as MC analysis
#
# Formula: -t = -(p_MC_beam_proton - p_reconstructed_lambda)²
#
# This gives us:
# - Same physics observable as MC analysis  
# - Shows reconstructed lambda performance vs MC
# - Bypasses the "beam protons never detected" issue

import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

edm4eic_dir = "/mnt/c/Users/Sweetie Pie/Desktop/CUA REU/edm4eic"

PDG_LAMBDA = 3122

def mandelstam_t(p_proton, p_lambda):
    """Calculate Mandelstam t = (p_proton - p_lambda)²"""
    q = p_proton - p_lambda
    return q[0]**2 - np.sum(q[1:]**2)

def get_mc_beam_protons(tree, n_events):
    """
    Extract MC beam protons using the EXACT same method as your working MC analysis
    """
    try:
        # Use exact same branch as your MC code
        beam_proton_branch = "MCBeamProtons_objIdx.index"
        if beam_proton_branch not in tree:
            print(f"    ❌ MCBeamProtons branch not found")
            return None, None
        
        proton_indices = tree[beam_proton_branch].array(library="np")
        
        # Read MCParticles data (exact same as your MC code)
        px = tree["MCParticles.momentum.x"].array(library="np")
        py = tree["MCParticles.momentum.y"].array(library="np")
        pz = tree["MCParticles.momentum.z"].array(library="np")
        mass = tree["MCParticles.mass"].array(library="np")
        
        actual_events = min(n_events, len(proton_indices), len(px))
        
        mc_proton_4vec = []
        mc_proton_indices = []
        
        valid_beam_protons = 0
        
        for event_idx in range(n_events):
            if event_idx < actual_events:
                event_proton_indices = proton_indices[event_idx]
                event_px = px[event_idx]
                event_py = py[event_idx]
                event_pz = pz[event_idx]
                event_mass = mass[event_idx]
                
                if len(event_proton_indices) > 0 and len(event_px) > 0:
                    proton_idx = event_proton_indices[0]  # First beam proton
                    
                    if 0 <= proton_idx < len(event_px):
                        # Calculate energy: E = sqrt(p² + m²) - same as your MC code
                        p_squared = event_px[proton_idx]**2 + event_py[proton_idx]**2 + event_pz[proton_idx]**2
                        energy = np.sqrt(p_squared + event_mass[proton_idx]**2)
                        
                        mc_proton_4vec.append(np.array([energy, event_px[proton_idx], 
                                                      event_py[proton_idx], event_pz[proton_idx]]))
                        mc_proton_indices.append(proton_idx)
                        valid_beam_protons += 1
                    else:
                        mc_proton_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        mc_proton_indices.append(-1)
                else:
                    mc_proton_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    mc_proton_indices.append(-1)
            else:
                mc_proton_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                mc_proton_indices.append(-1)
        
        print(f"    ✓ MC beam protons: {valid_beam_protons}/{n_events}")
        
        # Show sample to verify it matches your MC
        if valid_beam_protons > 0:
            first_valid_idx = np.where(~np.isnan([p[0] for p in mc_proton_4vec]))[0][0]
            sample_proton = mc_proton_4vec[first_valid_idx]
            print(f"    Sample MC beam proton: [{sample_proton[0]:.1f}, {sample_proton[1]:.1f}, {sample_proton[2]:.3f}, {sample_proton[3]:.1f}]")
        
        return np.array(mc_proton_4vec), mc_proton_indices
        
    except Exception as e:
        print(f"    Error extracting MC beam protons: {e}")
        return None, None

def get_reconstructed_lambdas_best(tree, n_events):
    """
    Get reconstructed lambdas using the best available method
    """
    # Method 1: Try ReconstructedFarForwardZDCLambdas (we know these exist)
    try:
        zdc_energy_branch = "ReconstructedFarForwardZDCLambdas.energy"
        if zdc_energy_branch in tree:
            print(f"    ✓ Using ReconstructedFarForwardZDCLambdas")
            
            px = tree["ReconstructedFarForwardZDCLambdas.momentum.x"].array(library="np")
            py = tree["ReconstructedFarForwardZDCLambdas.momentum.y"].array(library="np")
            pz = tree["ReconstructedFarForwardZDCLambdas.momentum.z"].array(library="np")
            energy = tree["ReconstructedFarForwardZDCLambdas.energy"].array(library="np")
            
            actual_events = min(n_events, len(energy))
            
            reco_lambda_4vec = []
            reco_lambda_indices = []
            valid_lambdas = 0
            
            for event_idx in range(n_events):
                if event_idx < actual_events:
                    event_px = px[event_idx]
                    event_py = py[event_idx]
                    event_pz = pz[event_idx]
                    event_energy = energy[event_idx]
                    
                    if len(event_energy) > 0:
                        # Take highest energy lambda
                        best_idx = np.argmax(event_energy)
                        reco_lambda_4vec.append(np.array([event_energy[best_idx], 
                                                        event_px[best_idx], 
                                                        event_py[best_idx], 
                                                        event_pz[best_idx]]))
                        reco_lambda_indices.append(best_idx)
                        valid_lambdas += 1
                    else:
                        reco_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        reco_lambda_indices.append(-1)
                else:
                    reco_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    reco_lambda_indices.append(-1)
            
            print(f"    ✓ Reconstructed ZDC lambdas: {valid_lambdas}/{n_events}")
            
            # Show sample
            if valid_lambdas > 0:
                first_valid_idx = np.where(~np.isnan([p[0] for p in reco_lambda_4vec]))[0][0]
                sample_lambda = reco_lambda_4vec[first_valid_idx]
                print(f"    Sample reco lambda: [{sample_lambda[0]:.1f}, {sample_lambda[1]:.1f}, {sample_lambda[2]:.3f}, {sample_lambda[3]:.1f}]")
            
            return np.array(reco_lambda_4vec), reco_lambda_indices
    except Exception as e:
        print(f"    Error with ZDC lambdas: {e}")
    
    # Method 2: Try general ReconstructedParticles with PDG filtering
    try:
        reco_energy_branch = "ReconstructedParticles.energy"
        reco_pdg_branch = "ReconstructedParticles.PDG"
        
        if reco_energy_branch in tree and reco_pdg_branch in tree:
            print(f"    Trying ReconstructedParticles with lambda PDG...")
            
            px = tree["ReconstructedParticles.momentum.x"].array(library="np")
            py = tree["ReconstructedParticles.momentum.y"].array(library="np")
            pz = tree["ReconstructedParticles.momentum.z"].array(library="np")
            energy = tree["ReconstructedParticles.energy"].array(library="np")
            pdg = tree["ReconstructedParticles.PDG"].array(library="np")
            
            actual_events = min(n_events, len(energy))
            
            reco_lambda_4vec = []
            reco_lambda_indices = []
            valid_lambdas = 0
            
            for event_idx in range(n_events):
                if event_idx < actual_events:
                    event_px = px[event_idx]
                    event_py = py[event_idx]
                    event_pz = pz[event_idx]
                    event_energy = energy[event_idx]
                    event_pdg = pdg[event_idx]
                    
                    lambda_indices = np.where(event_pdg == PDG_LAMBDA)[0]
                    if len(lambda_indices) > 0:
                        # Take highest energy lambda
                        best_idx = lambda_indices[np.argmax(event_energy[lambda_indices])]
                        reco_lambda_4vec.append(np.array([event_energy[best_idx], 
                                                        event_px[best_idx], 
                                                        event_py[best_idx], 
                                                        event_pz[best_idx]]))
                        reco_lambda_indices.append(best_idx)
                        valid_lambdas += 1
                    else:
                        reco_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        reco_lambda_indices.append(-1)
                else:
                    reco_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    reco_lambda_indices.append(-1)
            
            print(f"    ✓ General reconstructed lambdas: {valid_lambdas}/{n_events}")
            return np.array(reco_lambda_4vec), reco_lambda_indices
            
    except Exception as e:
        print(f"    Error with general lambdas: {e}")
    
    # Fallback: no lambdas found
    print(f"    ❌ No reconstructed lambdas found")
    reco_lambda_4vec = [np.array([np.nan, np.nan, np.nan, np.nan]) for _ in range(n_events)]
    reco_lambda_indices = [-1] * n_events
    return np.array(reco_lambda_4vec), reco_lambda_indices

def get_mc_lambdas_for_comparison(tree, n_events):
    """
    Get MC lambdas using same method as your MC analysis for direct comparison
    """
    try:
        # Read MCParticles data
        px = tree["MCParticles.momentum.x"].array(library="np")
        py = tree["MCParticles.momentum.y"].array(library="np")
        pz = tree["MCParticles.momentum.z"].array(library="np")
        mass = tree["MCParticles.mass"].array(library="np")
        pdg = tree["MCParticles.PDG"].array(library="np")
        
        actual_events = min(n_events, len(px))
        
        mc_lambda_4vec = []
        mc_lambda_indices = []
        valid_mc_lambdas = 0
        
        for event_idx in range(n_events):
            if event_idx < actual_events:
                event_px = px[event_idx]
                event_py = py[event_idx]
                event_pz = pz[event_idx]
                event_mass = mass[event_idx]
                event_pdg = pdg[event_idx]
                
                lambda_mask = event_pdg == PDG_LAMBDA
                if np.any(lambda_mask):
                    idx = np.where(lambda_mask)[0][0]  # First lambda
                    p_squared = event_px[idx]**2 + event_py[idx]**2 + event_pz[idx]**2
                    energy = np.sqrt(p_squared + event_mass[idx]**2)
                    mc_lambda_4vec.append(np.array([energy, event_px[idx], event_py[idx], event_pz[idx]]))
                    mc_lambda_indices.append(idx)
                    valid_mc_lambdas += 1
                else:
                    mc_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                    mc_lambda_indices.append(-1)
            else:
                mc_lambda_4vec.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                mc_lambda_indices.append(-1)
        
        print(f"    ✓ MC lambdas (for comparison): {valid_mc_lambdas}/{n_events}")
        return np.array(mc_lambda_4vec), mc_lambda_indices
        
    except Exception as e:
        print(f"    Error extracting MC lambdas: {e}")
        return None, None

def process_files_hybrid(max_files=None):
    """Process files using hybrid MC beam proton + reconstructed lambda approach"""
    
    file_pattern = os.path.join(edm4eic_dir, "k_lambda_18x275_5000evt_*.edm4eic.root")
    edm4eic_files = sorted(glob(file_pattern))
    
    if not edm4eic_files:
        print(f"No files found in {edm4eic_dir}")
        return None
    
    print(f"\nFound {len(edm4eic_files)} files")
    
    if max_files is not None:
        edm4eic_files = edm4eic_files[:max_files]
        print(f"Processing first {len(edm4eic_files)} files...")
    
    all_data = []
    total_valid_hybrid_t = 0
    total_valid_mc_t = 0
    
    for i, filepath in enumerate(edm4eic_files):
        print(f"\nProcessing file {i+1}/{len(edm4eic_files)}: {os.path.basename(filepath)}")
        
        try:
            with uproot.open(filepath) as file:
                tree_name = "events"
                
                if tree_name not in file:
                    available_keys = [k for k in file.keys() if not k.startswith('_')]
                    if available_keys:
                        tree_name = available_keys[0].replace(';1', '')
                    else:
                        print(f"  No suitable tree found")
                        continue
                
                tree = file[tree_name]
                n_events = len(tree)
                
                # Extract segment number
                import re
                match = re.search(r"5000evt_(\d+)\.edm4eic\.root", os.path.basename(filepath))
                segment = int(match.group(1)) if match else i + 1
                
                global_event_start = (segment - 1) * 5000
                global_events = np.arange(global_event_start, global_event_start + n_events)
                
                print(f"  Events: {n_events}, Segment: {segment}")
                
                # Get MC beam protons (same as your working MC analysis)
                print("  Extracting MC beam protons...")
                mc_proton_4vec, mc_proton_indices = get_mc_beam_protons(tree, n_events)
                
                if mc_proton_4vec is None:
                    print(f"  Failed to get MC beam protons")
                    continue
                
                # Get reconstructed lambdas
                print("  Extracting reconstructed lambdas...")
                reco_lambda_4vec, reco_lambda_indices = get_reconstructed_lambdas_best(tree, n_events)
                
                # Get MC lambdas for comparison
                print("  Extracting MC lambdas (for comparison)...")
                mc_lambda_4vec, mc_lambda_lambda_indices = get_mc_lambdas_for_comparison(tree, n_events)
                
                # Check overlaps
                mc_proton_events = set(np.where(~np.isnan(mc_proton_4vec[:, 0]))[0])
                reco_lambda_events = set(np.where(~np.isnan(reco_lambda_4vec[:, 0]))[0])
                mc_lambda_events = set(np.where(~np.isnan(mc_lambda_4vec[:, 0]))[0]) if mc_lambda_4vec is not None else set()
                
                hybrid_overlap = mc_proton_events.intersection(reco_lambda_events)
                mc_overlap = mc_proton_events.intersection(mc_lambda_events)
                
                print(f"    MC beam protons: {len(mc_proton_events)}")
                print(f"    Reconstructed lambdas: {len(reco_lambda_events)}")
                print(f"    MC lambdas: {len(mc_lambda_events)}")
                print(f"    Hybrid overlap (MC proton + Reco lambda): {len(hybrid_overlap)}")
                print(f"    MC overlap (MC proton + MC lambda): {len(mc_overlap)}")
                
                # Calculate -t values
                t_hybrid = np.full(n_events, np.nan)  # MC proton + Reco lambda
                t_mc = np.full(n_events, np.nan)      # MC proton + MC lambda (your original)
                
                # Hybrid -t calculation
                if len(hybrid_overlap) > 0:
                    print(f"    ✓ HYBRID SUCCESS! Computing -t with MC protons + Reco lambdas...")
                    
                    hybrid_valid = 0
                    for j in range(n_events):
                        if not (np.isnan(mc_proton_4vec[j]).any() or np.isnan(reco_lambda_4vec[j]).any()):
                            t_val = -mandelstam_t(mc_proton_4vec[j], reco_lambda_4vec[j])
                            t_hybrid[j] = t_val
                            hybrid_valid += 1
                    
                    print(f"    Hybrid -t calculated for {hybrid_valid} events")
                    total_valid_hybrid_t += hybrid_valid
                    
                    if hybrid_valid > 0:
                        valid_hybrid_t = t_hybrid[~np.isnan(t_hybrid)][:5]
                        print(f"    Sample hybrid -t: {[f'{val:.4f}' for val in valid_hybrid_t]} GeV²")
                
                # MC -t calculation (for comparison with your original results)
                if len(mc_overlap) > 0 and mc_lambda_4vec is not None:
                    print(f"    ✓ MC comparison: Computing -t with MC protons + MC lambdas...")
                    
                    mc_valid = 0
                    for j in range(n_events):
                        if not (np.isnan(mc_proton_4vec[j]).any() or np.isnan(mc_lambda_4vec[j]).any()):
                            t_val = -mandelstam_t(mc_proton_4vec[j], mc_lambda_4vec[j])
                            t_mc[j] = t_val
                            mc_valid += 1
                    
                    print(f"    MC -t calculated for {mc_valid} events")
                    total_valid_mc_t += mc_valid
                    
                    if mc_valid > 0:
                        valid_mc_t = t_mc[~np.isnan(t_mc)][:5]
                        print(f"    Sample MC -t: {[f'{val:.4f}' for val in valid_mc_t]} GeV²")
                
                # Create DataFrame
                file_data = {
                    "global_event": global_events,
                    "event": np.arange(n_events),
                    "segment": [segment] * n_events,
                    "setting": ["18x275"] * n_events,
                    "t_hybrid": t_hybrid,  # MC proton + Reco lambda  
                    "t_mc": t_mc,          # MC proton + MC lambda
                    "E_mc_proton": mc_proton_4vec[:, 0],
                    "px_mc_proton": mc_proton_4vec[:, 1],
                    "py_mc_proton": mc_proton_4vec[:, 2],
                    "pz_mc_proton": mc_proton_4vec[:, 3],
                    "E_reco_lambda": reco_lambda_4vec[:, 0],
                    "px_reco_lambda": reco_lambda_4vec[:, 1],
                    "py_reco_lambda": reco_lambda_4vec[:, 2],
                    "pz_reco_lambda": reco_lambda_4vec[:, 3],
                    "E_mc_lambda": mc_lambda_4vec[:, 0] if mc_lambda_4vec is not None else [np.nan] * n_events,
                    "px_mc_lambda": mc_lambda_4vec[:, 1] if mc_lambda_4vec is not None else [np.nan] * n_events,
                    "py_mc_lambda": mc_lambda_4vec[:, 2] if mc_lambda_4vec is not None else [np.nan] * n_events,
                    "pz_mc_lambda": mc_lambda_4vec[:, 3] if mc_lambda_4vec is not None else [np.nan] * n_events,
                }
                
                print(f"  Hybrid -t values this file: {np.sum(~np.isnan(t_hybrid))}")
                print(f"  MC -t values this file: {np.sum(~np.isnan(t_mc))}")
                
                all_data.append(pd.DataFrame(file_data))
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        
        output_file = "edm4eic_18x275_hybrid_mc_proton_reco_lambda_analysis.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved data: {len(result_df)} events to {output_file}")
        print(f"Total valid reconstructed -t: {total_valid_hybrid_t}")
        print(f"Total valid MC -t: {total_valid_mc_t}")
        print("Note: Reconstructed -t = MC beam protons + Reconstructed lambdas")
        
        return result_df
    else:
        print("No data processed")
        return None

def create_reconstructed_t_plots(df):
    """Create plots focused on reconstructed -t results"""
    plt.figure(figsize=(15, 10))
    
    # Main plot: Reconstructed -t distribution
    plt.subplot(2, 3, 1)
    t_hybrid_vals = df["t_hybrid"].dropna()
    
    if len(t_hybrid_vals) > 0:
        plt.hist(t_hybrid_vals, bins=500, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel("-t [GeV²]")
        plt.ylabel("Counts")
        plt.title(f"Reconstructed -t distribution (18x275)\nN = {len(t_hybrid_vals)} events")
        plt.xlim(0.0, 0.045)  # Set x-axis range for -t
        plt.grid(True)
        
        # Add statistics
        mean_t = t_hybrid_vals.mean()
        std_t = t_hybrid_vals.std()
        plt.text(0.05, 0.95, f'Mean: {mean_t:.4f} GeV²\nStd: {std_t:.4f} GeV²', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No reconstructed\n-t values found', 
                transform=plt.gca().transAxes, ha='center', va='center', 
                fontsize=14, color='red')
        plt.title("Reconstructed -t distribution")
        plt.xlim(0.0, 0.045)  # Set x-axis range even if no data
    
    # MC Beam proton energy
    plt.subplot(2, 3, 2)
    mc_proton_E = df["E_mc_proton"].dropna()
    if len(mc_proton_E) > 0:
        plt.hist(mc_proton_E, bins=100, alpha=0.7, color='blue')
        plt.xlabel("MC Beam Proton Energy [GeV]")
        plt.ylabel("Counts")
        plt.title("Beam Proton Energy")
        plt.grid(True)
        
        plt.text(0.05, 0.95, f'Mean: {mc_proton_E.mean():.1f} GeV', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Reconstructed lambda energy
    plt.subplot(2, 3, 3)
    reco_lambda_E = df["E_reco_lambda"].dropna()
    if len(reco_lambda_E) > 0:
        plt.hist(reco_lambda_E, bins=100, alpha=0.7, color='orange')
        plt.xlabel("Reconstructed Lambda Energy [GeV]")
        plt.ylabel("Counts")
        plt.title("Reconstructed Lambda Energy")
        plt.xlim(250, 275)  # Set x-axis range for lambda energy
        plt.grid(True)
        
        plt.text(0.05, 0.95, f'Mean: {reco_lambda_E.mean():.1f} GeV', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    else:
        plt.xlabel("Reconstructed Lambda Energy [GeV]")
        plt.title("Reconstructed Lambda Energy")
        plt.xlim(250, 275)  # Set x-axis range even if no data
    
    # Energy comparison: Beam proton vs Lambda
    plt.subplot(2, 3, 4)
    if len(mc_proton_E) > 0 and len(reco_lambda_E) > 0:
        plt.hist(mc_proton_E, bins=60, alpha=0.7, label="Beam Proton", color='blue', density=True)
        plt.hist(reco_lambda_E, bins=60, alpha=0.7, label="Reco Lambda", color='orange', density=True)
        plt.xlabel("Energy [GeV]")
        plt.ylabel("Normalized Counts")
        plt.title("Energy Comparison")
        plt.legend()
        plt.grid(True)
        # Note: Not setting xlim here since this shows both beam proton and lambda energies which have very different ranges
    
    # Correlation: -t vs Lambda energy
    plt.subplot(2, 3, 5)
    valid_mask = ~(df["t_hybrid"].isna() | df["E_reco_lambda"].isna())
    if valid_mask.sum() > 0:
        plt.scatter(df.loc[valid_mask, "t_hybrid"], 
                   df.loc[valid_mask, "E_reco_lambda"], 
                   alpha=0.7, s=20, color='purple')
        plt.xlabel("Reconstructed -t [GeV²]")
        plt.ylabel("Lambda Energy [GeV]")
        plt.title("-t vs Lambda Energy")
        plt.xlim(0.0, 0.045)  # Set x-axis range for -t
        plt.ylim(250, 275)   # Set y-axis range for lambda energy
        plt.grid(True)
    else:
        plt.xlabel("Reconstructed -t [GeV²]")
        plt.ylabel("Lambda Energy [GeV]")
        plt.title("-t vs Lambda Energy")
        plt.xlim(0.0, 0.045)  # Set x-axis range even if no data
        plt.ylim(250, 275)   # Set y-axis range even if no data
    
    # Statistics and summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate key statistics
    total_events = len(df)
    valid_t = len(t_hybrid_vals)
    efficiency = (valid_t / total_events * 100) if total_events > 0 else 0
    
    mean_t = t_hybrid_vals.mean() if len(t_hybrid_vals) > 0 else 0
    std_t = t_hybrid_vals.std() if len(t_hybrid_vals) > 0 else 0
    min_t = t_hybrid_vals.min() if len(t_hybrid_vals) > 0 else 0
    max_t = t_hybrid_vals.max() if len(t_hybrid_vals) > 0 else 0
    
    beam_proton_mean = mc_proton_E.mean() if len(mc_proton_E) > 0 else 0
    lambda_mean = reco_lambda_E.mean() if len(reco_lambda_E) > 0 else 0
    
    summary_text = f"""RECONSTRUCTED -t ANALYSIS

Method: MC Beam Protons + Reconstructed Lambdas
Formula: -t = -(p_beam - p_reco_lambda)²

Results:
• Total events: {total_events}
• Valid -t calculations: {valid_t}
• Efficiency: {efficiency:.1f}%

-t Statistics:
• Mean: {mean_t:.4f} GeV²
• Std: {std_t:.4f} GeV²
• Range: [{min_t:.4f}, {max_t:.4f}] GeV²

Particle Energies:
• Beam proton: {beam_proton_mean:.1f} GeV
• Reco lambda: {lambda_mean:.1f} GeV

Status: {'SUCCESS' if valid_t > 0 else 'NO DATA'}
"""
    
    color = 'lightgreen' if valid_t > 0 else 'lightcoral'
    plt.text(0.05, 0.5, summary_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("reconstructed_minus_t_distribution_18x275.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== EDM4eic Reconstructed -t Analysis ===")
    print()
    print("🎯 Calculating reconstructed -t using:")
    print("   • MC beam protons (same as your working MC analysis)")  
    print("   • Reconstructed lambdas (shows detector performance)")
    print("   • Formula: -t = -(p_MC_beam - p_reco_lambda)²")
    print()
    print("This gives the same -t observable as MC but uses reconstructed lambdas!")
    
    max_files = None  # Process all files
    result_df = process_files_hybrid(max_files=max_files)
    
    if result_df is not None:
        print(f"\n{'='*80}")
        print("RECONSTRUCTED -t ANALYSIS RESULTS:")
        print(f"Total events processed: {len(result_df)}")
        
        valid_hybrid_t = (~result_df['t_hybrid'].isna()).sum()
        
        print(f"Valid reconstructed -t: {valid_hybrid_t}")
        
        if valid_hybrid_t > 0:
            t_mc_vals = result_df['t_mc'].dropna()
            t_hybrid_vals = result_df['t_hybrid'].dropna()
            
            print(f"✓ SUCCESS! Found {len(t_hybrid_vals)} hybrid -t values")
            print(f"MC -t range: [{t_mc_vals.min():.4f}, {t_mc_vals.max():.4f}] GeV²")
            print(f"Hybrid -t range: [{t_hybrid_vals.min():.4f}, {t_hybrid_vals.max():.4f}] GeV²")
            print(f"MC -t mean: {t_mc_vals.mean():.4f} ± {t_mc_vals.std():.4f} GeV²")
            print(f"Hybrid -t mean: {t_hybrid_vals.mean():.4f} ± {t_hybrid_vals.std():.4f} GeV²")
            
            # Key result: How well do they match?
            ratio = t_hybrid_vals.mean() / t_mc_vals.mean() if len(t_mc_vals) > 0 and t_mc_vals.mean() > 0 else float('inf')
            print(f"\n🎯 KEY RESULT - Ratio (Hybrid/MC): {ratio:.3f}")
            
            if 0.8 <= ratio <= 1.2:
                print("🏆 EXCELLENT: Hybrid -t matches MC -t very well!")
                print("   Reconstructed lambdas are high quality!")
            elif 0.5 <= ratio <= 2.0:
                print("✅ GOOD: Hybrid -t reasonably matches MC -t")
                print("   Some reconstruction effects but physically reasonable")
            else:
                print("⚠️  CAUTION: Significant difference between hybrid and MC")
                print("   Check lambda reconstruction quality")
            
            # Show events with both values
            both_valid = ~(result_df['t_mc'].isna() | result_df['t_hybrid'].isna())
            if both_valid.sum() > 0:
                print(f"\nFirst few events with both MC and Hybrid -t:")
                comparison_events = result_df[both_valid][['global_event', 't_mc', 't_hybrid', 'E_mc_lambda', 'E_reco_lambda']].head()
                print(comparison_events.to_string())
            
            # Create plots
            create_reconstructed_t_plots(result_df)
            
        else:
            print("❌ No reconstructed -t values found!")
            print("Check reconstructed lambda availability")
    else:
        print("Reconstructed -t analysis failed")
