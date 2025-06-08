import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

class ElectronProtonScattering:
    def __init__(self):
        self.alpha = 1/137.036 #Fine structure constant
        self.mp = 0.938272 #GeV
        self.me = 0.000511 #GeV
        self.hbarc = 0.197327 #units: GeV * fm

    def dipole_form_factor(self, Q2):
        Lambda2 = 0.71 #GeV^2 (dipole mass squared)
        return (1+Q2/Lambda2)**(-2)
    
    def rosenbluth_cross_section(self, E_initial, theta):
        """Rosenbluth formula for e-p elastic scattering differential cross section"""
        
        #Scattered electron energy (elastic kinematics)
        E_final = E_initial/(1+(2*E_initial/self.mp)*np.sin(theta/2)**2)

        #Momentum Transfer squared (Q^2)
        Q2 = 4*E_initial*E_final*np.sin(theta/2)**2

        #Rosenbluth section factor
        Rosen_0 = (self.alpha*np.cos(theta/2))**2 / (4*E_initial**2*np.sin(theta/2)**4)

        GE = self.dipole_form_factor(Q2)
        GM = 2.79*self.dipole_form_factor(Q2) #2.79 is proton magnetic moment

        tau = Q2 / (4 * self.mp**2)
        epsilon = 1/(1+2*(1+tau)*np.tan(theta/2)**2)

        #Rosenbluth separation formula
        cross_section = Rosen_0 * (E_final/E_initial) * (GE**2 + tau*GM**2)/(1+tau) * epsilon

        #Convert to millibarns
        return cross_section * (self.hbarc * 1000)**2 #mb/sr/GeV
    
    def rosenbluth_cross_section_custom(self, E_initial, theta, GE_func, GM_func):
        """Calculate Rosenbluth cross section with custom form factors"""
        
        #Scattered electron energy (elastic kinematics)
        E_final = E_initial/(1+(2*E_initial/self.mp)*np.sin(theta/2)**2)
        
        #Momentum Transfer squared (Q^2)
        Q2 = 4*E_initial*E_final*np.sin(theta/2)**2
        
        #Rosenbluth section factor
        Rosen_0 = (self.alpha*np.cos(theta/2))**2 / (4*E_initial**2*np.sin(theta/2)**4)
        
        #Use custom form factors
        GE = GE_func(Q2)
        GM = GM_func(Q2)
        tau = Q2 / (4 * self.mp**2)
        epsilon = 1/(1+2*(1+tau)*np.tan(theta/2)**2)
        
        #Rosenbluth separation
        cross_section = Rosen_0 * (E_final/E_initial) * (GE**2 + tau*GM**2)/(1+tau) * epsilon
        
        #Convert to millibarns
        return cross_section * (self.hbarc * 1000)**2
    
    def sample_theta_rosenbluth(self, E_beam, theta_min, theta_max):
        """Sample scattering angle using Rosenbluth cross section for importance sampling"""

        #Create cumulative distribution function
        n_points = 1000
        theta_array = np.linspace(theta_min, theta_max, n_points)

        #Compute unnormalized probabilities with Rosenbluth cross section times sin(theta)
        prob = np.array([self.rosenbluth_cross_section(E_beam, theta)*np.sin(theta) for theta in theta_array])

        cumulative_prob = np.cumsum(prob)
        #Divide by total sum to get relative probabilities
        cumulative_prob = cumulative_prob / cumulative_prob[-1]

        #Get random number and find corresponding theta
        r = random.random()
        i = np.searchsorted(cumulative_prob, r)

        #Edge cases
        if i >= len(theta_array):
            i = len(theta_array)-1
        elif i == 0:
            i = 1

        #Linear interpolation between points
        r1 = cumulative_prob[i-1]
        r2 = cumulative_prob[i]

        t1 = theta_array[i-1]
        t2 = theta_array[i]

        theta = t1 + (t2-t1)*(r-r1) / (r2 - r1)
        return theta

    def output_kinematics(self, E_initial, theta):
        """Full kinematics for elastic scattering"""

        #Scattered electron energy
        E_final = E_initial / (1+(2*E_initial/self.mp)*np.sin(theta/2)**2)

        #Scattered electron momentum
        p_final = np.sqrt(E_final**2 - self.me**2)

        #Momentum transfer
        Q2 = 4*E_initial*E_final*np.sin(theta/2)**2

        return E_final, p_final, Q2
    
    def generate_weights(self, E_initial, theta, theta_min, theta_max):
        #Generate proper Monte Carlo weights for importance sampling
        
        # Calculate the cross section that was used for sampling
        weight = self.rosenbluth_cross_section(E_initial, theta)
        
        return weight

    def generate_events(self, E_initial, theta_min, theta_max, n_events, true_GE=None, true_GM=None):
        #Generate Monte Carlo events with Rosenbluth importance sampling

        events = []

        for x in range(n_events):
            theta = self.sample_theta_rosenbluth(E_initial, theta_min, theta_max)

            # Calculate full kinematics
            E_final, p_final, Q2 = self.output_kinematics(E_initial, theta)

            # Calculate proper physics-based weight
            weight = self.generate_weights(E_initial, theta, theta_min, theta_max)

            # Sample azimuthal angle uniformly
            phi = 2*np.pi*np.random.uniform(-1, 1)

            # Create the event dictionary
            event = {
                'theta': theta,
                'phi': phi,
                'weight': weight,
                'E_scattered': E_final,
                'p_scattered': p_final,
                'Q2': Q2
            }

            events.append(event)

        print("Event generation complete")
        return events

    def add_detector_effects(self, events, E_initial):
        """Add realistic detector resolution effects to MC events"""
        
        # Typical detector resolutions for electron scattering experiments
        angle_resolution = 0.001  # ~0.06 degrees in radians
        energy_resolution_percent = 0.01  # 1% energy resolution
        
        for event in events:
            # Smear the scattering angle
            theta_measured = event['theta'] + np.random.normal(0, angle_resolution)
            
            # Smear the scattered energy
            energy_smearing = event['E_scattered'] * energy_resolution_percent
            E_measured = event['E_scattered'] + np.random.normal(0, energy_smearing)
            
            # Recalculate kinematics with smeared values
            # Make sure energy is physical
            E_measured = max(E_measured, self.me)  # Can't be less than electron rest mass
            
            # Store both true and measured values
            event['theta_measured'] = theta_measured
            event['E_measured'] = E_measured
            
            # Recalculate Q2 with measured values
            event['Q2_measured'] = 4 * E_initial * E_measured * np.sin(theta_measured/2)**2
        
        return events

    def calculate_experimental_cross_sections(self, events, E_initial, n_bins=15):
        """Calculate cross sections from binned MC events like a real experiment"""
        
        # Get the measured kinematics
        Q2_measured = np.array([e['Q2_measured'] for e in events])
        weights = np.array([e['weight'] for e in events])
        
        # Create Q² bins
        Q2_min, Q2_max = Q2_measured.min(), Q2_measured.max()
        Q2_bins = np.linspace(Q2_min, Q2_max, n_bins + 1)
        Q2_centers = 0.5 * (Q2_bins[1:] + Q2_bins[:-1])
        
        # Calculate experimental cross sections for each bin
        experimental_data = []
        
        for i in range(n_bins):
            # Find events in this Q² bin
            mask = (Q2_measured >= Q2_bins[i]) & (Q2_measured < Q2_bins[i+1])
            
            if np.sum(mask) == 0:
                continue
                
            # Events in this bin
            bin_events = np.sum(mask)
            bin_weights = weights[mask]
            Q2_bin_center = Q2_centers[i]
            
            # Calculate the theoretical cross section at bin center
            # Convert Q2 back to theta for theoretical calculation
            # Q2 = 4*E_initial*E_final*sin²(θ/2), solve for θ
            # This is approximate - you might want more precise kinematics
            cos_half_theta = np.sqrt(1 - Q2_bin_center / (4 * E_initial**2))
            theta_center = 2 * np.arccos(cos_half_theta) if cos_half_theta <= 1 else np.pi
            
            theoretical_xs = self.rosenbluth_cross_section(E_initial, theta_center)
            
            # Simulate experimental cross section measurement
            # In a real experiment, you measure: N_events / (luminosity * acceptance * efficiency)
            # Here we'll simulate this with Poisson statistics and systematic uncertainties
            
            # Use actual event count for Poisson statistics (more realistic)
            actual_event_count = bin_events
            
            # Add Poisson noise based on actual event counts
            # Use sqrt(N) uncertainty for counting statistics
            if actual_event_count > 0:
                # Simulate finite statistics with Poisson fluctuations
                # Cap the lambda to avoid numpy overflow (max ~10^6)
                lambda_param = min(actual_event_count, 1000000)
                poisson_counts = np.random.poisson(lambda_param)
                count_uncertainty = np.sqrt(max(1, poisson_counts))
                
                # Calculate ratio and apply to theoretical cross section
                count_ratio = poisson_counts / actual_event_count if actual_event_count > 0 else 1.0
                measured_xs = theoretical_xs * count_ratio
                xs_statistical_error = theoretical_xs * (count_uncertainty / actual_event_count)
                
                # Add systematic uncertainty (typically 2-5% for well-calibrated experiments)
                systematic_error = 0.03 * measured_xs  # 3% systematic
                total_error = np.sqrt(xs_statistical_error**2 + systematic_error**2)
                
                experimental_data.append({
                    'Q2': Q2_bin_center,
                    'measured_xs': measured_xs,
                    'theoretical_xs': theoretical_xs,
                    'error': total_error,
                    'n_events': bin_events,
                    'ratio': measured_xs / theoretical_xs,
                    'ratio_error': total_error / theoretical_xs
                })
        
        return experimental_data

    def analyze_events(self, events, E_initial, true_GE=None, true_GM=None):
        """Analyze generated events and create plots with realistic experimental effects"""
        
        # Add detector effects to simulate real measurement
        events_with_detector = self.add_detector_effects(events.copy(), E_initial)
        
        # Extract data (using measured values where appropriate)
        thetas = [e['theta'] for e in events]
        thetas_measured = [e['theta_measured'] for e in events_with_detector]
        weights = [e['weight'] for e in events]
        Q2_values = [e['Q2'] for e in events]
        Q2_measured = [e['Q2_measured'] for e in events_with_detector]
        E_scattered = [e['E_scattered'] for e in events]
        E_measured = [e['E_measured'] for e in events_with_detector]
        p_scattered = [e['p_scattered'] for e in events]

        # Calculate experimental cross sections from binned data
        experimental_data = self.calculate_experimental_cross_sections(events_with_detector, E_initial)

        # Calculate cross-sections for comparison
        cross_sections = [self.rosenbluth_cross_section(E_initial, theta) 
                        for theta in thetas]

        # Build the new quantity E_final * dσ/dΩ
        rosen_compute = [e_sc * self.rosenbluth_cross_section(E_initial, theta)
                        for e_sc, theta in zip(E_scattered, thetas)]

        # 4x2 grid of plots
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = axes

        # 1) Scattering angle (true vs measured)
        ax1.hist(np.degrees(thetas), bins=50, weights=weights, density=True, 
                 alpha=0.7, label='True', color='blue')
        ax1.set(xlabel='Scattering Angle (°)', ylabel='Weighted (norm.)',
                title='Angle Distribution'); ax1.grid(alpha=0.3); ax1.legend()

        # 2) Q² (true vs measured)
        ax2.hist(Q2_values, bins=50, weights=weights, density=True, 
                 alpha=0.7, label='True', color='blue')
        ax2.set(xlabel='Q² (GeV²)', ylabel='Weighted (norm.)',
                title='Momentum Transfer'); ax2.grid(alpha=0.3); ax2.legend()

        # 3) Scattered energy (true vs measured)
        ax3.hist(E_scattered, bins=50, weights=weights, density=True, 
                 alpha=0.7, label='True', color='blue')
        ax3.set(xlabel='E_scattered (GeV)', ylabel='Weighted (norm.)',
                title='Energy Distribution'); ax3.grid(alpha=0.3); ax3.legend()

        # 4) Weights
        ax4.hist(weights, bins=50, alpha=0.7)
        ax4.set(xlabel='Event Weight', ylabel='Counts',
                title='Weight Distribution'); ax4.grid(alpha=0.3)

        # 5) Scattered momentum
        ax5.hist(p_scattered, bins=50, weights=weights, density=True, alpha=0.7)
        ax5.set(xlabel='p_scattered (GeV/c)', ylabel='Weighted (norm.)',
                title='Momentum Distribution'); ax5.grid(alpha=0.3)

        # 6) E_final × dσ/dΩ
        ax6.hist(rosen_compute, bins=50, weights=weights, density=True, alpha=0.7)
        ax6.set(xlabel=r'$E_{\!f}\,\frac{d\sigma}{d\Omega}$',
                ylabel='Weighted (norm.)',
                title=r'$E_f \times \frac{d\sigma}{d\Omega}$'); ax6.grid(alpha=0.3)

        # 7) Q² vs Cross-section Ratio
        if experimental_data:
            Q2_exp = [d['Q2'] for d in experimental_data]
            ratios = [d['ratio'] for d in experimental_data]
            ratio_errors = [d['ratio_error'] for d in experimental_data]
            n_events = [d['n_events'] for d in experimental_data]
            
            # Plot experimental points with error bars
            ax7.errorbar(Q2_exp, ratios, yerr=ratio_errors, fmt='ro', 
                        capsize=3, capthick=1, label='Experimental Data', markersize=6)
            
            # Add a horizontal line at ratio = 1 for reference
            ax7.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Perfect Dipole')
            
            # Color code points by number of events (statistics quality)
            scatter = ax7.scatter(Q2_exp, ratios, c=n_events, cmap='viridis', 
                                s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('Events per bin')
            
            ax7.set(xlabel='Q² (GeV²)', 
                    ylabel=r'$\frac{\sigma_{\rm measured}}{\sigma_{\rm dipole}}$',
                    title='Experimental Cross-section Ratio\n(with detector effects and finite statistics)')
            ax7.grid(alpha=0.3)
            ax7.legend()
            
            # Set reasonable y-axis limits
            y_min = max(0.5, min(ratios) - 3*max(ratio_errors))
            y_max = min(2.0, max(ratios) + 3*max(ratio_errors))
           # ax7.set_ylim(y_min, y_max)
            ax7.set_ylim()
            ax7.set_xlim()


        # 8) Statistical precision vs Q²
        if experimental_data:
            Q2_exp = [d['Q2'] for d in experimental_data]
            relative_errors = [d['ratio_error']/d['ratio'] * 100 for d in experimental_data]
            n_events = [d['n_events'] for d in experimental_data]
            
            ax8.scatter(Q2_exp, relative_errors, c=n_events, cmap='plasma', s=50)
            ax8.set(xlabel='Q² (GeV²)', ylabel='Relative Error (%)',
                    title='Statistical Precision vs Q²')
            ax8.grid(alpha=0.3)
            
            # Add colorbar
            cbar8 = plt.colorbar(ax8.collections[0], ax=ax8)
            cbar8.set_label('Events per bin')

        plt.tight_layout()
        plt.show()

        # Print experimental statistics
        print(f"\nExperimental Analysis Results:")
        print(f"Total MC events generated: {len(events)}")
        print(f"Experimental data points: {len(experimental_data)}")
        
        if experimental_data:
            ratios = [d['ratio'] for d in experimental_data]
            ratio_errors = [d['ratio_error'] for d in experimental_data]
            print(f"Cross-section ratios:")
            print(f"  Mean: {np.mean(ratios):.3f} ± {np.mean(ratio_errors):.3f}")
            print(f"  Range: {min(ratios):.3f} - {max(ratios):.3f}")
            print(f"  Chi²/dof test vs dipole: {np.sum([(r-1)**2/e**2 for r,e in zip(ratios, ratio_errors)]):.2f}/{len(ratios)-1}")

def main():
    FastMC = ElectronProtonScattering()

    # Physics parameters
    E_initial = 4.0  # GeV
    theta_min = np.radians(2)   # Start at 2 degrees
    theta_max = np.radians(120.0)  # End at 120 degrees
    n_events = 10000  


    print("=" * 60)
    print("REALISTIC ELECTRON-PROTON SCATTERING EXPERIMENT SIMULATION")
    print("Including detector effects, finite statistics, and measurement uncertainties")
    print("=" * 60)


    print(f"\n{'='*20} EXPERIMENTAL SIMULATION {'='*20}")
    events = FastMC.generate_events(E_initial, theta_min, theta_max, n_events)


    FastMC.analyze_events(events, E_initial)

if __name__ == "__main__":
    main()