"""
LHC Predictions from FFM Model
Cross-section enhancements and timing anomalies
"""

import numpy as np
import matplotlib.pyplot as plt

class LHCPredictions:
    """Predict LHC anomalies from FFM"""
    
    def __init__(self):
        self.E_nyquist = 246.2  # GeV
        self.dt0 = 1e-15  # s (1 fs)
        
    def cross_section_correction(self, E):
        """
        Cross-section correction factor C(E)
        E: energy in GeV
        """
        if E < self.E_nyquist:
            return 1 + 0.01 * (E / self.E_nyquist)**2
        else:
            return 1 + 0.1 * np.exp((E - self.E_nyquist) / 10)
    
    def timing_anomaly(self, E):
        """
        Timing anomaly in seconds
        Δt = 1e-15 × (E/246.2)^4
        """
        return self.dt0 * (E / self.E_nyquist)**4
    
    def particle_ratio_modification(self, E, particle_type='photon'):
        """
        Modify particle production ratios
        """
        if particle_type == 'photon':
            return 1 + 0.5 * np.exp(-((E - self.E_nyquist)/50)**2)
        elif particle_type == 'hadron':
            if E < self.E_nyquist:
                return 1 - 0.3 * (E / self.E_nyquist)**2
            else:
                return 0.7  # Constant above threshold
        else:
            return 1.0
    
    def generate_predictions(self, energy_range=(50, 14000)):
        """Generate predictions for energy range"""
        energies = np.logspace(np.log10(energy_range[0]), 
                              np.log10(energy_range[1]), 500)
        
        results = {
            'energy': energies,
            'cross_section': [],
            'timing_anomaly': [],
            'photon_ratio': [],
            'hadron_ratio': []
        }
        
        for E in energies:
            results['cross_section'].append(self.cross_section_correction(E))
            results['timing_anomaly'].append(self.timing_anomaly(E))
            results['photon_ratio'].append(self.particle_ratio_modification(E, 'photon'))
            results['hadron_ratio'].append(self.particle_ratio_modification(E, 'hadron'))
        
        return results
    
    def plot_predictions(self, results):
        """Plot all predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        E = results['energy']
        
        # Cross-section correction
        ax = axes[0, 0]
        ax.loglog(E, results['cross_section'], linewidth=2)
        ax.axvline(self.E_nyquist, color='r', linestyle='--', alpha=0.7, label=f'E_N = {self.E_nyquist} GeV')
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Cross-section correction C(E)')
        ax.set_title('Cross-section Enhancement')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Timing anomaly
        ax = axes[0, 1]
        ax.loglog(E, np.array(results['timing_anomaly'])*1e15, linewidth=2)
        ax.axvline(self.E_nyquist, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Timing anomaly Δt (fs)')
        ax.set_title('Timing Anomalies')
        ax.grid(True, alpha=0.3, which='both')
        
        # Particle ratios
        ax = axes[1, 0]
        ax.semilogx(E, results['photon_ratio'], label='γ ratio', linewidth=2)
        ax.semilogx(E, results['hadron_ratio'], label='Hadron ratio', linewidth=2)
        ax.axvline(self.E_nyquist, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Production ratio R(E)')
        ax.set_title('Particle Production Ratios')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Zoom around Nyquist threshold
        ax = axes[1, 1]
        mask = (E > 100) & (E < 1000)
        E_zoom = E[mask]
        cs_zoom = np.array(results['cross_section'])[mask]
        
        ax.plot(E_zoom, cs_zoom, linewidth=2)
        ax.axvline(self.E_nyquist, color='r', linestyle='--', alpha=0.7, label='Nyquist threshold')
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('C(E)')
        ax.set_title('Zoom around 246 GeV')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lhc_predictions.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_lhc_table(self):
        """Generate table for specific LHC energies"""
        lhc_energies = {
            'Injection': 450,  # GeV
            'Standard run': 6500,  # GeV per beam
            'Design': 7000,  # GeV
            'HL-LHC': 7500,  # GeV
            'FCC-hh': 50000,  # GeV (future)
        }
        
        print("="*70)
        print("LHC PREDICTIONS FROM FFM MODEL")
        print("="*70)
        print(f"\nNyquist threshold: E_N = {self.E_nyquist} GeV")
        print("\n" + "-"*70)
        print(f"{'Scenario':<15} {'E (GeV)':<10} {'C(E)':<12} {'Δt (fs)':<12} {'R(γ)':<10} {'R(h)':<10}")
        print("-"*70)
        
        for scenario, energy in lhc_energies.items():
            C = self.cross_section_correction(energy)
            dt = self.timing_anomaly(energy) * 1e15  # convert to fs
            R_gamma = self.particle_ratio_modification(energy, 'photon')
            R_hadron = self.particle_ratio_modification(energy, 'hadron')
            
            print(f"{scenario:<15} {energy:<10.0f} {C:<12.3f} {dt:<12.3f} {R_gamma:<10.3f} {R_hadron:<10.3f}")

if __name__ == "__main__":
    predictor = LHCPredictions()
    
    # Generate and plot predictions
    results = predictor.generate_predictions()
    predictor.plot_predictions(results)
    
    # Print LHC table
    predictor.generate_lhc_table()
    
    # Specific prediction for Nyquist threshold
    print("\n" + "="*70)
    print("SPECIFIC PREDICTIONS AT NYQUIST THRESHOLD")
    print("="*70)
    print(f"\nAt E = {predictor.E_nyquist} GeV:")
    print(f"Cross-section enhancement: {predictor.cross_section_correction(predictor.E_nyquist):.3f}")
    print(f"Timing anomaly: {predictor.timing_anomaly(predictor.E_nyquist)*1e15:.3f} fs")
    print(f"Photon/hadron ratio enhancement: {predictor.particle_ratio_modification(predictor.E_nyquist, 'photon'):.3f}")