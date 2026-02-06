"""
FFM Structure Energy Calculations
Main numerical computations for the paper
"""

import numpy as np
from scipy import constants as const

class FFMModel:
    """Main FFM calculation class"""
    
    def __init__(self):
        # Fundamental constants (CODATA 2018)
        self.hbar = const.hbar  # 1.054571817e-34 J·s
        self.c = const.c        # 299792458 m/s
        self.e = const.e        # 1.602176634e-19 C
        
        # FFM calibrated parameters
        self.f_eff = 2.26758e23    # Hz
        self.tau_vac = 0.969818
        self.tau_p = 0.910000
        self.tau_n = 0.902750
        
        # Derived constants
        self.E_base = self.hbar * self.f_eff * self.J_to_MeV()
        
    def J_to_MeV(self):
        """Convert Joules to MeV"""
        return 1 / (self.e * 1e6)
    
    def structure_energy(self, tau):
        """
        Calculate structure energy for given tau
        E_struct = ħ f_eff (1/τ - 1)
        """
        return self.E_base * (1/tau - 1)
    
    def calculate_all(self):
        """Calculate all key quantities"""
        results = {}
        
        # Structure energies
        results['E_p'] = self.structure_energy(self.tau_p)
        results['E_n'] = self.structure_energy(self.tau_n)
        results['delta_E'] = results['E_n'] - results['E_p']
        
        # Experimental values
        results['delta_E_exp'] = 1.293  # MeV
        
        # Errors
        results['abs_error'] = abs(results['delta_E'] - results['delta_E_exp'])
        results['rel_error'] = results['abs_error'] / results['delta_E_exp'] * 100
        
        # Total energies
        results['E_total_p'] = 938.272 + results['E_p']  # MeV
        results['E_total_n'] = 939.565 + results['E_n']  # MeV
        
        return results
    
    def generate_tau_table(self, tau_values=None):
        """Generate table of tau vs energy"""
        if tau_values is None:
            tau_values = [1.00, 0.99, 0.97, 0.95, 0.93, 0.91, 0.90, 
                         0.89, 0.87, 0.85, 0.80, 0.70, 0.60, 0.50,
                         0.30, 0.10, 0.01]
        
        table = []
        for tau in tau_values:
            E = self.structure_energy(tau)
            table.append({
                'tau': tau,
                'tau_term': 1/tau - 1,
                'E_MeV': E,
                'description': self.get_tau_description(tau)
            })
        
        return table
    
    def get_tau_description(self, tau):
        """Get description for tau value"""
        if tau == 1.00:
            return "Ideal vacuum"
        elif abs(tau - 0.969818) < 0.001:
            return "Physical vacuum / free particle"
        elif abs(tau - 0.91) < 0.001:
            return "Proton in nucleus"
        elif abs(tau - 0.90275) < 0.001:
            return "Neutron in nucleus"
        elif tau > 0.95:
            return "Light particle"
        elif tau > 0.85:
            return "Nucleus / compressed"
        elif tau > 0.50:
            return "Compact object"
        else:
            return "Singularity / black hole"

# Main execution
if __name__ == "__main__":
    model = FFMModel()
    results = model.calculate_all()
    
    print("="*70)
    print("FFM STRUCTURE ENERGY CALCULATIONS")
    print("="*70)
    
    print(f"\nBase energy: E_base = ħ f_eff = {model.E_base:.6f} MeV")
    print(f"Effective frequency: f_eff = {model.f_eff:.3e} Hz")
    
    print(f"\nStructure energies:")
    print(f"E_struct(p, τ={model.tau_p}) = {results['E_p']:.6f} MeV")
    print(f"E_struct(n, τ={model.tau_n}) = {results['E_n']:.6f} MeV")
    print(f"ΔE_struct = {results['delta_E']:.6f} MeV")
    
    print(f"\nComparison with experiment:")
    print(f"Experimental Δm(n-p) = {results['delta_E_exp']:.6f} MeV")
    print(f"Absolute error = {results['abs_error']:.6f} MeV")
    print(f"Relative error = {results['rel_error']:.3f}%")
    
    print(f"\nTotal energies (rest mass + structure):")
    print(f"E_total(p) = 938.272 + {results['E_p']:.3f} = {results['E_total_p']:.3f} MeV")
    print(f"E_total(n) = 939.565 + {results['E_n']:.3f} = {results['E_total_n']:.3f} MeV")
    
    # Generate and print tau table
    print(f"\n" + "="*70)
    print("TAU-ENERGY CONVERSION TABLE")
    print("="*70)
    print(f"{'τ':<8} {'1/τ-1':<10} {'E_struct (MeV)':<15} Description")
    print("-"*50)
    
    table = model.generate_tau_table()
    for row in table:
        print(f"{row['tau']:<8.2f} {row['tau_term']:<10.4f} {row['E_MeV']:<15.3f} {row['description']}")