"""
Tau Evolution Simulation
Time evolution of vacuum tension parameter
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class TauEvolution:
    """Simulate evolution of tau parameter"""
    
    def __init__(self):
        self.G0 = 6.6743e-11  # m³/kg·s²
        self.c = 299792458    # m/s
        self.kappa = -0.2116
        
    def G_of_tau(self, tau):
        """G as function of tau"""
        return self.G0 * (tau/0.969818)**self.kappa
    
    def Lambda_of_tau(self, tau):
        """Cosmological constant as function of tau"""
        E_vac = 149.267 * (1/tau - 1) * 1.602e-13  # J
        V_cell = (1.32e-15)**3  # m³ (Compton volume)
        rho_vac = E_vac / V_cell
        return (8*np.pi*self.G_of_tau(tau)/self.c**4) * rho_vac
    
    def friedmann_equation(self, t, y, Omega_m0=0.3, Omega_r0=1e-4):
        """
        Modified Friedmann equation with tau evolution
        y = [a, tau, tau_dot]
        """
        a, tau, tau_dot = y
        
        # Energy densities
        rho_m = Omega_m0 * 9.9e-27 / a**3  # kg/m³
        rho_r = Omega_r0 * 9.9e-27 / a**4
        
        # Hubble parameter
        H2 = (8*np.pi*self.G_of_tau(tau)/3) * (rho_m + rho_r) + \
             self.Lambda_of_tau(tau)*self.c**2/3 + \
             tau_dot**2/(2*tau**2)
        
        H = np.sqrt(H2) if H2 > 0 else 0
        
        # Tau evolution equation
        tau_ddot = -3*H*tau_dot - 149.267*1.602e-13/(tau**2)  # Simplified
        
        return [H*a, tau_dot, tau_ddot]
    
    def simulate_cosmic_evolution(self, t_span=(0, 13.8), y0=None):
        """Simulate cosmic evolution from Big Bang to present"""
        if y0 is None:
            y0 = [1e-30, 0.01, 0.1]  # a, tau, tau_dot at t≈0
        
        sol = solve_ivp(
            self.friedmann_equation,
            t_span,
            y0,
            method='RK45',
            dense_output=True,
            rtol=1e-8,
            atol=1e-12
        )
        
        return sol
    
    def plot_evolution(self, sol):
        """Plot evolution results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = sol.t
        a = sol.y[0]
        tau = sol.y[1]
        
        # Scale factor
        axes[0,0].semilogy(t, a)
        axes[0,0].set_xlabel('Time (Gyr)')
        axes[0,0].set_ylabel('Scale factor a(t)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Tau evolution
        axes[0,1].plot(t, tau)
        axes[0,1].set_xlabel('Time (Gyr)')
        axes[0,1].set_ylabel('Vacuum tension τ(t)')
        axes[0,1].axhline(0.969818, color='r', linestyle='--', alpha=0.5, label='Current value')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Hubble parameter
        H = np.gradient(np.log(a), t)
        axes[1,0].plot(t, H)
        axes[1,0].set_xlabel('Time (Gyr)')
        axes[1,0].set_ylabel('Hubble parameter H(t) (Gyr⁻¹)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Energy densities
        rho_vac = 149.267 * (1/tau - 1) * 1.602e-13 / (1.32e-15)**3
        axes[1,1].semilogy(t, rho_vac, label='Vacuum energy')
        axes[1,1].set_xlabel('Time (Gyr)')
        axes[1,1].set_ylabel('Energy density (J/m³)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tau_cosmic_evolution.pdf', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    simulator = TauEvolution()
    sol = simulator.simulate_cosmic_evolution()
    simulator.plot_evolution(sol)