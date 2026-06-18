import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def nephron_transport_model():
    print("Initializing Weinstein-Stephenson Nephron ODE Model...")
    
    # Tubule properties
    L = 1.0  # Length of proximal tubule (cm)
    r = 0.002  # Tubule radius (cm)
    A = np.pi * r**2  # Cross-sectional area
    
    # Transport coefficients (simplified parameters)
    P_f = 0.5  # Water permeability (cm/s)
    P_Na = 0.3 # NaCl permeability
    V_max = 0.1 # Active transport max rate
    K_m = 10.0 # Michaelis constant
    
    # Boundary conditions from glomerulus
    Q0 = 0.005 # Initial Volume Flow (cm^3/s)
    C0 = 140.0 # Initial NaCl Concentration (mM)
    
    def transport_odes(x, y):
        Q, C = y # y[0] = Flow, y[1] = Concentration
        
        # Prevent negative flow/concentration mathematically
        Q = max(Q, 1e-9)
        C = max(C, 1e-9)
        
        # 1. Active NaCl reabsorption (Michaelis-Menten kinetics)
        J_active = V_max * C / (K_m + C)
        
        # 2. Passive NaCl diffusion (driven by gradient, assuming interstitial is 140)
        C_interstitial = 140.0
        J_passive = P_Na * (C - C_interstitial)
        
        Total_NaCl_Flux = J_active + J_passive
        
        # 3. Water reabsorption (Osmosis driven by solute flux to maintain isotonicity)
        # Simplified: water follows salt
        Water_Flux = P_f * Total_NaCl_Flux
        
        # The ODEs: Change over length x
        # dQ/dx = -Water_Flux * Circumference
        # d(Q*C)/dx = -Total_NaCl_Flux * Circumference => Q*dC/dx + C*dQ/dx = ...
        
        circum = 2 * np.pi * r
        dQ_dx = -Water_Flux * circum
        
        # Chain rule: dC/dx = (-Total_NaCl_Flux * circum - C * dQ_dx) / Q
        dC_dx = (-Total_NaCl_Flux * circum - C * dQ_dx) / Q
        
        return [dQ_dx, dC_dx]

    print("Solving stiff ODE system along proximal tubule...")
    # Solve using stiff-aware Radau method
    sol = solve_ivp(transport_odes, [0, L], [Q0, C0], method='Radau', dense_output=True, max_step=0.01)
    
    x_eval = np.linspace(0, L, 100)
    y_eval = sol.sol(x_eval)
    
    Q_profile = y_eval[0]
    C_profile = y_eval[1]
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save profiles
    np.save(os.path.join(results_dir, 'nephron_Q.npy'), Q_profile)
    np.save(os.path.join(results_dir, 'nephron_C.npy'), C_profile)
    np.save(os.path.join(results_dir, 'nephron_x.npy'), x_eval)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Tubule Length (cm)')
    ax1.set_ylabel('Fluid Flow Rate $Q$ (cm$^3$/s)', color=color)
    ax1.plot(x_eval, Q_profile, color=color, linewidth=2, label='Fluid Flow')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('NaCl Concentration $C$ (mM)', color=color)
    ax2.plot(x_eval, C_profile, color=color, linewidth=2, linestyle='--', label='NaCl Concentration')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Proximal Tubule Reabsorption Profile (Weinstein ODE Model)')
    fig.tight_layout()
    out_path = os.path.join(results_dir, 'nephron_profile.png')
    plt.savefig(out_path, dpi=150)
    print(f"ODE Simulation completed. Results saved to {out_path}")

if __name__ == "__main__":
    nephron_transport_model()
