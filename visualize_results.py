import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists('results'):
        print("Results directory not found. Please run the simulation first.")
        return
        
    u = np.load('results/u.npy')
    v = np.load('results/v.npy')
    p = np.load('results/p.npy')
    
    if os.path.exists('results/mask.npy'):
        mask = np.load('results/mask.npy')
    else:
        mask = np.zeros_like(u)
        
    if os.path.exists('results/nu.npy'):
        nu = np.load('results/nu.npy')
    else:
        nu = np.zeros_like(u)
        
    nx = u.shape[1]
    ny = u.shape[0]
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    speed = np.sqrt(u**2 + v**2)
    
    # Professionally mask out the solid tissue regions so contourf doesn't draw them.
    # This leaves the solid tissue area entirely blank so the imshow mask can color it solid grey.
    # This is the industry standard for visualizing flow inside complex geometries!
    speed[mask == 1] = np.nan
    p[mask == 1] = np.nan
    nu[mask == 1] = np.nan
    
    # We also mask u and v for streamplot to avoid plotting streamlines inside the solid.
    u_strm = u.copy()
    v_strm = v.copy()
    u_strm[mask == 1] = np.nan
    v_strm[mask == 1] = np.nan
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # 1. Velocity Contour and Streamlines
    axs[0].imshow(mask, cmap='gray_r', extent=[0, 2, 0, 1], origin='lower', alpha=0.5)
    cs0 = axs[0].contourf(X, Y, speed, alpha=0.8, cmap='jet', levels=20)
    fig.colorbar(cs0, ax=axs[0], label='Velocity Magnitude (Linear)')
    axs[0].streamplot(X, Y, u_strm, v_strm, color='white', linewidth=1, density=1.0)
    axs[0].set_title('Velocity Field & Streamlines (Stokes Flow)')
    axs[0].set_xlabel('Length [x]')
    axs[0].set_ylabel('Height [y]')
    
    # 2. Pressure Contour
    axs[1].imshow(mask, cmap='gray_r', extent=[0, 2, 0, 1], origin='lower', alpha=0.5)
    cs1 = axs[1].contourf(X, Y, p, alpha=0.8, cmap='viridis', levels=20)
    fig.colorbar(cs1, ax=axs[1], label='Pressure $p$')
    axs[1].set_title('Pressure Field')
    axs[1].set_xlabel('Length [x]')
    axs[1].set_ylabel('Height [y]')
    
    # 3. Dynamic Viscosity Contour (Non-Newtonian Effects)
    axs[2].imshow(mask, cmap='gray_r', extent=[0, 2, 0, 1], origin='lower', alpha=0.5)
    cs2 = axs[2].contourf(X, Y, nu, alpha=0.8, cmap='plasma', levels=20)
    fig.colorbar(cs2, ax=axs[2], label='Kinematic Viscosity $\\nu$')
    axs[2].set_title('Non-Newtonian Viscosity Field (Carreau-Yasuda)')
    axs[2].set_xlabel('Length [x]')
    axs[2].set_ylabel('Height [y]')
    
    plt.tight_layout()
    plt.savefig('results/masked_cfd_simulation_results.png', dpi=150)
    print("Plot saved to 'results/masked_cfd_simulation_results.png'.")

if __name__ == '__main__':
    plot_results()
