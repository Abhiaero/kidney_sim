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
        
    nx = u.shape[1]
    ny = u.shape[0]
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # Apply mask for visualization (set solid parts to NaN)
    speed = np.sqrt(u**2 + v**2)
    speed[mask == 1] = np.nan
    p[mask == 1] = np.nan
    
    # 1. Velocity Contour and Streamlines over Geometry
    # Background as geometry
    axs[0].imshow(mask, cmap='gray_r', extent=[0, 2, 0, 1], origin='lower', alpha=0.3)
    strm = axs[0].streamplot(X, Y, u, v, color=speed, linewidth=1, cmap='jet', density=1.5)
    if strm.lines:
        fig.colorbar(strm.lines, ax=axs[0], label='Velocity Magnitude')
    axs[0].set_title('Velocity Field overlaid on Glomerulus Cross-Section')
    axs[0].set_xlabel('Length [x]')
    axs[0].set_ylabel('Height [y]')
    
    # 2. Pressure Contour over Geometry
    axs[1].imshow(mask, cmap='gray_r', extent=[0, 2, 0, 1], origin='lower', alpha=0.3)
    cp = axs[1].contourf(X, Y, p, alpha=0.8, cmap='viridis', levels=20)
    fig.colorbar(cp, ax=axs[1], label='Pressure $p$')
    axs[1].set_title('Pressure Field overlaid on Glomerulus Cross-Section')
    axs[1].set_xlabel('Length [x]')
    axs[1].set_ylabel('Height [y]')
    
    plt.tight_layout()
    plt.savefig('results/masked_cfd_simulation_results.png', dpi=300)
    print("Plot saved to 'results/masked_cfd_simulation_results.png'.")

if __name__ == '__main__':
    plot_results()
