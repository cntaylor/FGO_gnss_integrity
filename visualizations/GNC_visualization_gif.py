import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def quadratic_cost(x, data):
    """Standard Least Squares: sum((x - data)^2)"""
    return np.sum((x[:, np.newaxis] - data)**2, axis=1)

def gnc_geman_mcclure_cost(x, data, c, mu):
    """
    Graduated Geman-McClure robust cost function:
    sum( (mu * c^2 * y^2) / (mu * c^2 + y^2) )
    where y = x - data_point
    """
    diff = x[:, np.newaxis] - data
    y_sq = diff**2
    scale = mu * (c**2)
    return np.sum((scale * y_sq) / (scale + y_sq), axis=1)
    
def animate_robust_costs():
    # 1. Setup Data
    data = np.array([-0.2, 0.2, 4.0])
    c_val = 1.0
    x_range = np.linspace(-2, 6, 1000)
    
    # Pause parameters (number of frames)
    # At 20fps, 40 frames = 2 seconds
    start_pause = 40 
    mid_pause = 30
    end_pause = 60

    # Define mu schedule: from high (100) down to 1 (robust)
    # We split it into two halves to insert a pause in the middle
    mu_start = 100.0
    mu_end = 1.0
    mu_mid = 10**((np.log10(mu_start) + np.log10(mu_end)) / 2) # Geometric mean for log scale
    
    # Generate the moving frames
    first_half = np.logspace(np.log10(mu_start), np.log10(mu_mid), 50)
    second_half = np.logspace(np.log10(mu_mid), np.log10(mu_end), 50)
    
    # Construct the full frame sequence with pauses
    mu_values = np.concatenate([
        np.full(start_pause, mu_start),    # Pause at start
        first_half,                        # Move to middle
        np.full(mid_pause, mu_mid),        # Pause in middle
        second_half,                       # Move to end
        np.full(end_pause, mu_end)         # Pause at end
    ])
    
    # 2. Initialize Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], color='red', lw=2, label='GNC Cost Function')
    opt_line = ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Global Optimum')
    
    # Text display for current values
    stats_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Static elements
    ax.scatter(data, np.zeros_like(data), color='black', marker='x', s=100, zorder=5, label='Data Points')
    ax.set_xlim(-2, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('Total Cost')
    ax.grid(True, alpha=0.3)
    title = ax.set_title('')
    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        stats_text.set_text('')
        return line, opt_line, title, stats_text

    def update(mu):
        cost = gnc_geman_mcclure_cost(x_range, data, c=c_val, mu=mu)
        line.set_data(x_range, cost)
        
        # Dynamic Y-Axis Scaling
        ax.set_ylim(-0.5, np.max(cost) * 1.1)
        
        # Update global optimum vertical line
        opt_val = x_range[np.argmin(cost)]
        opt_line.set_xdata([opt_val, opt_val])
        
        # Update Title and Stats Text
        title.set_text(f'Graduated Non-Convexity: Geman-McClure Transformation')
        stats_text.set_text(f'Current $\\theta$: {mu:.2f}\nGlobal Optimum: {opt_val:.3f}')
        
        return line, opt_line, title, stats_text

    # 3. Create Animation
    ani = FuncAnimation(fig, update, frames=mu_values, init_func=init, blit=False, interval=50)
    
    # Save as GIF
    try:
        ani.save('gnc_gmtransformation.gif', writer='pillow', fps=20)
        print("Animation saved as gnc_gm_transformation.gif")
    except ImportError:
        print("Pillow library required to save as GIF. Please install it with 'pip install Pillow'.")
    
    plt.show()

def gnc_truncated_least_squares_cost(x, data, c, mu):
    """
    Graduated Truncated Least Squares robust cost function based on specific piecewise formula.
    
    Formula:
     y^2 when y^2 < c^2 * (mu/(mu+1))
     2*c*abs(y)*sqrt(mu*(mu+1)) - mu*(c^2 + y^2) when (mu/(mu+1)) <= y^2/c^2 <= ((mu+1)/mu)
     c^2 otherwise
    """
    diff = x[:, np.newaxis] - data
    y_sq = diff**2
    y_abs = np.abs(diff)
    c_sq = c**2
    
    # Pre-calculate terms to avoid division by zero if mu=0
    # We use a tiny epsilon for mu=0 cases
    mu_safe = max(mu, 1e-10)
    
    term1_bound = c_sq * (mu_safe / (mu_safe + 1))
    term2_upper_bound = c_sq * ((mu_safe + 1) / mu_safe)
    
    # Case 1: y^2 < c^2 * (mu/(mu+1))
    mask1 = y_sq < term1_bound
    
    # Case 2: (mu/(mu+1)) <= y^2/c^2 <= ((mu+1)/mu)
    mask2 = (y_sq >= term1_bound) & (y_sq <= term2_upper_bound)
    
    # Case 3: otherwise (y^2/c^2 > (mu+1)/mu)
    mask3 = y_sq > term2_upper_bound
    
    # Calculate costs
    costs = np.zeros_like(y_sq)
    
    costs[mask1] = y_sq[mask1]
    
    # Complex middle term
    costs[mask2] = (2 * c * y_abs[mask2] * np.sqrt(mu_safe * (mu_safe + 1)) 
                    - mu_safe * (c_sq + y_sq[mask2]))
    
    costs[mask3] = c_sq
    
    return np.sum(costs, axis=1)
    
def animate_robust_costs_tls():
    # 1. Setup Data
    data = np.array([-0.2, 0.2, 4.0])
    c_val = 1.0
    x_range = np.linspace(-2, 6, 1000)
    
    # Pause parameters (number of frames)
    start_pause = 40 
    mid_pause = 30
    end_pause = 60

    # Define mu schedule: from high (effectively quadratic) down to 0 (robust)
    mu_start = 1/1000.0
    mu_end = 1.0
    mu_mid = 10**((np.log10(mu_start) + np.log10(mu_end)) / 2) # Geometric mean for log scale
    
    # Generate the moving frames
    first_half = np.logspace(np.log10(mu_start), np.log10(mu_mid), 50)
    second_half = np.logspace(np.log10(mu_mid), np.log10(mu_end), 50)
    
    # Construct the full frame sequence with pauses
    mu_frames = np.concatenate([
        np.full(start_pause, mu_start),    # Pause at start
        first_half,                        # Move to middle
        np.full(mid_pause, mu_mid),        # Pause in middle
        second_half,                       # Move to end
        np.full(end_pause, mu_end)         # Pause at end
    ])
    
    
    # 2. Initialize Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], color='purple', lw=2, label='Graduated TLS Cost')
    opt_line = ax.axvline(0, color='purple', linestyle='--', alpha=0.5, label='Global Optimum')
    
    stats_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.scatter(data, np.zeros_like(data), color='black', marker='x', s=100, zorder=5, label='Data Points')
    ax.set_xlim(-2, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('Total Cost')
    ax.grid(True, alpha=0.3)
    title = ax.set_title('')
    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        stats_text.set_text('')
        return line, opt_line, title, stats_text

    def update(mu):
        cost = gnc_truncated_least_squares_cost(x_range, data, c=c_val, mu=mu)
        line.set_data(x_range, cost)
        
        # Dynamic Y-Axis Scaling
        ax.set_ylim(-0.1, np.max(cost) * 1.1)
        
        # Update global optimum vertical line
        opt_val = x_range[np.argmin(cost)]
        opt_line.set_xdata([opt_val, opt_val])
        
        title.set_text(f'GNC: Piecewise Truncated Least Squares')
        stats_text.set_text(f'Current $\\theta$: {mu:.4f}\nGlobal Optimum: {opt_val:.3f}')
        
        return line, opt_line, title, stats_text

    # 3. Create Animation
    ani = FuncAnimation(fig, update, frames=mu_frames, init_func=init, blit=False, interval=50)
    
    # Save as GIF
    try:
        ani.save('gnc_piecewise_tls.gif', writer='pillow', fps=20)
        print("Animation saved as gnc_piecewise_tls.gif")
    except ImportError:
        print("Pillow library required to save as GIF.")
    
    plt.show()


if __name__ == "__main__":
    # animate_robust_costs_gm()
    animate_robust_costs_tls()    
    

