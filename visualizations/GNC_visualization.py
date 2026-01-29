import numpy as np
import matplotlib.pyplot as plt

def quadratic_cost(x, data):
    """Standard Least Squares: sum((x - data)^2)"""
    return np.sum((x[:, np.newaxis] - data)**2, axis=1)

def geman_mcclure_cost(x, data, c):
    """
    Standard Geman-McClure Robust Cost Function:
    sum( (y^2/2) / (1 + (y/c)^2) )
    where y = x - data_point
    """
    diff = x[:, np.newaxis] - data
    numerator = (diff**2) / 2
    denominator = 1 + (diff**2 / c**2)
    return np.sum(numerator / denominator, axis=1)

def gnc_geman_mcclure_cost(x, data, c, mu):
    """
    Graduated Geman-McClure robust cost function:
    sum( (mu * c^2 * y^2) / (mu * c^2 + y^2) )
    where y = x - data_point
    As mu -> infinity, this approaches quadratic.
    As mu -> 1, this approaches standard Geman-McClure.
    """
    diff = x[:, np.newaxis] - data
    y_sq = diff**2
    scale = mu * (c**2)
    return np.sum((scale * y_sq) / (scale + y_sq), axis=1)

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

def plot_robust_costs_gm():
    # 1. Setup Data
    # Measurements on the real line: -.2, .2, and 4 (the outlier)
    data = np.array([-0.2, 0.2, 4.0])
    c_val = 1.0  # Constant scale for the robust kernel
    
    # Range of x (theta) values to evaluate
    x_range = np.linspace(-2, 6, 1000)
    
    # 2. Initialize Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Graduated Non-Convexity (GNC): Geman-McClure Transformation\nData points at: ' + str(data), fontsize=16)
    
    # --- Pane 1: Quadratic (Mean) ---
    cost_quad = quadratic_cost(x_range, data)
    axes[0, 0].plot(x_range, cost_quad, color='blue', lw=2)
    axes[0, 0].set_title(r'1. Quadratic Cost (Initial State)' + '\n' + r'$\theta \simeq \infty$')
    axes[0, 0].grid(True, alpha=0.3)
    mean_val = np.mean(data)
    axes[0, 0].axvline(mean_val, color='blue', linestyle='--', alpha=0.5, label=f'Global Optimum: {mean_val:.2f}')
    axes[0, 0].legend()

    # --- Pane 2: GNC Geman-McClure (mu=10) ---
    cost_gnc_high = gnc_geman_mcclure_cost(x_range, data, c=c_val, mu=10.0)
    axes[0, 1].plot(x_range, cost_gnc_high, color='green', lw=2)
    axes[0, 1].set_title(r'2. Graduated Geman-McClure ($\theta=10$)' + '\nHighly Convex Approximation')
    axes[0, 1].grid(True, alpha=0.3)
    # Finding the local minimum for the current cost function
    opt_val_2 = x_range[np.argmin(cost_gnc_high)]
    axes[0, 1].axvline(opt_val_2, color='green', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_2:.2f}')
    axes[0, 1].legend()

    # --- Pane 3: GNC Geman-McClure (mu=2) ---
    cost_gnc_mid = gnc_geman_mcclure_cost(x_range, data, c=c_val, mu=2.0)
    axes[1, 0].plot(x_range, cost_gnc_mid, color='orange', lw=2)
    axes[1, 0].set_title(r'3. Graduated Geman-McClure ($\theta=2$)' + '\nTransition to Non-Convexity')
    axes[1, 0].grid(True, alpha=0.3)
    opt_val_3 = x_range[np.argmin(cost_gnc_mid)]
    axes[1, 0].axvline(opt_val_3, color='orange', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_3:.2f}')
    axes[1, 0].legend()

    # --- Pane 4: Robust Geman-McClure (mu=1) ---
    cost_gm_robust = gnc_geman_mcclure_cost(x_range, data, c=c_val, mu=1.0)
    axes[1, 1].plot(x_range, cost_gm_robust, color='red', lw=2)
    axes[1, 1].set_title(r'4. Geman-McClure ($\theta=1$)' + '\nRobust & Non-Convex')
    axes[1, 1].grid(True, alpha=0.3)
    opt_val_4 = x_range[np.argmin(cost_gm_robust)]
    axes[1, 1].axvline(opt_val_4, color='red', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_4:.2f}')
    axes[1, 1].legend()
    
    # Mark the data points on the x-axis for all plots
    for ax in axes.flat:
        ax.scatter(data, np.zeros_like(data), color='black', marker='x', s=100, zorder=5, label='Data')
        ax.set_xlabel('x')
        ax.set_ylabel('Total Cost')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('gm_gnc_explanation.pdf', format='pdf')
    print("Figure saved as robust_cost_comparison.pdf")

    plt.show()
def plot_robust_costs_gm():
    # 1. Setup Data
    # Measurements on the real line: -.2, .2, and 4 (the outlier)
    data = np.array([-0.2, 0.2, 4.0])
    c_val = 1.0  # Constant scale for the robust kernel
    
    # Range of x (theta) values to evaluate
    x_range = np.linspace(-2, 6, 1000)
    
    # 2. Initialize Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Graduated Non-Convexity (GNC): Truncated Least Squares Transformation\nData points at: ' + str(data), fontsize=16)
    
    # --- Pane 1: Quadratic (Mean) ---
    cost_quad = quadratic_cost(x_range, data)
    axes[0, 0].plot(x_range, cost_quad, color='blue', lw=2)
    axes[0, 0].set_title(r'1. Quadratic Cost (Initial State)' + '\n' + r'$\theta \simeq 0$')
    axes[0, 0].grid(True, alpha=0.3)
    mean_val = np.mean(data)
    axes[0, 0].axvline(mean_val, color='blue', linestyle='--', alpha=0.5, label=f'Global Optimum: {mean_val:.2f}')
    axes[0, 0].legend()

    # --- Pane 2: GNC Geman-McClure (mu=10) ---
    cost_gnc_high = gnc_truncated_least_squares_cost(x_range, data, c=c_val, mu=.01)
    axes[0, 1].plot(x_range, cost_gnc_high, color='green', lw=2)
    axes[0, 1].set_title(r'2. Truncated Least Squares ($\theta=.01$)' + '\nHighly Convex Approximation')
    axes[0, 1].grid(True, alpha=0.3)
    # Finding the local minimum for the current cost function
    opt_val_2 = x_range[np.argmin(cost_gnc_high)]
    axes[0, 1].axvline(opt_val_2, color='green', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_2:.2f}')
    axes[0, 1].legend()

    # --- Pane 3: GNC Geman-McClure (mu=2) ---
    cost_gnc_mid = gnc_truncated_least_squares_cost(x_range, data, c=c_val, mu=.1)
    axes[1, 0].plot(x_range, cost_gnc_mid, color='orange', lw=2)
    axes[1, 0].set_title(r'3. Truncated Least Squares ($\theta=.1$)' + '\nTransition to Non-Convexity')
    axes[1, 0].grid(True, alpha=0.3)
    opt_val_3 = x_range[np.argmin(cost_gnc_mid)]
    axes[1, 0].axvline(opt_val_3, color='orange', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_3:.2f}')
    axes[1, 0].legend()

    # --- Pane 4: Robust Geman-McClure (mu=1) ---
    cost_gm_robust = gnc_truncated_least_squares_cost(x_range, data, c=c_val, mu=1.0)
    axes[1, 1].plot(x_range, cost_gm_robust, color='red', lw=2)
    axes[1, 1].set_title(r'4. Truncated Least Squares ($\theta=1$)' + '\nRobust & Non-Convex')
    axes[1, 1].grid(True, alpha=0.3)
    opt_val_4 = x_range[np.argmin(cost_gm_robust)]
    axes[1, 1].axvline(opt_val_4, color='red', linestyle='--', alpha=0.5, label=f'Global Optimum: {opt_val_4:.2f}')
    axes[1, 1].legend()
    
    # Mark the data points on the x-axis for all plots
    for ax in axes.flat:
        ax.scatter(data, np.zeros_like(data), color='black', marker='x', s=100, zorder=5, label='Data')
        ax.set_xlabel('x')
        ax.set_ylabel('Total Cost')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('tls_gnc_explanation.pdf', format='pdf')
    print("Figure saved as tls_gnc_explanation.pdf")

    plt.show()

if __name__ == "__main__":
    plot_robust_costs_gm()
