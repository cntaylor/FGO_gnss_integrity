import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def l2_loss(r):
    """Standard Squared Error (L2) Loss."""
    return 0.5 * r**2

def l2_deriv(r):
    return r

def huber_loss(r, delta=1.345):
    """Huber Loss: Quadratic for small errors, Linear for large ones."""
    return np.where(np.abs(r) <= delta, 
                    0.5 * r**2, 
                    delta * (np.abs(r) - 0.5 * delta))

def huber_deriv(r, delta=1.345):
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

def cauchy_loss(r, c=2.3849):
    """Cauchy (Lorentzian) Loss: Logarithmic growth."""
    return (c**2 / 2) * np.log(1 + (r / c)**2)

def cauchy_deriv(r, c=2.3849):
    return r / (1 + (r / c)**2)

def geman_mcclure_loss(r, sigma=2.0):
    """Geman-McClure Loss: Saturates at large errors."""
    return (r**2 / 2) / (1 + (r / sigma)**2)

def geman_mcclure_deriv(r, sigma=2.0):
    return r / (1 + (r / sigma)**2)**2

def truncated_least_squares_loss(r, beta=3.0):
    """Truncated Least Squares: Constant for errors beyond beta."""
    return np.where(np.abs(r) <= beta, 0.5 * r**2, 0.5 * beta**2)

def truncated_least_squares_deriv(r, beta=3.0):
    return np.where(np.abs(r) <= beta, r, 0)

def generate_robust_plots(filename="Robust_Loss_Comparison.pdf"):
    # residuals range
    r = np.linspace(-10, 10, 1000)
    
    # Setup the PDF
    with PdfPages(filename) as pdf:
        
        # --- Page 1: The Cost Functions ---
        plt.figure(figsize=(10, 7))
        plt.plot(r, l2_loss(r), 'k--', label='L2 (Squared Error)', alpha=0.7)
        plt.plot(r, huber_loss(r), label=r'Huber ($c=1.345$)')
        plt.plot(r, cauchy_loss(r), label='Cauchy ($c=2.385$)')
        plt.plot(r, geman_mcclure_loss(r), label='Geman-McClure')
        plt.plot(r, truncated_least_squares_loss(r), label='Truncated LS ($c=3.0$)')
        
        plt.title('Comparison of Robust Cost Functions $\\rho(y)$', fontsize=14)
        plt.xlabel('Residual (y)', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.ylim(0, 15)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.savefig('RCF_plot1.pdf')
        plt.show()
        plt.close()
        
        # --- Page 2: The Influence Functions (Derivatives) ---
        plt.figure(figsize=(10, 7))
        plt.plot(r, l2_deriv(r), 'k--', label='L2 Influence', alpha=0.7)
        plt.plot(r, huber_deriv(r), label='Huber Influence')
        plt.plot(r, cauchy_deriv(r), label='Cauchy Influence')
        plt.plot(r, geman_mcclure_deriv(r), label='Geman-McClure Influence')
        plt.plot(r, truncated_least_squares_deriv(r), label='Truncated LS Influence')
        
        plt.title('Influence Functions $\\psi(y) = \\rho\'(y)$', fontsize=14)
        plt.xlabel('Residual (y)', fontsize=12)
        plt.ylabel('Influence', fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        # Add annotation explaining influence
        plt.text(-9, 8, "L2 influence grows linearly\n(Vulnerable to outliers)", 
                 bbox=dict(facecolor='white', alpha=0.5))
        plt.text(4.5, 2.5, "Robust influences are bounded\nor redescending", 
                 bbox=dict(facecolor='white', alpha=0.5))
        
        plt.savefig('RCF_plot2.pdf')
        plt.show()
        plt.close()

    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    generate_robust_plots()