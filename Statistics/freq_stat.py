### Bootstrap approximation of the sampling distribution of any estimator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Define the true model
def true_model(N, theta_star):
    """
    Generate data from the true model.
    Here, we assume a normal distribution with mean theta_star and variance 1.
    """
    return np.random.normal(loc=theta_star, scale=1, size=N)

# Step 2: Define the MLE estimator
def mle_estimator(data):
    """
    Compute the MLE for the mean of a normal distribution.
    """
    return np.mean(data)

# Step 3: Define the Fisher information
def fisher_information(theta_star):
    """
    Compute the Fisher information for the normal distribution with known variance.
    For N(θ, σ^2), the Fisher information is 1/σ^2.
    Here, σ^2 = 1, so F(θ*) = 1.
    """
    return 1

# Step 4: Simulate the sampling distribution
def simulate_sampling_distribution(theta_star, N, S):
    """
    Simulate the sampling distribution of the MLE.
    - theta_star: True parameter value.
    - N: Sample size for each dataset.
    - S: Number of synthetic datasets to generate.
    """
    estimates = []
    for _ in range(S):
        data = true_model(N, theta_star)  # Generate dataset
        theta_hat = mle_estimator(data)  # Compute MLE
        estimates.append(theta_hat)
    return np.array(estimates)

# Step 5: Compare to the theoretical asymptotic distribution
def plot_sampling_distribution(estimates, theta_star, N):
    """
    Plot the sampling distribution of the MLE and compare it to the theoretical asymptotic normal distribution.
    """
    # Compute theoretical asymptotic distribution
    F = fisher_information(theta_star)
    theoretical_mean = theta_star
    theoretical_std = np.sqrt(1 / (N * F))  # Standard deviation of the asymptotic normal distribution

    # Plot histogram of estimates
    plt.hist(estimates, bins=50, density=True, alpha=0.6, color='blue', label='Sampling Distribution')

    # Plot theoretical asymptotic normal distribution
    x = np.linspace(theoretical_mean - 3 * theoretical_std, theoretical_mean + 3 * theoretical_std, 1000)
    y = norm.pdf(x, loc=theoretical_mean, scale=theoretical_std)
    plt.plot(x, y, 'r-', label='Theoretical Asymptotic Distribution')

    # Add labels and legend
    plt.xlabel(r'$\hat{\theta}$')
    plt.ylabel('Density')
    plt.title(f'Sampling Distribution of MLE (N={N})')
    plt.legend()
    plt.show()

# Parameters
theta_star = 2.0  # True parameter value
N = 100  # Sample size for each dataset
S = 1000  # Number of synthetic datasets

# Simulate and plot
estimates = simulate_sampling_distribution(theta_star, N, S)
plot_sampling_distribution(estimates, theta_star, N)

# Generate a synthetic dataset
np.random.seed(42)
N = 100  # Sample size
true_mean = 5.0  # True mean of the normal distribution
true_std = 2.0  # True standard deviation of the normal distribution
data = np.random.normal(loc=true_mean, scale=true_std, size=N)  # Observed data

# Define the estimator (e.g., sample median)
def estimator(data):
    return np.median(data)

# Parametric Bootstrap
def parametric_bootstrap(data, S=1000):
    """
    Perform parametric bootstrap to estimate the sampling distribution of the estimator.
    - data: Observed dataset.
    - S: Number of bootstrap samples.
    """
    mu_hat = np.mean(data)  # Estimate mean
    sigma_hat = np.std(data)  # Estimate standard deviation
    
    bootstrap_estimates = []
    for _ in range(S):
        synthetic_data = np.random.normal(loc=mu_hat, scale=sigma_hat, size=N)
        theta_hat = estimator(synthetic_data)
        bootstrap_estimates.append(theta_hat)
    return np.array(bootstrap_estimates)

# Non-Parametric Bootstrap
def non_parametric_bootstrap(data, S=1000):
    """
    Perform non-parametric bootstrap to estimate the sampling distribution of the estimator.
    - data: Observed dataset.
    - S: Number of bootstrap samples.
    """
    bootstrap_estimates = []
    for _ in range(S):
        synthetic_data = np.random.choice(data, size=N, replace=True)
        theta_hat = estimator(synthetic_data)
        bootstrap_estimates.append(theta_hat)
    return np.array(bootstrap_estimates)

# Compute confidence intervals
def compute_confidence_interval(bootstrap_estimates, alpha=0.05):
    """
    Compute the confidence interval for the estimator using bootstrap estimates.
    - bootstrap_estimates: Array of bootstrap estimates.
    - alpha: Significance level (default: 0.05 for 95% CI).
    """
    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    return lower, upper

# Plot the bootstrap sampling distribution with confidence intervals
def plot_bootstrap_distribution(bootstrap_estimates, method, alpha=0.05):
    """
    Plot the bootstrap sampling distribution of the estimator with confidence intervals.
    - bootstrap_estimates: Array of bootstrap estimates.
    - method: String indicating the bootstrap method ("Parametric" or "Non-Parametric").
    - alpha: Significance level (default: 0.05 for 95% CI).
    """
    lower, upper = compute_confidence_interval(bootstrap_estimates, alpha)
    
    plt.hist(bootstrap_estimates, bins=50, density=True, alpha=0.6, color='blue', label='Bootstrap Distribution')
    plt.axvline(np.mean(bootstrap_estimates), color='red', linestyle='dashed', linewidth=2, label='Mean of Bootstrap Estimates')
    plt.axvline(lower, color='green', linestyle='dashed', linewidth=2, label=f'{100*(1-alpha)}% CI Lower Bound')
    plt.axvline(upper, color='green', linestyle='dashed', linewidth=2, label=f'{100*(1-alpha)}% CI Upper Bound')
    plt.xlabel('Estimator Value')
    plt.ylabel('Density')
    plt.title(f'{method} Bootstrap Sampling Distribution of the Median')
    plt.legend()
    plt.show()

# Parameters and execution
S = 1000  # Number of bootstrap samples
alpha = 0.05  # Significance level for 95% CI

# Run parametric bootstrap
parametric_estimates = parametric_bootstrap(data, S)
plot_bootstrap_distribution(parametric_estimates, "Parametric", alpha)

# Run non-parametric bootstrap
non_parametric_estimates = non_parametric_bootstrap(data, S)
plot_bootstrap_distribution(non_parametric_estimates, "Non-Parametric", alpha)

# Print confidence intervals
parametric_ci = compute_confidence_interval(parametric_estimates, alpha)
non_parametric_ci = compute_confidence_interval(non_parametric_estimates, alpha)

print(f"Parametric Bootstrap 95% CI: {parametric_ci}")
print(f"Non-Parametric Bootstrap 95% CI: {non_parametric_ci}")

class BootstrapAnalysis:
    def __init__(self, N=100, S=1000, alpha=0.05):
        """
        Initialize the bootstrap analysis.
        
        Parameters:
        - N: Sample size for each dataset
        - S: Number of bootstrap samples
        - alpha: Significance level for confidence intervals
        """
        self.N = N
        self.S = S
        self.alpha = alpha
        self.data = None
        
    def generate_data(self, true_mean=5.0, true_std=2.0, seed=42):
        """Generate synthetic dataset"""
        np.random.seed(seed)
        self.data = np.random.normal(loc=true_mean, scale=true_std, size=self.N)
        return self.data

    @staticmethod
    def estimator(data):
        """Default estimator using median"""
        return np.median(data)
    
    def parametric_bootstrap(self):
        """Perform parametric bootstrap"""
        if self.data is None:
            raise ValueError("Data not generated. Call generate_data() first.")
            
        mu_hat = np.mean(self.data)
        sigma_hat = np.std(self.data)
        
        bootstrap_estimates = []
        for _ in range(self.S):
            synthetic_data = np.random.normal(loc=mu_hat, scale=sigma_hat, size=self.N)
            theta_hat = self.estimator(synthetic_data)
            bootstrap_estimates.append(theta_hat)
        return np.array(bootstrap_estimates)

    def non_parametric_bootstrap(self):
        """Perform non-parametric bootstrap"""
        if self.data is None:
            raise ValueError("Data not generated. Call generate_data() first.")
            
        bootstrap_estimates = []
        for _ in range(self.S):
            synthetic_data = np.random.choice(self.data, size=self.N, replace=True)
            theta_hat = self.estimator(synthetic_data)
            bootstrap_estimates.append(theta_hat)
        return np.array(bootstrap_estimates)

    def compute_confidence_interval(self, bootstrap_estimates):
        """Compute confidence interval from bootstrap estimates"""
        lower = np.percentile(bootstrap_estimates, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_estimates, 100 * (1 - self.alpha / 2))
        return lower, upper

    def plot_bootstrap_distribution(self, bootstrap_estimates, method):
        """Plot bootstrap distribution with confidence intervals"""
        lower, upper = self.compute_confidence_interval(bootstrap_estimates)
        
        plt.figure(figsize=(10, 6))
        plt.hist(bootstrap_estimates, bins=50, density=True, alpha=0.6, 
                color='blue', label='Bootstrap Distribution')
        plt.axvline(np.mean(bootstrap_estimates), color='red', linestyle='dashed',
                   linewidth=2, label='Mean of Bootstrap Estimates')
        plt.axvline(lower, color='green', linestyle='dashed',
                   linewidth=2, label=f'{100*(1-self.alpha)}% CI Lower Bound')
        plt.axvline(upper, color='green', linestyle='dashed',
                   linewidth=2, label=f'{100*(1-self.alpha)}% CI Upper Bound')
        plt.xlabel('Estimator Value')
        plt.ylabel('Density')
        plt.title(f'{method} Bootstrap Sampling Distribution of the Median')
        plt.legend()
        plt.show()

    def run_analysis(self):
        """Run complete bootstrap analysis"""
        if self.data is None:
            self.generate_data()

        # Run parametric bootstrap
        parametric_estimates = self.parametric_bootstrap()
        self.plot_bootstrap_distribution(parametric_estimates, "Parametric")
        parametric_ci = self.compute_confidence_interval(parametric_estimates)

        # Run non-parametric bootstrap
        non_parametric_estimates = self.non_parametric_bootstrap()
        self.plot_bootstrap_distribution(non_parametric_estimates, "Non-Parametric")
        non_parametric_ci = self.compute_confidence_interval(non_parametric_estimates)

        # Print results
        print(f"Parametric Bootstrap {100*(1-self.alpha)}% CI: {parametric_ci}")
        print(f"Non-Parametric Bootstrap {100*(1-self.alpha)}% CI: {non_parametric_ci}")


# Example usage:
if __name__ == "__main__":
    # Create instance with default parameters
    bootstrap = BootstrapAnalysis(N=100, S=1000, alpha=0.05)
    
    # Generate data and run analysis
    bootstrap.generate_data(true_mean=5.0, true_std=2.0)
    bootstrap.run_analysis()
