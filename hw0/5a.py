# Correcting the function and providing the necessary estimates

lambda_ = 3  # packages per hour
hours_per_day = 24
mu = [120, 4]  # mean values for S and W
Sigma = [[1.5, 1], [1, 1.5]]  # covariance matrix
noise_variance = 5

def simulate_T_star():
    total_packages = poisson.rvs(lambda_ * hours_per_day)  # Total packages in a day
    if total_packages > 0:
        sizes_weights = multivariate_normal.rvs(mean=mu, cov=Sigma, size=total_packages)
        epsilon = norm.rvs(loc=0, scale=np.sqrt(noise_variance), size=total_packages)
        T = 60 + 0.6 * sizes_weights[:, 1] + 0.2 * sizes_weights[:, 0] + epsilon
        return np.sum(T)
    else:
        return 0

# Simulate 1000 samples of T*
samples = [simulate_T_star() for _ in range(1000)]

# Calculate mean and standard deviation
mean_T_star = np.mean(samples)
std_T_star = np.std(samples)

mean_T_star, std_T_star