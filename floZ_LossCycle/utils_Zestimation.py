import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


# Function to select samples within 1$\sigma$ ellispoid B in latent space
def get_restricted_ellipsoidB(theta_samples, log_prob_theta_samples, delta_squared = 1.):
    # Assuming theta_samples is a 2D NumPy array with shape (num_samples, num_dimensions)
    # and log_prob_theta_samples is a 1D NumPy array with shape (num_samples,)
    
    # Step 1: Compute mean and covariance matrix
    mean_theta = np.mean(theta_samples, axis=0)
    covariance_matrix_theta = np.cov(theta_samples, rowvar=False)
    
    # Step 2: Find the mode (MAP estimate)
    initial_guess = np.zeros(theta_samples.shape[1])  # Replace with a suitable initial guess
    
    result = minimize(lambda x: -np.sum(-0.5 * np.log(np.linalg.det(covariance_matrix_theta)) - 0.5 * np.sum((theta_samples - x) @ np.linalg.inv(covariance_matrix_theta) * (theta_samples - x), axis=1) + log_prob_theta_samples),
                      initial_guess, method='L-BFGS-B')
    
    theta_MAP = result.x
    
    # # Print results
    # print(f"Mean of theta: {mean_theta}")
    # print(f"Covariance matrix of theta:\n{covariance_matrix_theta}")
    # print(f"MAP estimate of theta: {theta_MAP}")
    
    # Compute the Mahalanobis distance for each sample
    mahalanobis_distances_squared = np.sum((theta_samples - theta_MAP) @ np.linalg.inv(covariance_matrix_theta) * (theta_samples - theta_MAP), axis=1)
    
    # Identify samples that satisfy the condition
    theta_samples_in_B = theta_samples[mahalanobis_distances_squared < delta_squared]
    log_p_in_B = log_prob_theta_samples[mahalanobis_distances_squared < delta_squared]
    # # Print results
    print(f"Number of samples in set B: {len(theta_samples_in_B)} out of total {len(theta_samples)}")

    return mahalanobis_distances_squared < delta_squared


# Under Beta-Testing: A function to symmetrize samples about boundaries
def symmetrize(samples, log_p, ax=None, upper=False):
    ndims = samples.shape[1]
    min_theta = np.min(samples, axis = 0)
    max_theta = np.max(samples, axis = 0)
    
    symmetric_samples = np.array(samples)
    symmetric_log_p = np.array(log_p)
    
    n = np.random.uniform(size=(ndims, len(samples)))

    if ax is None:
        ax = range(ndims)
    
    for dim in ax:
        if not upper:
            symmetric_samples[n[dim]>.5, dim] = 2*min_theta[dim] - symmetric_samples[n[dim]>.5, dim]
            symmetric_log_p -= np.log(2)
        else:
            symmetric_samples[n[dim]>.5, dim] = 2*max_theta[dim] - symmetric_samples[n[dim]>.5, dim]
            symmetric_log_p -= np.log(2)
    return symmetric_samples, symmetric_log_p

# Under Beta-Testing: An alternative function to symmetrize samples about boundaries
def alternate_symmetrize(samples, log_p):
    ndims = samples.shape[1]
    min_theta = np.min(samples, axis = 0)
    max_theta = np.max(samples, axis = 0)
    
    symmetric_samples = np.array(samples)
    symmetric_log_p = np.array(log_p)
    
    n_lower = np.random.uniform(size=(ndims, len(samples)))
    n_upper = np.random.uniform(size=(ndims, len(samples)))

    # Calculate mean along each dimension
    mean = np.mean(samples, axis=0)
    # Calculate covariance matrix
    cov = np.cov(samples, rowvar=False)
    # Degrees of freedom (number of dimensions)
    df = ndims
    print("Mahalanobis distance to lower and upper boundaries; threshold = {:.2}".format(2*(ndims**.5)))
    
    def compute_pval_mahad(mean, cov, df, boundary):
        # Calculate Mahalanobis distance
        mahalanobis_distance = np.sqrt(np.dot(np.dot((boundary - mean), np.linalg.inv(cov)), (boundary - mean).T))
        # # Calculate the p-value corresponding to the Mahalanobis distance
        # p_value = 1 - chi2.cdf(mahalanobis_distance ** 2, df)

        # print("P-value:", p_value)
        
        # return p_value, mahalanobis_distance
    
        return mahalanobis_distance
    
    ax = range(ndims)
    for dim in ax:
        # Min boundary
        boundary = np.zeros_like(mean)
        boundary[dim] = min_theta[dim]
        mahalanobis_distance_lower = compute_pval_mahad(mean, cov, df, boundary)
        # if p_value>2*.05:
        
        # Max boundary
        boundary = np.zeros_like(mean)
        boundary[dim] = max_theta[dim]
        mahalanobis_distance_upper = compute_pval_mahad(mean, cov, df, boundary)
        # if p_value>2*.05:
        
        if mahalanobis_distance_lower < 1*(ndims**.5):
            symmetric_log_p -= np.log(2)
            symmetric_samples[n_lower[dim]>.5, dim] = 2*min_theta[dim] - symmetric_samples[n_lower[dim]>.5, dim]

        elif mahalanobis_distance_upper < 1*(ndims**.5):
            symmetric_log_p -= np.log(2)
            symmetric_samples[n_upper[dim]>.5, dim] = 2*max_theta[dim] - symmetric_samples[n_upper[dim]>.5, dim]

        print('axis {}: {:.2}, {:.2}'.format(dim, mahalanobis_distance_lower, mahalanobis_distance_upper))
            
    return symmetric_samples, symmetric_log_p