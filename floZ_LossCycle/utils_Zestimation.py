import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


# Function to select samples within 1$\sigma$ ellispoid B in latent space
def get_restricted_ellipsoidB(latent_samples, log_zeta, delta_squared=None):
    """
    Restrict to the samples whose image in the LATENT space of the flow lies inside the ball
    |z|^2 < delta_squared (default: the number of dimensions, i.e. the 1-sigma region of the
    base Gaussian; a fraction chi2.cdf(d, d) ~ 0.4-0.6 of the samples). The flow is more accurate
    in the bulk, so the evidence is estimated there.

    latent_samples : (N, d) array, flow._transform(whitened samples)
    log_zeta       : (N,) array, log p - log q
    Returns (log_zeta restricted, boolean mask).
    """
    latent_samples = np.asarray(latent_samples); log_zeta = np.asarray(log_zeta)
    d = latent_samples.shape[1]
    if delta_squared is None:
        delta_squared = float(d)
    mask = np.sum(latent_samples ** 2, axis=1) < delta_squared
    print(f"Number of samples in set B: {mask.sum()} out of total {len(mask)} (expected fraction {chi2.cdf(delta_squared, d):.3f})")
    return log_zeta[mask], mask


# Under Beta-Testing: A function to symmetrize samples about boundaries
def symmetrize(samples, log_p, ax, bounds=None, upper=False):
    """
    Reflect a random half of the samples about a sharp boundary of the posterior, so that the flow
    does not have to fit a discontinuity. Reflecting about the edge a of dimension k produces the
    density p_s(x) = p(x)/2 + p(2a - x)/2, which equals p/2 at every kept or reflected point (the
    two supports are disjoint) and integrates to the same Z: hence log 2 is subtracted from log p
    for every reflected dimension.

    ax     : the dimensions that have a sharp edge (REQUIRED: reflecting a dimension without one
             turns a smooth posterior into a bimodal one with a cusp)
    bounds : the edge per dimension in `ax` (e.g. the prior bound). Default: the sample min (or
             max, if upper) of that dimension, which is only an O(1/N) approximation of the edge.
    upper  : reflect about the upper edge instead of the lower one.
    """
    ndims = samples.shape[1]
    if ax is None:
        raise ValueError("symmetrize: pass the list of dimensions with a sharp edge (ax); "
                         "reflecting every dimension is almost never what you want")
    ax = list(ax)
    symmetric_samples = np.array(samples, dtype=np.float64)
    symmetric_log_p = np.array(log_p, dtype=np.float64)
    if bounds is None:
        edge = (np.max(samples, axis=0) if upper else np.min(samples, axis=0))
        print("symmetrize: reflecting about the sample " + ("max" if upper else "min") + " (pass bounds= for the true edge)")
    else:
        edge = np.zeros(ndims)
        for k, b in zip(ax, np.atleast_1d(bounds)):
            edge[k] = b
    n = np.random.uniform(size=(ndims, len(samples)))
    for dim in ax:
        flip = n[dim] > .5
        symmetric_samples[flip, dim] = 2 * edge[dim] - symmetric_samples[flip, dim]
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
        # Min boundary: the point of the mean with this coordinate at the edge
        boundary = mean.copy()
        boundary[dim] = min_theta[dim]
        mahalanobis_distance_lower = compute_pval_mahad(mean, cov, df, boundary)
        # if p_value>2*.05:
        
        # Max boundary
        boundary = mean.copy()
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