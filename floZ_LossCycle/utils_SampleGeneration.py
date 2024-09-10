from __future__ import print_function
from scipy.special import erf
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import integrate
import random
from scipy.integrate import quad
import numpy as np
import pandas as pd
import os
from scipy.stats import norm

def get_priors(samples, min_theta, max_theta):
    # Calculate prior values for samples, constant prior
    # print('Flat prior: ln(Prior) = ', -np.log(np.prod(max_theta-min_theta)))
    return np.ones(len(samples))/np.prod(max_theta-min_theta)

def get_log_priors(samples, min_theta, max_theta):
    # Calculate prior values for samples, constant prior
    # print('Flat prior: ln(Prior) = ', -np.log(np.prod(max_theta-min_theta)))
    return np.zeros(len(samples))-np.sum(np.log(max_theta-min_theta))


###################### N Gaussian functions ######################

def calc_NGaussian_log_p_lnZ(N, samples, min_theta, max_theta, vars):
    likelihood = np.zeros(len(samples))
    # log_likelihood = np.zeros(len(samples))
    Z = 0
    log_prior = get_log_priors(samples, min_theta, max_theta)
    print('\n\nlnPRIOR = ', log_prior)
    for id in range(len(vars)):
        # Calculate likelihood values for all samples
        var = vars[id]
        likelihood += var.pdf(samples)
        # log_likelihood += var.logpdf(samples)
        Z += var.cdf(max_theta) - var.cdf(min_theta)
    likelihood /= N
    ln_Z = np.log(Z) + log_prior[0] - np.log(N)
    print('ln Evidence ', ln_Z)
    print('\n\nLikelihood = ', likelihood)
    # log_p = log (likelihood x prior)
    log_p = np.log(likelihood) + log_prior
    # log_p = log_likelihood + log_prior

    return log_p, ln_Z

def save_NGaussian(path, samples, log_p, limits, ln_Z, means, covs):
    if not os.path.exists(path):
        os.makedirs(path)
    # Saving samples, log(likelihood x prior), logZ
    np.savetxt(os.path.join(path, 'samples_logp.txt'), np.column_stack((samples, log_p)))
    np.savetxt(os.path.join(path, 'limits.txt'), limits) 
    np.savetxt(os.path.join(path, 'ln_Z.txt'), [ln_Z])
    # Saving likelihood properties
    np.save(os.path.join(path, 'means.npy'), means)
    np.save(os.path.join(path, 'covs.npy'), covs)

def load_NGaussian(path):
    s = np.loadtxt(os.path.join(path, 'samples_logp.txt'))
    samples = s[:, :-1]
    log_p = s[:, -1]
    ln_Z = np.loadtxt(os.path.join(path, 'ln_Z.txt'))
    limits = np.loadtxt(os.path.join(path, 'limits.txt'))
    means = np.load(os.path.join(path, 'means.npy'))
    covs = np.load(os.path.join(path, 'covs.npy'))
    return samples, log_p, ln_Z, limits, means, covs


def init_NGaussian(N, ndims, ninput, floZpath_ref, means = None, covs = None):
    samples = []
    vars = []
    meanmean = np.zeros(ndims)
    
    if(means is None):
        means = []
        covs = []
        for idx in range(N):
            # Generate random mean for each variable
            mean = np.random.uniform(1.,100.,ndims)
            if(ndims>15 and N>1):
                print('\n\nbig means')
                mean = np.random.uniform(1.,30.,ndims)
            
            # mean = np.zeros(ndims)
            
            print('Mean ', idx+1)
            print(mean)
            means += [mean]
            meanmean += mean
    
            # Generate random variances for each variable
            variances = np.random.uniform(2.8, 8., size=ndims)
            if(ndims>15):
                print('\nbig variances')
                variances = np.random.uniform(5, 10., size=ndims)
                
            # print('Variances')
            # print(variances)
            
            # Generate a random correlation matrix
            correlation_matrix = np.random.uniform(0, 1, size=(ndims, ndims))
            
            # Set diagonal elements of the correlation matrix to variances
            for i in range(ndims):
                correlation_matrix[i,i] = 1.*variances[i]
                
            # Ensure symmetry
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  
            # print("Correlation Matrix:")
            # print(correlation_matrix)
            
            # Generate the covariance matrix
            covariance_matrix = np.outer(variances, variances) * correlation_matrix
            
            # Make the matrix positive semidefinite
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            covariance_matrix_pos_def = np.dot(eigenvectors, np.dot(np.diag(np.maximum(eigenvalues, 0)), eigenvectors.T))
            
            cov = covariance_matrix_pos_def
            print("Covariance Matrix ", idx+1)
            print(cov)
            covs += [cov]
            
            # Assume flat prior and Gaussian likelihood. Posteriors can be sampled from likelihood, and we take parameter limits outside the sample space.
            samples += list(np.random.multivariate_normal(mean, cov, ninput//N))
            vars += [multivariate_normal(mean=mean, cov=cov)]

    else:
        for idx in range(N):
            mean = means[idx]
            cov = covs[idx]
            print('Mean ', idx+1)
            print(mean)
            print("Covariance Matrix ", idx+1)
            print(cov)
            meanmean += mean
            samples += list(np.random.multivariate_normal(mean, cov, ninput//N))
            vars += [multivariate_normal(mean=mean, cov=cov)]
    
    meanmean /= N
    
    samples = np.array(samples)
    # print(samples.shape)
    means = np.array(means)
    covs = np.array(covs)
    print('Shapes of means {}, and covariances {}'.format(means.shape, covs.shape))

    min_theta = np.min(samples, axis = 0)
    max_theta = np.max(samples, axis = 0)
    log_p, ln_Z = calc_NGaussian_log_p_lnZ(N, samples, min_theta, max_theta, vars)
    print('ln Z = ', ln_Z)
    
    limits = [min_theta, max_theta]
    print('Limits ', limits)
    save_NGaussian(floZpath_ref, samples, log_p, limits, ln_Z, means, covs)


###################### Rosenbrock functions ######################

def log_pdf_rosenbrock(x, A=100, B=20):
    x = x.T
    return -np.sum(A*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2, axis = 0)/B

# Define the Rosenbrock function
def rosenbrock(x, y):
    return np.exp(log_pdf_rosenbrock(np.array([x,y])))

# Define proposal distribution (Gaussian with diagonal covariance matrix)
def proposal(x, sigma=0.1):
    return np.random.normal(loc=x, scale=sigma)

# Metropolis-Hastings algorithm
def metropolis_hastings(n_samples, initial_state, sigma=0.1):
    samples = [initial_state]
    current_state = initial_state
    for _ in range(n_samples):
        proposed_state = proposal(current_state, sigma)
        acceptance_ratio = min(1, np.exp((-log_pdf_rosenbrock(current_state.T) + log_pdf_rosenbrock(proposed_state.T))))
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
        samples.append(current_state)
    return np.array(samples)

def calc_Rosenbrock_log_p_Z(samples, min_theta, max_theta, ndims):
    # Calculate likelihood values for samples
    likelihood = np.exp(log_pdf_rosenbrock(x=samples))

    prior = get_priors(samples, min_theta, max_theta)

    # log_p = log (likelihood x prior)
    log_p = np.log(likelihood) + np.log(prior)
    
    # Calculation of Evidence
    if(ndims < 3):
        # Define the limits of integration
        x_lower, x_upper = min_theta[1], max_theta[1]
        y_lower, y_upper = min_theta[0], max_theta[0]
        result, error = integrate.dblquad(rosenbrock, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper)
        print("Integral of Rosenbrock function over the specified domain:", result)
        Z = result/np.prod(max_theta-min_theta)
    else:
        print('Dim > 2, Rosenbrock not integratable: numerically intractable')
        Z = np.nan
    return log_p, Z
    
def save_Rosenbrock(path, samples, log_p, limits, Z):
    if not os.path.exists(path):
        os.makedirs(path)
    # Saving samples, log(likelihood x prior), logZ
    np.savetxt(os.path.join(path, 'samples_logp.txt'), np.column_stack((samples, log_p)))
    np.savetxt(os.path.join(path, 'limits.txt'), limits) 
    np.savetxt(os.path.join(path, 'Z.txt'), [Z])

def load_Rosenbrock(path):
    s = np.loadtxt(os.path.join(path, 'samples_logp.txt'))
    samples = s[:, :-1]
    log_p = s[:, -1]
    Z = np.loadtxt(os.path.join(path, 'Z.txt'))
    limits = np.loadtxt(os.path.join(path, 'limits.txt'))
    return samples, log_p, Z, limits
    
    
def init_Rosenbrock(ndims, ninput, floZpath_ref):
    # Initial state
    initial_state = np.zeros(ndims)
    
    # Generate MCMC samples
    samples = metropolis_hastings(ninput, initial_state)
    max_theta = np.max(samples, axis = 0)
    min_theta = np.min(samples, axis = 0)

    log_p, Z = calc_Rosenbrock_log_p_Z(samples, min_theta, max_theta, ndims)
    
    print('log Z ', np.log(Z))

    limits = [min_theta, max_theta]
    print('Limits ', limits)
    save_Rosenbrock(floZpath_ref, samples, log_p, limits, Z)


###################### Exponential functions ###########################

def generate_n_dimensional_exponential(N, lambdas, size):
    """
    Generate an N-dimensional exponential distribution with different lambdas for each dimension.

    Parameters:
    - N: Dimensionality of the distribution.
    - lambdas: Array of rate parameters for each dimension.
    - size: Number of samples to generate.

    Returns:
    - ndarray: N-dimensional array of samples from the exponential distribution.
    """
    samples = np.random.exponential(scale=1 / lambdas, size=(size, N))
    return samples


def n_dimensional_exponential_cdf(x, lambdas):
    """
    Compute the cumulative distribution function (CDF) of an N-dimensional exponential distribution.

    Parameters:
    - x: Numpy array of values at which to compute the CDF for each dimension (shape: (N, M)).
    - lambdas: Numpy array of rate parameters for each dimension (shape: (N,)).

    Returns:
    - ndarray: CDF values for the N-dimensional distribution at the input x (shape: (M,)).
    """
    
    cdf_values = 1 - np.exp(-lambdas * x)
    return cdf_values
    # return np.prod(cdf_values, axis=0)


def n_dimensional_exponential_pdf(x, lambdas):
    """
    Calculate the probability density for an N-dimensional exponential distribution.

    Parameters:
    - x: N-dimensional array of samples.
    - lambdas: Array of rate parameters for each dimension.

    Returns:
    - float: Probability density for the given sample.
    """
    # pdf_values = lambdas * np.exp(-lambdas * x)
    pdf_values = np.ones(x.shape[0])
    for dim in range(lambdas.shape[0]):
        pdf_values *= lambdas[dim] * np.exp(-lambdas[dim] * x[:,dim])
    return pdf_values
    # return np.prod(pdf_values, axis=1)
    

def calc_Exponential_log_p_Z(samples, min_theta, max_theta, lambdas):
    # Calculate likelihood values for samples
    log_likelihood = np.log(n_dimensional_exponential_pdf(x=samples, lambdas=lambdas))
    
    prior = get_priors(samples, min_theta, max_theta)

    # log_p = log (likelihood x prior)
    log_p = log_likelihood + np.log(prior)
    
    # Calculation of Evidence
    # Z = (n_dimensional_exponential_cdf(x=max_theta, lambdas=lambdas) - n_dimensional_exponential_cdf(x=min_theta, lambdas=lambdas)) *prior[0]
    Z = (n_dimensional_exponential_cdf(x=max_theta, lambdas=lambdas) - n_dimensional_exponential_cdf(x=min_theta, lambdas=lambdas)).prod(axis = 0) *prior[0]
    return log_p, Z
    
def save_Exponential(path, samples, log_p, limits, Z, lambdas):
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Saving samples, log(likelihood x prior), logZ
    np.savetxt(os.path.join(path, 'samples_logp.txt'), np.column_stack((samples, log_p)))
    np.savetxt(os.path.join(path, 'limits.txt'), limits) 
    # Saving evidence
    np.savetxt(os.path.join(path, 'Z.txt'), [Z])
    # Saving prior bounds
    np.savetxt(os.path.join(path, 'limits.txt'), limits)
    # Saving likelihood properties
    np.savetxt(os.path.join(path, 'lambdas.txt'), lambdas)

def load_Exponential(path):
    s = np.loadtxt(os.path.join(path, 'samples_logp.txt'))
    samples = s[:, :-1]
    log_p = s[:, -1]
    Z = np.loadtxt(os.path.join(path, 'Z.txt'))
    limits = np.loadtxt(os.path.join(path, 'limits.txt'))
    lambdas = np.loadtxt(os.path.join(path, 'lambdas.txt'))
    return samples, log_p, Z, limits, lambdas


def init_Exponential(ndims, ninput, floZpath_ref, lambdas = None):
    # Generate random mean for each variable
    if(lambdas is None):
        lambdas = np.random.uniform(0,1e-2,ndims)
    
    lambdas = np.round(lambdas, decimals=6)
    print('Lambdas')
    print(lambdas)

    samples = generate_n_dimensional_exponential(N=ndims, lambdas=lambdas, size=ninput)

    min_theta = np.min(samples, axis = 0)
    max_theta = np.max(samples, axis = 0)
    
    # Exponential should have lower limit at 0. As density is higest here, small changes can lead to different evidence calculation=.
    min_theta *= 0
    
    log_p, Z = calc_Exponential_log_p_Z(samples, min_theta, max_theta, lambdas)
    print('ln Z = ', np.log(Z))
    
    limits = [min_theta, max_theta]
    print('Limits ', limits)
    save_Exponential(floZpath_ref, samples, log_p, limits, Z, lambdas)


###################### Funnel functions ######################

def save_Funnel(path, samples, log_p, limits, ln_Z):
    if not os.path.exists(path):
        os.makedirs(path)
    # Saving samples, log(likelihood x prior), logZ
    np.savetxt(os.path.join(path, 'samples_logp.txt'), np.column_stack((samples, log_p)))
    np.savetxt(os.path.join(path, 'limits.txt'), limits) 
    np.savetxt(os.path.join(path, 'ln_Z.txt'), [ln_Z])
    

def load_Funnel(path):
    s = np.loadtxt(os.path.join(path, 'samples_logp.txt'))
    samples = s[:, :-1]
    log_p = s[:, -1]
    ln_Z = np.loadtxt(os.path.join(path, 'ln_Z.txt'))
    limits = np.loadtxt(os.path.join(path, 'limits.txt'))
    return samples, log_p, ln_Z, limits


def calc_Funnel_log_p_lnZ(samples, min_theta, max_theta, a, b):
    n_samples, n_dim = samples.shape
    log_p = np.zeros(n_samples)
    
    for j in range(n_samples):
        x1 = samples[j, 0]
        # Log-Likelihood for x1
        log_likelihood_x1 = norm.logpdf(x1, 0, a)
        
        # Log-Likelihood for x2 to xn
        log_likelihood_xi = 0.0
        for i in range(1, n_dim):
            variance = np.exp(2 * b * x1)
            log_likelihood_xi += norm.logpdf(samples[j, i], 0, np.sqrt(variance))
        
        # Sum log-likelihoods
        log_p[j] = log_likelihood_x1 + log_likelihood_xi
        
        # Compute log-prior
        log_prior_x1 = np.log(1/8) if -4 <= x1 <= 4 else -np.inf  # Flat prior for x1 over [-4, 4]
        log_prior_xi = np.sum([np.log(1/60) if -30 <= samples[j, i] <= 30 else -np.inf for i in range(1, n_dim)])  # Flat prior for x2 to xn over [-30, 30]
        
        # Sum log-likelihood and log-prior
        log_p[j] += log_prior_x1 + log_prior_xi


    # Integral of the likelihood over the prior range
    # For simplicity, we'll assume the normalizing constant is the volume of the prior space
    prior_x1_range = 8  # [-4, 4]
    prior_xi_range = 60  # [-30, 30] for each x_i, i=2 to n_dim
    
    # Calculate the volume of the prior space
    log_prior = np.log(prior_x1_range) + np.log(prior_xi_range)*(n_dim - 1)
    
    # Log-normalizing constant
    log_z = log_prior
    
    return log_p, log_z

    
def init_Funnel(ndims, ninput, floZpath_ref, a = 1, b = .5):

    # Example usage:
    n_samples = 1000
    # Preallocate space for samples
    samples = np.zeros((ninput, ndims))
    
    # Sample x1 from a uniform distribution between -4 and 4
    x1_samples = np.random.uniform(-4, 4, ninput)
    samples[:, 0] = x1_samples
    
    # Sample x2 to xn from a normal distribution with mean 0 and variance exp(2bx1)
    for i in range(1, ndims):
        variance = np.exp(2 * b * x1_samples)
        samples[:, i] = np.random.normal(0, np.sqrt(variance))
        
    samples = np.array(samples)
    print(samples.shape)
    
    min_theta = np.array([-4] + (ndims-1)*[-30])#np.min(samples, axis = 0)
    max_theta = np.array([4] + (ndims-1)*[30])#np.max(samples, axis = 0)
    print(min_theta)
    
    log_p, ln_Z = calc_Funnel_log_p_lnZ(samples, min_theta, max_theta, a=a, b=b)
    print('ln Z = ', ln_Z)
    
    limits = [min_theta, max_theta]
    print('Limits ', limits)
    save_Funnel(floZpath_ref, samples, log_p, limits, ln_Z)