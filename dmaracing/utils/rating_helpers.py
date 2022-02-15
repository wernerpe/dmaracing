import numpy as np
from scipy.stats import norm

def compute_winprob_i(ratings_mu, ratings_sigma, i, beta = np.sqrt(25.0/6)):
    mu_i = ratings_mu[i]
    sigma_i = ratings_sigma[i] + beta

    #1v1probs
    winprob = 1.0
    mu_other = np.hstack((np.array(ratings_mu)[:i], np.array(ratings_mu)[i+1:]))
    sigma_other = np.hstack((np.array(ratings_sigma)[:i], np.array(ratings_sigma)[i+1:])) + beta
    
    for mu_idx, sigma_idx in zip(mu_other, sigma_other):
        mu_pair = mu_i-mu_idx
        var_pair = sigma_i**2 + sigma_idx**2
        winprob *= 1 - norm.cdf((0-mu_pair)/(np.sqrt(2*var_pair)))
    
    return winprob