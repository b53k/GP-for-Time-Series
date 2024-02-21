import torch
import numpy as np
import pyro
import pyro.distributions as dist
import argparse

from ESS import Elliptical_Slice_Sampler

obs = torch.load('Canada_data.pt')

def parse_arguments():
    '''
    Command line argument parser
    '''
    parser = argparse.ArgumentParser('Sequential GP for Online Learning')

    # Hyper-parameters
    parser.add_argument('--a0', type = float, default = 1.0, help = 'Parameter for Gamma distribution to initialize alpha')
    parser.add_argument('--b0', type = float, default = 1.0, help = 'Parameter for Gamma distribution to initialize alpha')

    parser.add_argument('--mu0', type = float, default = 0.0, help = 'Mean for multivariate-t distribution used in Eqn 1')
    parser.add_argument('--Psi0', type = int, default = 1, help = 'Covariance for multivariate-t distribution used in Eqn 1')
    parser.add_argument('--nu0', type = float, default = 5.0, help = 'DoF for multivariate-t distribution used in Eqn 1')
    parser.add_argument('--lambda0', type = float, default = 0.0, help = 'Scaler for mu0 for multivariate-t distribution used in Eqn 1')

    parser.add_argument('--m0', type = str, default = '0.0, 0.0', help = 'Prior (theta_mean, sigma_mean) for log-normally distributed of k-th mixture')
    parser.add_argument('--s02', type = float, default = 0.25, help = 'Prior for log-normally distributed variance of k-th mixture')


    args = parser.parse_args()
    return args

# to call parse_aruments:
args = parse_arguments()

# Initialize mean and var --> m0 and s0^2
m0 = args.m0.split(',')
mean_vector = np.array([float(m0[0]), float(m0[1])])   # [m0_theta, m0_sigma]
covariance_matrix = args.s02 * np.identity(2)


def log_likelihood_func():
    pass


# Instantiate the Elliptical Slice Sampler
ESS = Elliptical_Slice_Sampler(mean = mean_vector, covariance = covariance_matrix, log_likelihood_func = log_likelihood_func)



def Chinese_Restaurant_Prior(args, alpha, cluster_counts, x, prev_observations, assignments):
    '''
        Args:
            args: See parse_arguments() for details
            alpha: Concentration parameter for Dirichlet Process (DP)
            cluster_counts: Maintains the count of data points in each cluster. Its index gives the corresponding cluster ID
            x: Current observation --- must be a tensor
            prev_observations: List of previous observations
            assignments: Keeps Track of the cluster assignments for each data point -- tells how data points are distributed across clusters
        
        Returns:
            zi: Categorical value representing to which cluster x belongs 

    '''

    n_clusters = len(cluster_counts)
    # probs holds probability based on existing cluster sizes and the t-distribution for each existing cluster i.e. probs[0], prob[1],...prob[n_clusters-1]
    probs = torch.zeros(n_clusters + 1)

    assignments_tensor = torch.tensor(assignments)
    observations_tensor = torch.tensor(prev_observations)

    for k in range(n_clusters):
        # Get data-points corresponding to cluster 'k'
        indices = (assignments_tensor == k).nonzero(as_tuple=True)[0]
        X_k_prime = observations_tensor[indices]

        N_k_prime = cluster_counts[k]               # No. of Data Points in Cluster k until (i-1)th observation
        nu_k_prime = args.nu0 + N_k_prime           # D = 1 since the data is 1-D
        lambda_k_prime = args.lambda0 + N_k_prime

        assert len(X_k_prime) == N_k_prime, 'Length of Tensor Xk must equal No. of Data Points in Cluster {}'.format(k)

        xk_bar_prime = torch.sum(X_k_prime)/N_k_prime                                                  # TODO: Need to include present observation?
        S_k_prime = torch.sum(X_k_prime ** 2) - 2 * xk_bar_prime * torch.sum(X_k_prime) + N_k_prime * (xk_bar_prime ** 2)
        S_xk_prime = (args.lambda0 * N_k_prime * (xk_bar_prime - args.mu0)**2)/lambda_k_prime
        frac = (lambda_k_prime + 1)/(lambda_k_prime * nu_k_prime)
        add = args.Psi0 + S_k_prime + S_xk_prime

        mu_k_prime = (args.lambda0 * args.mu0 + N_k_prime * xk_bar_prime) / lambda_k_prime            # TODO: check if its x_bar or x_bar_prime.
        Psi_k_prime = frac * add
        
        # Probability of data falling in existing k-th cluster
        probs[k] = N_k_prime * dist.MultivariateStudentT(df = torch.tensor([nu_k_prime]), 
                                                         loc = torch.tensor([mu_k_prime]), 
                                                         scale_tril = torch.tensor([[Psi_k_prime]])).log_prob(x).exp()
    
    # Probability of Forming a New Cluster
    probs[-1] = alpha * dist.MultivariateStudentT(df = torch.tensor([args.nu0]), 
                                                  loc = torch.tensor([args.mu0]), 
                                                  scale_tril = torch.tensor([[args.Psi0]])).log_prob(x).exp()

    # Normalize
    probs /= probs.sum()

    return pyro.sample('z', dist.Categorical(probs))




def get_alpha(args, alpha, i, K):
    '''
        Args:
            args: See parse_arguments() for details
            alpha: Concentration parameter for Dirichlet Process (DP) for each paricle j
            i: i-th observation
            K: Total number of clusters
        Returns:
            alpha: Updated alpha
            
    '''
    
    N = i+1 # Total number of observations
    rho = pyro.sample('beta', dist.Beta(alpha+1, N))
    numerator = args.a0 + K - 1
    denominator = args.b0 - torch.log(rho)
    ratio = numerator/(N*denominator)
    pi_a = ratio/(1+ratio)
    Gamma1 = (1 - pi_a)*pyro.sample('Gamma', dist.Gamma(numerator, denominator))
    Gamma2 = pi_a * pyro.sample('Gamma', dist.Gamma(numerator+1, denominator))
    alpha = Gamma1 + Gamma2

    return alpha
    



def GP_MOE(args, observations):
    '''
        Args:
            args: See parse_arguments() for details
            observations: Input data - Sequential (one-at-a-time)---but this time for test purpose it is a tensor of many observations

        Returns:
            paticle weights and particles
    '''

    X = []  # Holds previous observations
    cluster_counts = []  # Maintains the count of data points in each cluster. Its index gives the corresponding cluster ID
    assignments = [] # Keeps Track of the cluster assignments for each data point -- tells how data points are distributed across clusters

    alpha = pyro.sample('Gamma', dist.Gamma(args.a0, args.b0))      # Initial concentration parameter for Dirichlet Process

    for i, data in enumerate(observations):
        xi, yi = data[0], data[1]

        if i == 0:
            # Initialize New Cluster
            zi = pyro.sample('z_0', dist.Categorical(torch.tensor([1.0])))
            cluster_counts.append(1)

            # Initialize weight_1 for each j-th particle        # TODO: Need to complete this

        else:
            zi = Chinese_Restaurant_Prior(args = args, 
                                         alpha = alpha, 
                                         cluster_counts = cluster_counts, 
                                         x = xi.unsqueeze(-1), 
                                         prev_observations = X, 
                                         assignments = assignments)
            
            if zi == len(cluster_counts):
                # Implies that a new cluster has been created
                cluster_counts.append(1)
            else:
                # Implies that xi belongs to an existing cluster
                cluster_counts[zi] += 1

        # Θ_{zi}: Elliptical Slice Sampler --> Get θ_k & σ_k
        # Exponentiate the samples to get log-normal samples,                   #TODO: Might need to make sure if this is a correct approach.
        theta_k, sigma_k_sq = np.exp(ESS.sample(n_samples = 1, burnin = 300))

        # α: From Equation (3)
        alpha = get_alpha(args = args, 
                          alpha = alpha, 
                          i = i, 
                          K = len(cluster_counts))
        
        # ω: Update weight Equation (4)                   # TODO: Complete this





        X.append(xi)
        assignments.append(zi)

    return assignments