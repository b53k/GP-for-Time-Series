import numpy as np

class Elliptical_Slice_Sampler:
    '''
        Elliptical Slice Sampler Class
    '''
    def __init__(self, mean, covariance, log_likelihood_func):
        '''Initialize the parameters of the elliptical slice sampler object'''
        # Tip: This algorithm might hit an infinite loop, when the log_likelihood function is buggy.

        self.mean = mean
        self.covariance = covariance
        self.log_likehood_func = log_likelihood_func


    def __sample(self, f, log_f):
        '''
        Args:
            f: current state
            log_f: cached log-likelihood to speed-up the algorithm
        Returns:
            fp: new state
            log_fp: updated log-likelihood of new state
        '''
        # Choose ellipse:
        nu = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance)          # TODO: Cholesky Decomposition? 

        # Log-likelihood threshold:
        log_y = log_f + np.log(np.random.uniform())

        # Draw an initial proposal, also defining a bracker
        theta = np.random.uniform(low = 0.0, high = 2.0*np.pi)
        theta_min, theta_max = theta - 2.0*np.pi, theta

        # Iteratively shrink the bracket until you find a new acceptable point
        while True:
            # Genearates a point on the ellipse defined by 'nu' and the input. We
            # also computee the log-likelihood of the candidate and compare to our threshold
            fp = (f - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            log_fp = self.log_likehood_func(fp)

            if log_fp > log_y:
                return fp, log_fp
            
            if theta == 0.0:
                raise Exception('Theta reached zero due to numerical underflow. Check your log-likelihood function.')
            
            # Shrink the bracket and try a new point:
            if theta<0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = np.random.uniform(low = theta_min, high = theta_max)


    def sample(self, n_samples, burnin):

        total_samples = n_samples + burnin
        samples = np.zeros((total_samples, self.covariance.shape[0]))

        # First sample is a draw from the prior
        samples[0] = np.random.multivariate_normal(mean = self.mean, cov = self.covariance)         # TODO: Cholesky Decomposition?
        log_f = self.log_likehood_func(samples[0])

        for i in range(1, total_samples):
            samples[i], log_f = self.__sample(f = samples[i-1], log_f = log_f)

        return samples[burnin:]
    

# COMMENT:
# For high-dimensional distributions (as found in Gaussian process applications), 
# you'll want to compute a decomposition of the covariance (usually the Cholesky decomposition) once, and reuse it. 
# By using the built in multivariate normal sampler, you're re-decomposing the covariance (a cubic operation) on every step of the Markov chain. 
# Alternatively, pass in a function handle that knows how to sample nu, so a user can choose what covariance/precision representation to use, 
# and can exploit any sampling tricks they have for their setting.