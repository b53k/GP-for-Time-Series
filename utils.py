import numpy as np
import scipy
from scipy.linalg import expm, det, inv
from scipy.integrate import quad
import copy

#=====================================================================================================================================================
## Infinte-Horizon Gaussian Process -- Hyperparameter Optimization
#=====================================================================================================================================================

def get_dF(i, l):
    if i == 0 or i == 1:
        dF = np.zeros((2,2))
    elif i == 2:
        dF = np.array([[0, 0], [6/l**3, 2 * np.sqrt(3)/l**2]])
    return dF

def get_dA(i, l, dt):
    dF = get_dF(i, l)
    dA = scipy.linalg.expm(dF * dt)
    return dA

def get_dP_inf(i, sigma, l):
    if i == 0:
        dP_inf = np.zeros((2,2))
    elif i == 1:
        dP_inf = np.array([[1, 0],[0, 3/l**2]])
    elif i == 2:
        dP_inf = np.array([[0, 0], [0, -6*sigma/l**3]])
    return dP_inf

def get_dQ(i, sigma, l, dt, A, P_inf):
    dP_inf = get_dP_inf(i, sigma, l)
    dA = get_dA(i, l, dt)
    dQ = dP_inf - dA @ P_inf @ A - A.T @ dP_inf @ A - A.T @ P_inf @ dA
    return dQ


def optimize_hyper(initial_theta, Y_k, dt, eta, tolerance, max_iterations, m_k_prev):
    '''
    Args: 
        initial_theta: (shape: 3x1) hyperparmeters. [σ_n^2, σ^2, ℓ]
        Y_k: Window of latest w number of observations
        dt: t_{k} - t_{k-1}
        eta: learning rate
        tolerance: tolerance value to terminate optimization
        max_iterations: maximum iterations before terminating
        m_k_prev: previous filter conditional mean
    Returns:
        theta: updated theta
    '''
    sigma_n_2, sigma_2, l = initial_theta.reshape(-1)
    theta = np.abs(initial_theta)
    iteration = 0
    m = m_k_prev

    MF = np.zeros((2, len(Y_k)))
    PF = np.zeros((2,2, len(Y_k)))
    MP = np.zeros((2, len(Y_k)))
    PP = np.zeros((2,2, len(Y_k)))
    GS = np.zeros((2, 2, len(Y_k)))

    while True:

        F = np.array([[0.0, 1.0],[-3/(l**2), -2*np.sqrt(3)/np.abs(l)]])
        Qc = 12 * np.sqrt(3) * sigma_2 / l**3                                           
        C = np.array([1,0]).reshape(1,2)   
        P_inf = np.array([[(np.sqrt(np.abs(sigma_2))), 0.0], [0.0, 3 * (np.sqrt(np.abs(sigma_2)))/l**2]])   # Force sigma_2 to be positive

        # Solve LTI SDE 
        A = scipy.linalg.expm(F * dt)
        Q = P_inf - A @ P_inf @ np.transpose(A)

        #print ('\nP_inf: \n', P_inf)
        #print ('A: \n', A)
        #print ('Q: \n', Q)
        #norm = np.linalg.norm(Q - Q.conj().T,1)
        #spacing = np.spacing(np.linalg.norm(Q,1)) ** 100
        #print ('norm: ',norm)
        #print ('spacing: \n', spacing)
        
        R = sigma_n_2

        # Solve Discrete Algebriac Riccati Equation (DARE) to get P
        P_dare = scipy.linalg.solve_discrete_are(A, C.T, Q, R)
        P = P_inf
        
        for i, y_k in enumerate(Y_k):
            # Kalman Prediction
            mp = A @ m
            Pp = A @ P @ A.T + Q

            # Pre-calculate smoother Gain
            Gs = np.linalg.lstsq(Pp.T, (P @ A.T).T, rcond = None)[0].T

            # Kalman Update
            v = (y_k - C @ m).astype(np.float32)                      #TODO: Check if its mp or m
            S = C @ P @ C.T + R                                       #TODO: Check if its Pp or P
            K = np.linalg.lstsq(S.T, (P @ C.T).T, rcond = None)[0].T
            m += K * v
            P -= K @ C @ P

            MF[:, i] = m.T
            PF[:, :, i] = P
            MP[:, i] = mp.T
            PP[:, :, i] = Pp
            GS[:, :, i] = Gs

        # Run RTS smoother
        MS = copy.deepcopy(MF)
        PS = copy.deepcopy(PF)
        ms = MS[:, -1]
        Ps = PS[:,:,-1]

        for j in range(len(Y_k)-2, -1, -1):
            ms = MF[:, j] + GS[:, :, j+1] @ (ms - MP[:, j+1])
            Ps = PF[:,:,j] + GS[:,:,j+1] @ (Ps - PP[:,:,j+1]) @ (GS[:,:,j+1]).T
            MS[:,j] = ms
            PS[:,:,j] = Ps
        
            
        # Get 'm'
        m_ = MS
        
        # Get Likelihood and gradient from a different routine                     
        L, grad = gradient(A, C, P_dare, Y_k, theta, m_, dt)

        # Update theta
        new_theta = theta + eta * grad
        iteration += 1

        # Check for convergence
        if np.linalg.norm(new_theta - theta) <= tolerance or iteration >= max_iterations:
            break
        
        theta = np.abs(new_theta)
        sigma_n_2, sigma_2, l = theta.reshape(-1)

        #print ('theta:\n', theta)

    return theta
    

def gradient(A, C, P, Y, theta, m, dt):
    '''
    Compute the Likelihood and its gradient
    '''
    sigma_n_2, sigma_2, l = theta.reshape(-1,1)
    sigma_n_2 = sigma_n_2[0]
    sigma_2 = sigma_2[0]
    l = l[0]

    #P_inf = np.array([[np.sqrt(sigma_2), 0], [0, 3 * np.sqrt(sigma_2)/ l]])
    P_inf = np.array([[np.sqrt(np.abs(sigma_2)), 0.0], [0.0, 3 * np.sqrt(np.abs(sigma_2))/l**2]])
    R = sigma_n_2
    
    s_hat = C @ P @ C.T + R
    K = np.linalg.lstsq(s_hat.T, C @ P.T, rcond=None)[0].T
    dR = np.array([1.0, 0.0, 0.0]).reshape(-1,1)

    dS = []
    dK = []
    for i in range(3):
        dA = get_dA(i, l, dt)
        dQ = get_dQ(i, sigma = np.sqrt(np.abs(sigma_2)), l = l, dt = dt, A = A, P_inf = P_inf)

        s_hat_inv = np.linalg.lstsq(s_hat, C @ P @ A, rcond=None)[0]
        A_bar = A - C.T @ s_hat_inv


        Q_bar_i = A.T @ P @ C.T @ np.linalg.lstsq(s_hat, dR[i].reshape(-1,1), rcond = None)[0] @ s_hat_inv + dQ


        # Solve Discrete Lyapunov
        #P'_i = dlyap(A_bar.T , Q_bar_i)

        P_prime = scipy.linalg.solve_discrete_lyapunov(A_bar.T, Q_bar_i)

        assert P_prime.shape == A.shape, 'Shape of A and B are not the same'

        dSi = C @ P_prime @ C.T + dR[i].reshape(-1,1)
        dKi = P_prime @ C.T / s_hat[0][0] - P @ C.T @ dSi / (s_hat[0][0])**2

        dS.append(dSi)
        dK.append(dKi)

    dS = np.array(dS)
    dK = np.array(dK)

    dS = dS.reshape(-1,1)

    N = len(Y)
    L = N/2 * np.log(2 * np.pi) + N/2 * np.log(s_hat[0][0])
    grad = N/2 * dS / s_hat[0][0]
    

    dm = np.zeros((3,2,1)) # place holder for dm vector
    for i, yi in enumerate(Y):
        mi = m[:,i].reshape(-1,1)       # Shape = (2,1)
        assert mi.shape == (2,1), 'mi shape needs to be (2,1)'
        v = yi - (C @ A @ mi)[0][0]        # scalar
        L += 0.5* v**2/s_hat[0][0]         # scalar

        dv0 = -C @ get_dA(0, l, dt) @ mi - C @ A @ dm[0]
        dv1 = -C @ get_dA(1, l, dt) @ mi - C @ A @ dm[1]
        dv2 = -C @ get_dA(2, l, dt) @ mi - C @ A @ dm[2]

        dv = np.array([dv0, dv1, dv2]).reshape(-1,1)

        dm[0] = A @ K @ C @ A @ dm[0] + dK[0] @ np.array(yi).reshape(-1,1)
        assert dm[0].shape == (2,1), 'dm[0] shape must be (2,1)'
        dm[1] = A @ K @ C @ A @ dm[1] + dK[1] @ np.array(yi).reshape(-1,1)
        dm[2] = A @ K @ C @ A @ dm[2] + dK[2] @ np.array(yi).reshape(-1,1)


        grad += v * dv /s_hat[0][0] - 0.5 * v**2 * dS.reshape(-1,1) / s_hat[0][0]**2

        #print ('Likelihood: {},\nGradient:\n{}\n'.format(L, grad))

    return L, grad


#===================================================================================================================================================
## MCMC for Hyper-parameter estimation  (Equation 11.25 in Sarkka & Solin)
#===================================================================================================================================================


def metropolis_hasting(target_density, proposal_density, proposal_sampler, num_samples, initial_state, x, L, Q, burn_in, dt):
    samples = []
    num_accepted = 0

    current_state = initial_state

    for i in range(num_samples + burn_in):
        proposed_state = proposal_sampler(current_state)
        hasting_ratio = proposal_density(current_state, proposed_state) / proposal_density(proposed_state, current_state)
        metropolis_ratio = target_density(proposed_state, x, L, Q, dt) / target_density(current_state, x, L, Q, dt)
        ratio = metropolis_ratio * hasting_ratio

        acceptance_prob = min(ratio, 1)

        if np.random.rand() <= acceptance_prob:
            current_state = proposed_state
            num_accepted += 1

        if i >= burn_in:
            samples.append(current_state)

    acceptance_rate = num_accepted / (num_samples + burn_in)
    #print(f'Acceptance rate: {acceptance_rate}')
    return samples


def lognormal(proposed_state, current_state, sigma=0.2):
    # Proposal Density
    log_ratio_ell = np.log(proposed_state[0]) - np.log(current_state[0])            # corresponds to (ln(x) - \mu)
    log_ratio_sigma_sq = np.log(proposed_state[1]) - np.log(current_state[1])
    
    prob_ell = (1 / (proposed_state[0] * np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-log_ratio_ell ** 2 / (2 * sigma ** 2))
    prob_sigma_sq = (1 / (proposed_state[1] * np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-log_ratio_sigma_sq ** 2 / (2 * sigma ** 2))
    
    return prob_ell * prob_sigma_sq   # assumes independence between two hyper-parameters

def proposal_sampler(current_state, sigma=0.1):
    # Samples from lognormal distribution
    proposed_ell = np.random.lognormal(mean=np.log(current_state[0]), sigma=sigma)
    proposed_sigma_sq = np.random.lognormal(mean=np.log(current_state[1]), sigma=sigma)
    
    return np.array([proposed_ell, proposed_sigma_sq])

# Function to update F, Q and P based on the hyper-parameters
def update_F_Q_P(ell, sigma_sq):
    F = np.array([[0, 1], [-3/(ell**2), -2*np.sqrt(3)/ell]])
    Q = 12 * np.sqrt(3) * sigma_sq / (ell**3)
    P = sigma_sq * np.diag([1, 3/ell**2])
    return F, Q, P


# Evaluate negative log-likelihood
def nll(params, x, L, Q, dt):
    #Need the states not the observations.
    ell, sigma_sq = params
    F, Q, P = update_F_Q_P(ell, sigma_sq)

    nll = 0
    for k in range(len(x)-1):
        A_k = expm(F * dt)
        delta_xk = x[k+1] - A_k @ x[k]

        def integrand(tau, i, j):
            exp1 = expm(F * (dt - tau))
            exp2 = expm(F.T * (dt - tau))
            variance = L.reshape(-1,1) @ Q.reshape(-1,1) @ (L.reshape(-1,1)).T
            return (exp1 @ variance @ exp2)[i, j]

        Sigma_k = np.zeros_like(F)
        for n in range(F.shape[0]):
            for m in range(F.shape[1]):
                Sigma_k[n,m], _ = quad(integrand, 0, dt, args = (n, m))
        
        #Cholesky
        Lo = np.linalg.cholesky(Sigma_k) + 1e-12
        a = np.linalg.solve(Lo.T, np.linalg.solve(Lo,delta_xk))      # a = Sigma_k_inv @ delta_xk
    
        nll += 0.5 * np.log(det(2 * np.pi * Sigma_k))
        nll += 0.5 * delta_xk.T @ a

    return nll


#=====================================================================================================================================================
## Optimize via minimizing negative-log marginal likelihood from GP formulation
#=====================================================================================================================================================

def matern32_kernel(X, Y, lengthscale=1.5, variance=1.0):
    # lengthscale = 4.0 , variance = 2.5 ---> replicate figure.
    distance = np.abs(X - Y.T)
    # Calculate the kernel matrix
    sqrt3 = np.sqrt(3.0)
    #lengthscale = np.abs(lengthscale)
    #variance = np.abs(variance)
    K = variance * (1.0 + sqrt3 * distance / lengthscale) * np.exp(-sqrt3 * distance / lengthscale)
    return K

def dKdn2(Ky):
    # Differentiate w.r.t. noise variance
    return np.eye(Ky.shape[0])

def dKds2(d, l):
    # Differentiate w.r.t. scale variance
    return (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)

def dKdl(sigma_2, d, l):
    # Differentiate w.r.t. length-scale
    return (3 * sigma_2 * d**2/l**3) * np.exp(-np.sqrt(3)*d/l)

def grad_i(Ky, y, dKdx):
    # Get gradient of log-marginal likelihood w.r.t. hyper-params
    '''dKdx: one of these (dKdn2, dKds2, dKdl)'''
    α = np.linalg.lstsq(Ky, y, rcond = None)[0]   # a = Ky^-1 @ y
    comp1 = α @ α.T @ dKdx
    comp2 = np.linalg.lstsq(Ky, dKdx, rcond = None)[0]            # comp2 = Ky^-1 @ dKdx 
    return -0.5 * np.trace(comp1 - comp2)                           # NOTE: inserted -ve beacuse we'd like to minimize the function

def neg_log_like(params, x, y, grad = False):
    # Evalaute negative log marginal likelihood for optimization
    sigma2_noise, sigma2, l = params
    Ky = matern32_kernel(x.reshape(-1,1), x.reshape(-1,1), l, sigma2) + sigma2_noise * np.eye(x.shape[0])
    α = np.linalg.lstsq(Ky, y, rcond = None)[0]
    neg_log_likelihood = 0.5 * y.T @ α + 0.5 * np.log(np.linalg.det(Ky)) +  0.5 * x.shape[0] * np.log(2 * np.pi)

    if grad:
        gradient = gradients(params, x, y)
        return neg_log_likelihood[0][0], gradient
    else:
        return neg_log_likelihood[0][0] #, np.zeros_like(params)

def gradients(params, x, y):
    'Get complete gradient'
    sigma2_noise, sigma2, l = params
    #x = x.reshape(-1,1)
    Ky = matern32_kernel(x.reshape(-1,1), x.reshape(-1,1), l, sigma2) + sigma2_noise * np.eye(x.shape[0])
    d = np.abs(x - x.T)
    dK_noise = dKdn2(Ky)
    dK_sigma2 = dKds2(d, l)
    dK_l = dKdl(sigma2, d, l)

    # Gradients
    grad_noise = grad_i(Ky, y, dK_noise)
    grad_sigma2 = grad_i(Ky, y, dK_sigma2)
    grad_l = grad_i(Ky, y, dK_l)

    return np.array([grad_noise, grad_sigma2, grad_l])


#===================================================================================================================================================
## MCMC for Hyper-parameter estimation -- Using a standard approach
#===================================================================================================================================================

def metropolis_hastingII(target_density, proposal_density, proposal_sampler, num_samples, initial_state, x, y, burn_in):
    samples = []
    num_accepted = 0

    current_state = initial_state

    for i in range(num_samples + burn_in):
        proposed_state = proposal_sampler(current_state)
        print ('proposed state: ', proposed_state)
        print ('current state: ', current_state)
        log_hasting_ratio = proposal_density(current_state, proposed_state) - proposal_density(proposed_state, current_state)
        log_metropolis_ratio = target_density(proposed_state, x, y) - target_density(current_state, x, y)   # returns 
        log_ratio = log_metropolis_ratio + log_hasting_ratio
        
        #print ('log hasting ratio: ', log_hasting_ratio)
        #print ('log metropolis ratio ', log_metropolis_ratio)
        print ('')

        acceptance_prob = min(np.exp(log_ratio), 1)
        #acceptance_prob = log_ratio

        if np.random.rand() <= acceptance_prob:
            current_state = proposed_state
            num_accepted += 1

        if i >= burn_in:
            samples.append(current_state)

    acceptance_rate = num_accepted / (num_samples + burn_in)
    print(f'Acceptance rate: {acceptance_rate}')
    return samples


def lognormalII(proposed_state, current_state):
    # log of Proposal Density
    log_density_sigma_sq_noise = scipy.stats.lognorm(s=0.1, scale=np.exp(np.log(current_state[0]))).logpdf(proposed_state[0])
    log_density_ell = scipy.stats.lognorm(s=0.1, scale=np.exp(np.log(current_state[1]))).logpdf(proposed_state[1])
    log_density_sigma_sq = scipy.stats.lognorm(s=0.1, scale=np.exp(np.log(current_state[2]))).logpdf(proposed_state[2])

    # The total log density is the sum of the log densities for each parameter
    total_log_density = log_density_sigma_sq_noise + log_density_ell + log_density_sigma_sq
    return total_log_density


def proposal_samplerII(current_state):
    # Samples from lognormal distribution
    print ('Proposal Sampler got current state: ', current_state)
    if np.any(current_state <= 0):
        raise ValueError("Current state must have strictly positive elements")
    
    proposed_sigma_sq_noise =  np.random.lognormal(mean=current_state[0], sigma=1)
    proposed_ell = np.random.lognormal(mean=current_state[1], sigma=1)
    proposed_sigma_sq = np.random.lognormal(mean=current_state[2], sigma=1)

    proposed_state = np.array([proposed_sigma_sq_noise, proposed_sigma_sq, proposed_ell])

    return proposed_state


def unnormalized_posterior(states, x, y):
    noise_var, sigma_sq, ell = states
    Ky = matern32_kernel(x, x, ell, sigma_sq) + noise_var * np.eye(y.shape[0])
    try:
        α = np.linalg.lstsq(Ky, y, rcond = None)[0]
        likelihood = -0.5 * y.T @ α - 0.5 * np.log(np.linalg.det(Ky)) -  0.5 * x.shape[0] * np.log(2 * np.pi)
        #likelihood = scipy.stats.multivariate_normal.logpdf(y, mean = np.zeros((len(y))), cov = Ky)
    except np.linalg.LinAlgError:
        print ('got here!')
        return -np.inf
    
    # sample from prior distribution
    prior_noise_var = scipy.stats.lognorm.logpdf(noise_var, s = 0.5, loc = 0, scale = np.exp(0))
    prior_sigma_sq = scipy.stats.lognorm.logpdf(sigma_sq, s = 0.1, loc = 0, scale = np.exp(0))
    prior_ell = scipy.stats.lognorm.logpdf(ell, s = 0.1, loc = 0, scale = np.exp(0))

    prior = prior_noise_var + prior_sigma_sq + prior_ell

    return likelihood[0][0] + prior

#===================================================================================================================================================
## MAP for Hyper-parameter estimation -- Using a standard approach
#===================================================================================================================================================
def MAP(params, x, y, grad = False):
    # Evalaute negative log marginal likelihood for optimization
    sigma2_noise, sigma2, l = params
    Ky = matern32_kernel(x.reshape(-1,1), x.reshape(-1,1), l, sigma2) + sigma2_noise * np.eye(x.shape[0])
    α = np.linalg.lstsq(Ky, y, rcond = None)[0]
    neg_log_likelihood = 0.5 * y.T @ α + 0.5 * np.log(np.linalg.det(Ky)) +  0.5 * x.shape[0] * np.log(2 * np.pi)

    prior_means = np.array([0.1, 1.0, 1.0])
    prior_sigmas = np.array([0.01, 2, 2])

    # Gaussian Prior
    neg_log_prior = 0.5 * np.sum(((params - prior_means) / prior_sigmas) ** 2) 

    # Log-normal prior
    #neg_log_prior = np.sum(np.log(params * prior_sigmas * np.sqrt(2 * np.pi)) + ((np.log(params) - np.log(prior_means)) ** 2) / (2 * prior_sigmas ** 2))

    if grad:
        gradient = gradients(params, x, y)
        return neg_log_likelihood[0][0] + neg_log_prior, gradient
    else:
        return neg_log_likelihood[0][0] + neg_log_prior#, np.zeros_like(params)
    
#===================================================================================================================================================
## GP Regression -- Using a standard approach
#===================================================================================================================================================
def gp_regression(X1, X2, y1, kernel, noise = None):

    K11 = kernel(X1, X1)
    if noise is None:
        K11 += np.eye(K11.shape[0]) * 0.1**2
    K12 = kernel(X1, X2)
    K22 = kernel(X2, X2)


    L = scipy.linalg.cholesky(K11, lower = True)
    E = scipy.linalg.solve_triangular(L, K12, lower = True)
    v = scipy.linalg.solve_triangular(L, y1, lower = True)

    μ = E.T @ v
    Σ = K22 - E.T @ E

    return μ, Σ