import numpy as np
from scipy.optimize import minimize
import scipy
import pyro, torch
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import (MCMC, NUTS, HMC)


class Vanilla_GP:
    '''This class takes in the parameters without log transformation at first'''

    def __init__(self, params):
        # Initialize model parameters
        self.params = params

    def matern32_kernel(self, X, Y, lengthscale=1.0, variance=1.0):
        distance = np.abs(X - Y.T)
        sqrt3 = np.sqrt(3.0)
        K = variance * (1.0 + sqrt3 * distance / lengthscale) * np.exp(-sqrt3 * distance / lengthscale)
        return K

    def dk1(self, r, p):
        return (1 + np.sqrt(3) * np.abs(r)/p[1]) * np.exp(-np.sqrt(3)*np.abs(r)/p[1])

    def dk2(self, r, p):
        return p[0]*(3 * r**2/p[1]**3) * np.exp(-np.sqrt(3)*np.abs(r)/p[1])

    def gp_solve(self, x, y,  xt = None, opt = True, get_likelihood = False):
        '''
        Args:
        params  - Log-Parameter (sigma2, magnSigma2, lengthscale)
        x       - Training inputs
        y       - Training outputs
        opt     - True if optimizing params; False if inferring
        
        Returns:
        (if opt = True)
        e       - Negative log-marginal likelihood
        eg      - its gradient (optional)

        (if opt = False)
        Eft     - Predicted mean
        Varft   - Predicted marginal variance
        Covft   - Predicted joint covariance matrix
        lb      - 95% confidence lower bound
        ub      - 95% confidence upper bound
        '''

        if get_likelihood:
            param = self.params
        else:
            param = np.exp(self.params)

        sigma2 = param[0]
        magnSigma2 = param[1]
        lengthscale = param[2]

        K11 = self.matern32_kernel(x.reshape(-1,1), x.reshape(-1,1), lengthscale, magnSigma2) + np.eye(x.shape[0]) * sigma2

        try:
            L = np.linalg.cholesky(K11)
        except np.linalg.LinAlgError:
            jitter = 1e-9 * np.eye(x.shape[0])
            L = np.linalg.cholesky(K11 + jitter)

        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        vv = np.linalg.solve(L, y)
        alpha = np.linalg.solve(L.T, vv)

        if xt == None:
            xt = x

        if not opt:
            K21 = self.matern32_kernel(xt, x, lengthscale, magnSigma2)
            K22 = self.matern32_kernel(xt, xt, lengthscale, magnSigma2)
            
            # Solve for Mean
            Eft = K21 @ alpha
            
            # Solve for variance
            v = np.linalg.solve(L, K21.T)
            #vk = np.linalg.solve(L.T, v)
            #cov = K22 - K21 @ vk
            cov = K22 - v.T @ v

            # Marginal
            var = np.diag(cov)
            lb = Eft.ravel() -  1.96 * np.sqrt(var)
            ub = Eft.ravel() + 1.96 * np.sqrt(var)

            return Eft, var, cov, lb, ub
        
        else:
            # Evaluate negative log marginal likelihood
            beta1 = vv.T @ vv
            e = 0.5 * beta1 + np.sum(np.log(np.diag(L))) + 0.5*x.shape[0]*np.log(2*np.pi)

            # calculate gradient of log marginal likelihood
            eg = np.zeros(len(param))

            # Derivate w.r.t sigma2 i.e. noise variance
            invk = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(x.shape[0])))
            eg[0] = 0.5 * np.trace(invk) - 0.5 * (alpha.T @ alpha)[0][0]

            # Derivative w.r.t. rest of the parameters
            for i in range(1, len(param)-1):
                r = x[:, None] - x[None,:]
                dK = self.dk1(r, param[1:]) if i == 1 else self.dk2(r, param[1:])
                dK = dK.reshape(dK.shape[0],dK.shape[1])
                eg[i] = 0.5 * np.sum(invk * dK) - 0.5 * (alpha.T @ (dK @ alpha))[0][0]
                
            # during optimization if jac = True, you need to return the gradient as well; 
            # if jac = False, gradient will be approximated numerically
            return e.item() #, eg
        
    
    def optimize(self, x, y, xt = None):
        # Define objective function for optimization
        def objective_function(params):
            self.params = params
            return self.gp_solve(x, y, xt = None, opt = True)
        
        result = minimize(fun = objective_function, 
                          x0 = np.log(self.params), 
                          method = 'Powell',          # Nelder-Mead, Powell, BFGS, CG, Newton-CG
                          jac = False, 
                          options={'disp': True})
        
        self.params = result.x    # log (params)

    def get_params(self):
        # Reurns optimized params
        return np.exp(self.params)
    

class IHGP:
    '''This class also takes in params without log transformation'''
    def __init__(self, params):
        # Initialize model parameters
        self.params = params

    def cf_matern_to_ss(self, magnSigma2, lengthscale, opt = False):
        # Form State Space Model
        Lambda = np.sqrt(3)/lengthscale
        F = np.array([[0, 1],[-Lambda**2, -2*Lambda]])
        L = np.array([0,1]).reshape(2,1)
        Qc = 12 * (np.sqrt(3)/lengthscale**3) * magnSigma2
        H = np.array([1, 0]).reshape(1,2)
        Pinf = magnSigma2 * np.diag([1, 3/lengthscale**2])


        if opt == True:
            # Calculate Derivatives
            dFmagnSigma2 = np.zeros(shape = (F.shape[0], F.shape[1]))
            dFlengthscale = np.array([[0, 0],[6/lengthscale**3, 2*np.sqrt(3)/lengthscale**2]])
            
            dQcmagnSigma2 = 12 * np.sqrt(3)/lengthscale**3
            dQclengthscale = -3 * 12 * np.sqrt(3)/lengthscale**4 * magnSigma2

            dPinfmagnSigma2 = np.array([[1, 0], [0, 3 / lengthscale**2]])
            dPinflengthscale = magnSigma2 * np.array([[0, 0], [0, -6 / lengthscale**3]])

            # stack all derivates
            dF = np.zeros((2, 2, 2))
            dQc = np.zeros((1, 1, 2))
            dPinf = np.zeros((2, 2, 2))

            dF[:,:,0] = dFmagnSigma2
            dF[:,:,1] = dFlengthscale

            dQc[0,0,0] = dQcmagnSigma2
            dQc[0,0,1] = dQclengthscale

            dPinf[:,:,0] = dPinfmagnSigma2
            dPinf[:,:,1] = dPinflengthscale


            return F, L, Qc, H, Pinf, dF, dQc, dPinf
        
        else:
            return F, L, Qc, H, Pinf
        

    def ihgpr(self, x, y, xt = None, opt = False, get_likelihood = False):
        '''
        Adapted from IHGP Paper: A. Solin et.al. (2018)

        Args:
        params  - Log-Parameters [sigma^2, magSigma^2, lengthscale]
        x       - Training Inputs
        y       - Training Outputs
        xt      - Testing Points (default = None)
        opt     - To Optimize Hyperparameters or Not (Boolean)

        Returns:
        (If opt is False)
        Eft     - Predicted Mean
        var     - Predicted Variance (Marginal)
        lb      - Lower Bound for Eft (95 %)
        ub      - Upper Bound for Eft (95 %)

        (If opt is True)
        edata   - Negative Log-Marginal Likelihood
        gdata   - Its Gradient (Optional)
        '''

        # Combine observations and test points
        
        if xt == None:
            xt = x

        xall = np.concatenate([x.flatten(), xt.flatten()])
        yall = np.concatenate([y.flatten(), np.nan * np.ones(len(xt))])

        # Make sure points are unique and are in ascending order
        xall, sort_ind = np.unique(xall, return_index = True)
        yall = yall[sort_ind]

        # Only return test indices
        return_ind = np.arange(len(xall)) + 1
        return_ind = return_ind[-len(xt):]

        if np.std(np.diff(xall)) > 1e-12:
            raise Exception('This function only accepts equidistant time-stamps only')
        
        if get_likelihood:
            param = self.params
        else:
            param = np.exp(self.params)
        
        d = len(x)
        sigma2 = param[0]

        # Form State-Space Model
        if opt:
            F, L, Qc, H, Pinf, dF_, dQc_, dPinf_ = self.cf_matern_to_ss(param[1], param[2], opt)

            # Concatenate Derivatives
            dF = np.zeros((F.shape[0], F.shape[1], len(param)))
            dPinf = np.zeros((F.shape[0], F.shape[1], len(param)))
            dQc = np.zeros((1,1,len(param)))

            dF[:,:,1:] = dF_
            dQc[:,:,1:] = dQc_
            dPinf[:,:,1:] = dPinf_
            dR = np.zeros((1,1,len(param)))
            dR[:,:,0] = 1
        else:
            F, L, Qc, H, Pinf = self.cf_matern_to_ss(param[1], param[2], opt)

    
        # Stationary Stuffs
        dt = xall[1] - xall[0]
        A = scipy.linalg.expm(F * dt)
        Q = Pinf - A @ Pinf @ np.transpose(A)
        Q = (Q + Q.T)/2
        R = sigma2

        try:
            Pp = scipy.linalg.solve_discrete_are(A.T, H.T, Q, R)
        except Exception as e:
            print (f'Unstable DARE Solution: {e}')


        # Innovation variance
        S = H @ Pp @ H.T + R

        # Stationary Gain
        K = Pp @ H.T / S.item()

        # Pre-calculate
        AKHA = A - K @ H @ A


        if not opt:
            # -----Filtering and Smoothing -----

            # set initial state
            m = np.zeros((F.shape[0], 1))
            PF = Pp - K @ H @ Pp

            # Allocate space for results
            MS = np.zeros((F.shape[0], len(yall)))
            PS = np.zeros((F.shape[0], F.shape[1], len(yall)))

            # Filter Forward Recursion
            for k in range(len(yall)):
                if ~np.isnan(yall[k]):
                    m = AKHA @ m + K * yall[k]             # O(m^2) Complexity

                    # Store Estimate
                    MS[:, k] = m.T                   
                    PS[:,:,k] = PF                         # Same for all points
                else:
                    m = A @ m
                    MS[:, k] = m.T
                    PS[:,:,k] = Pinf
            
            # Backward Smoother
            GS = np.zeros([2, 2, len(yall)])               # Allocate space for Smoother Gain
            Lo = np.linalg.cholesky(Pp)

            #G = PF @ A.T @ np.linalg.inv(Lo.T) @ np.linalg.inv(Lo)
            G = (np.linalg.solve((Lo @ Lo.T), A @ PF.T)).T


            # Solve Riccati Equation
            QQ = PF - G @ Pp @ G.T
            QQ = (QQ + QQ.T)/2
            P = scipy.linalg.solve_discrete_are(G.T, np.zeros_like(G), QQ, Q)      # Q -- >  np.eye(F.shape[0])
            PS[:,:,-1] = P


            # RTS Smoothing
            m = m.T.reshape(2)
            for k in range(MS.shape[1] - 2, -1, -1):  # MS.shape[1] gives the number of columns in MS
                # Backward iteration
                m = MS[:, k] + G @ (m - A @ MS[:, k])
                # Store estimate
                MS[:, k] = m
                PS[:, :, k] = P
                GS[:, :, k] = G
            
            # Return smoother mean and marginal variance
            Eft = (H @ MS).T
            Varft = []
            for i in range(len(MS[1])):
                Varft.append( H @ PS[:, :, i] @ H.T)

            lb = Eft.reshape(-1) - 1.96 * np.sqrt(np.array(Varft).reshape(-1))
            ub = Eft.reshape(-1) + 1.96 * np.sqrt(np.array(Varft).reshape(-1))

            return Eft.reshape(-1), np.array(Varft).reshape(-1), lb, ub
        
        else:
            # -----Optimize Hyperparameters-----
            d = np.size(F, 1)
            nparam = len(param)

            # Allocate space for derivatives
            dA = np.zeros((d, d, nparam))
            dPP = np.zeros((d, d, nparam))
            dAKHA = np.zeros((d, d, nparam))
            dK = np.zeros((d, 1, nparam))
            dS = np.zeros((1, 1, nparam))
            HdA = np.zeros((d, nparam))

            
            # Pre-calculate Z and B
            Z = np.zeros((d,d))
            B = A @ K


            # Evaluate all derivatives
            for j in range(len(param)):
                # First matrix for the matrix factor decomposition
                FF_top = np.hstack([F, Z])
                FF_bottom = np.hstack([dF[:,:,j], F])
                FF = np.vstack([FF_top, FF_bottom])  
    
                
                # Solve matrix exponential
                AA = scipy.linalg.expm(FF * dt)
                dA[:,:,j] = AA[d:, :d]
                dQ = dPinf[:,:,j] - dA[:,:,j] @ Pinf @ A.T - A @ dPinf[:,:,j] @ A.T - A @ Pinf @ dA[:, :, j].T
                dQ = (dQ + dQ.T)/2

                # Precalculate C
                C = dA[:,:,j] @ Pp @ A.T + A @ Pp @ dA[:,:,j].T - dA[:,:,j] @ Pp @ H.T @ B.T - B @ H @ Pp @ dA[:, :, j].T + B @ dR[:,:,j] @ B.T + dQ
                C = (C + C.T)/2

                try:
                    dPP[:,:,j] = scipy.linalg.solve_discrete_are((A - B @ H).T, np.zeros((d,d)), C, np.eye(d))
                except Exception as e:
                    print (f'Unstable DARE Solution while evaluating derivatives: {e}')
                
                # Evaluate dS and dK
                dS[:,:,j] = (H @ dPP[:,:,j] @ H.T).item() + dR[:,:,j]
                dK[:,:,j] = dPP[:,:,j] @ H.T / S.item() - Pp @ H.T * (((H @ dPP[:,:,j] @ H.T).item() + dR[:,:,j])/S.item()**2)
                dAKHA[:,:,j] = dA[:,:,j] - dK[:,:,j] @ H @ A - K @ H @ dA[:,:,j]
                HdA[:,j] = (H @ dA[:,:,j]).reshape(-1).T     # (2,1) -- > (2,) 

            # Reshape for vectorization
            dAKHAp = dAKHA.transpose(2,0,1).reshape(-1,2)
            dKp = dK.reshape(2,-1)

        
            # Size of inputs
            steps = len(yall)
            m = np.zeros((d,1))
            dm = np.zeros((d, nparam))

            # Allocate space for results
            edata = 0.5 * np.log(2*np.pi)*steps + 0.5 * np.log(S.item())*len(x)
            gdata = 0.5 * steps * dS.ravel().reshape(-1, nparam) / S.item()


            # Loop over all observations
            for k in range(steps):

                if ~np.isnan(y[k]):

                    # Innovation Mean
                    v = y[k] - (H @ A @ m).item()

                    # Marginal Likelihood Approximation
                    edata += 0.5 * v**2/S.item()

                    # Same as above without the loop
                    dv  = -m.T @ HdA - H @ A @ dm
                    gdata += v * dv/S.item() - 0.5 * v**2 * dS.ravel().reshape(-1, nparam) / S.item()**2
                    dm = AKHA @ dm + dKp * y[k]   # (2,3)
                    dm = dm.reshape(-1,1)         # (6,1)
                    dm += dAKHAp @ m              # (6,1)
                    dm = dm.reshape(d, -1)        # (2,3)

                    # Stationary Filter Recursion
                    m = AKHA @ m + K * y[k]

                else:
                    for j in range(len(param)):
                        dm[:,j] = (A @ dm[:,j].reshape(d,1) + dA[:,:,j] @ m).reshape(-1)
                    m = A @ m
            

            # Account for log-scale
            gdata = gdata * np.exp(self.params)

        return edata #, gdata.reshape(3)
    

    def optimize(self, x, y, xt = None):
        # Define objective function for optimization
        def objective_function(params):
            self.params = params
            #ss = lambda x, p, f: self.cf_matern_to_ss(self.params[1], self.params[1], f)
            return self.ihgpr(x, y, xt = None, opt = True)
        
        result = minimize(fun = objective_function, 
                          x0 = np.log(self.params), 
                          method = 'Powell',          # Nelder-Mead, Powell, BFGS, CG, Newton-CG
                          jac = False, 
                          options={'disp': True})
        
        self.params = result.x    # log (params)


    def get_params(self):
        # Reurns optimized params
        return np.exp(self.params)
 

class MarkovChainMC:
    def __init__(self, x, y, num_samples, num_warmups):
        if isinstance(x, np.ndarray):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.x = torch.tensor(x, device = self.device).reshape(-1,1)
            self.y = torch.tensor(y, device = self.device).reshape(-1,1)
        
        self.num_samples = num_samples
        self.num_warmups = num_warmups

    def kernel(self, x1, x2 ,obs_noise, variance, lengthscale):
        distance = torch.abs(x1 - x2.T)
        sqrt3 = torch.sqrt(torch.tensor(3.0, device = self.device))
        K = variance * (1.0 + sqrt3 * distance / lengthscale) * torch.exp(-sqrt3 * distance / lengthscale)
        return K + torch.eye(x1.shape[0], device = self.device) * obs_noise
    
    def model(self, x, y):
        # set uninformative prior
        var = pyro.sample('Kernel_amplitude', dist.LogNormal(loc = 0.0, scale = 1.0)) # mean and std 0.2 1.8 ->log normal # UNif -7.0 - 2.0/3.0
        noise = pyro.sample('Kernel_noise', dist.LogNormal(loc = 0.0, scale = 1.0))
        length = pyro.sample('Kernel_length', dist.LogNormal(loc = 0.0, scale = 1.0))

        var_ = pyro.deterministic('kernel_amplitude_', torch.exp(var))
        noise_ = pyro.deterministic('kernel_noise_', torch.exp(noise))
        length_ = pyro.deterministic('kernel_length_', torch.exp(length))

        # compute kernel
        K = self.kernel(x, x, noise_, var_, length_)

        # sample according to the standard gaussian process i.e. get samples from GP conditioned on the observations
        with pyro.plate('plate'):
            pyro.sample('y',
                        dist.MultivariateNormal(loc = torch.zeros(x.shape[0], device = self.device), covariance_matrix = K),
                        obs = y)
        
    def run_inference(self):
        #model_kernel = HMC(self.model, step_size = 0.01, num_steps = 5)
        model_kernel = NUTS(self.model)
        mcmc = MCMC(model_kernel, num_samples = self.num_samples, warmup_steps = self.num_warmups, num_chains = 1)
        mcmc.run(self.x, self.y)
        mcmc.summary()
        return mcmc.get_samples()
    

class MaximumAposteriori:
    '''
    If getting MAP value, simply pass parameters and run map_objective() to get the corresponding MPA values. Useful when plotting MAP landscape
    If optimizing, pass parameters and run run_inference() to get optimized parameters. Returned as log-scale.
    '''
    def __init__(self, x, y, init_params, num_steps, learning_rate = 0.005):
        if isinstance(x, np.ndarray):
            self.x = torch.tensor(x)
            self.y = torch.tensor(y)
            self.params = torch.tensor(init_params) # noise_var, amp_var, length
            self.lr = learning_rate
        
        self.num_steps = num_steps


    def kernel(self, x1, x2 ,obs_noise, variance, lengthscale):
        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        distance = torch.abs(x1 - x2.T)
        sqrt3 = torch.sqrt(torch.tensor(3.0))
        K = variance * (1.0 + sqrt3 * distance / lengthscale) * torch.exp(-sqrt3 * distance / lengthscale)
        return K + torch.eye(x1.shape[0]) * obs_noise


    def map_objective(self):
        '''Priors are assumed to be sampled from LogNormal (0.0, 1.0)'''
        pyro.clear_param_store()
        Ky = self.kernel(self.x, self.x, self.params[0], self.params[1], self.params[2])
        L = torch.linalg.cholesky(Ky) + 1e-14

        # reshape 'y' without modifying global 'y'
        y = self.y.reshape(-1,1)
        α = torch.linalg.solve(L.T, torch.linalg.solve(L, y))

        # Likelihood Term
        neg_log_likelihood = 0.5 * y.T @ α + 0.5 * 2 * torch.sum(torch.log(torch.diag(L))) +  0.5 * self.x.shape[0] * torch.log(torch.tensor(2.0 * torch.pi))
        
        # Prior
        neg_log_prior = 0
        for theta_j in (self.params):
            neg_log_prior += 0.5 * (torch.log(theta_j)**2) + torch.log(theta_j) + 0.5 * torch.log(torch.tensor(2.0 * torch.pi))

        return neg_log_likelihood.item() + neg_log_prior.item()
    

    def run_inference(self):
        pyro.clear_param_store()
        kernel = gp.kernels.Matern32(input_dim = 1, variance = self.params[1], lengthscale = self.params[2])

        gpr = gp.models.GPRegression(X = self.x, y = self.y, kernel= kernel, noise = self.params[0])

        # priors have support on the positive reals
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(loc = 0.0, scale = 1.0))
        gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(loc = 0.0, scale = 1.0))

        optimizer = torch.optim.Adam(params = gpr.parameters(), lr = self.lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        for i in range(self.num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()

        gpr.set_mode('guide')

        print ('----Exponentiated Params----')
        print("noise = {}".format(gpr.noise))
        print("variance = {}".format(gpr.kernel.variance))
        print("lengthscale = {}".format(gpr.kernel.lengthscale))

        return torch.log(gpr.noise).detach().item(), torch.log(gpr.kernel.variance).detach().item(), torch.log(gpr.kernel.lengthscale).detach().item()