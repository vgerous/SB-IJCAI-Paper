import numpy as np
import copy
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal

class GMMDist:
    def __init__(self, n_components, weights, means, variances):
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.variances = variances
        self.exp = self._e()
        self.var = self._var()
        self.std = np.sqrt(np.absolute(self._var()))
    
    def _e(self):
        return np.dot(self.weights, self.means)
    
    def _var(self):
        return sum([weight * (mu**2 + sigma**2) for weight, mu, sigma in zip(self.weights, self.means, np.sqrt(np.absolute(self.variances)))]) - self._e() ** 2


class GMMCDFCalculator:
    def __init__(self):
        self.norm_cache = {}

    def cdf(self, dist: GMMDist, y):
        cdfs = np.array([self.norm_cdf(mu, sigma, y) for mu, sigma in zip(dist.means, np.sqrt(np.absolute(dist.variances)))])
        return np.dot(dist.weights, cdfs)
        
    def norm_cdf(self, mu, sigma, y):
        key = str(mu)+"_"+str(sigma)+"_"+str(y)
        if key not in self.norm_cache.keys():
            cdf_val = norm.cdf(y, loc=mu, scale=sigma)
            self.norm_cache[key] = cdf_val
        return self.norm_cache[key]


class MultiGMMCDFCalculator:
    def __init__(self):
        self.norm_cache = {}

    def cdf(self, dist: GMMDist, y):
        cdfs = np.array([self.norm_cdf(mu, sigma, y) for mu, sigma in zip(dist.means, np.sqrt(np.absolute(dist.variances)))])
        return np.dot(dist.weights, cdfs)
        
    def norm_cdf(self, mu, sigma, y):
        key = str(mu)+"_"+str(sigma)+"_"+str(y)
        if key not in self.norm_cache.keys():
            cdf_val = norm.cdf(y, loc=mu[0], scale=sigma[0][0])
            self.norm_cache[key] = cdf_val
        return self.norm_cache[key]

    def find_cov_matrix(self, var, y, index):
        new_cov = np.zeros([len(y)-1, len(y)-1])
        x_index = 0
        for i in range(len(y)):
            y_index = 0
            if i == index:
                continue
            for j in range(len(y)):
                if j == index:
                    continue
                new_cov[x_index][y_index] = var[i][j]
                y_index = y_index + 1
            x_index = x_index + 1
        return np.linalg.inv(new_cov)
    
    def find_cov_vector(self, var, y, index):
        new_cov = np.zeros(len(y)-1)
        x_index = 0
        for i in range(len(y)):
            if i == index:
                continue
            new_cov[x_index] = var[index][i]
            x_index = x_index + 1
        return new_cov

    def find_cond_diff(self, means, y, index):
        new_diff = np.zeros(len(y)-1)
        x_index = 0
        for i in range(len(y)):
            if i == index:
                continue
            new_diff[x_index] = y[i] - means[i]
            x_index = x_index + 1
        return new_diff

    def cond_cdf(self, dist: GMMDist, y, index):
        cdfs = np.array([self.cond_norm_cdf(mu, sigma, y, index) for mu, sigma in zip(dist.means, dist.variances)])
        return np.dot(dist.weights, cdfs)

    def cond_partial_exp(self, dist: GMMDist, y, index):
        exps = np.array([self.cond_partial_normal_exp(mu, sigma, y, index) for mu, sigma in zip(dist.means, dist.variances)])
        return np.dot(dist.weights, exps)

    def cond_partial_normal_exp(self, mu, sigma, y, index):
        A = self.find_cov_matrix(sigma, y, index)
        B = self.find_cov_vector(sigma, y, index)
        C = self.find_cond_diff(mu, y, index)
        new_mu = mu[index] + np.dot(np.dot(B, A), C)
        new_cov = sigma[index][index] - np.dot(np.dot(B, A), B)
        x_j = y[index]
        X = norm.pdf(-new_mu/np.sqrt(np.absolute(new_cov)), loc=0, scale=1) / norm.cdf(-new_mu/np.sqrt(np.absolute(new_cov)), loc=0, scale=1)
        Y = norm.pdf((x_j-new_mu)/np.sqrt(np.absolute(new_cov)), loc=0, scale=1) / norm.cdf((x_j-new_mu)/np.sqrt(np.absolute(new_cov)), loc=0, scale=1)
        return np.sqrt(np.absolute(new_cov)) * (X - Y)


    def cond_norm_cdf(self, mu, sigma, y, index):
        A = self.find_cov_matrix(sigma, y, index)
        B = self.find_cov_vector(sigma, y, index)
        C = self.find_cond_diff(mu, y, index)
        new_mu = mu[index] + np.dot(np.dot(B, A), C)
        new_cov = sigma[index][index] - np.dot(np.dot(B, A), B)
        cdf_val = norm.cdf(y[index], loc=new_mu, scale=np.sqrt(np.absolute(new_cov)))
        return cdf_val


class NormModel:
    def __init__(self, n_components=1, gmm_params={"covariance_type": "full", 
                                                 "max_iter": 500, 
                                                 "init_params": "random",
                                                 "random_state": 666}):
        self.gmm = GaussianMixture()
        gmm_params["n_components"] = n_components
        self.gmm.set_params(**gmm_params)

    def fit(self, X):
        self.X = np.array(X).reshape(-1, 1)
        self.gmm.fit(self.X)
    
    def predict(self):
        return np.dot(self.gmm.weights_, self.gmm.means_)
    
    def get_distribution(self):
        n_components = self.gmm.n_components
        means = self.gmm.means_.flatten()
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_.flatten()
        return GMMDist(n_components, weights, means, variances)

class MultiNormModel:
    def __init__(self, n_dimensions=1, n_components=1, gmm_params={"covariance_type": "full", 
                                                 "max_iter": 500, 
                                                 "init_params": "random",
                                                 "random_state": 666}):
        self.gmm = GaussianMixture()
        gmm_params["n_components"] = n_components
        self.gmm.set_params(**gmm_params)
        self.n_dimensions = n_dimensions

    def fit(self, X):
        self.n_dimensions = len(X)
        self.X = X
        self.gmm.fit(self.X)
        
    
    def predict(self):
        return np.dot(self.gmm.weights_, self.gmm.means_)
    
    def get_distribution(self):
        n_components = self.gmm.n_components
        means = self.gmm.means_
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_
        return GMMDist(n_components, weights, means, variances)

    def get_distribution_with_means(self, new_means):
        means = []
        for i in self.gmm.means_:
            means.append(i + new_means)
        n_components = self.gmm.n_components
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_
        return GMMDist(n_components, weights, means, variances)


class MultiGmmModel:
    def __init__(self, tol=1e-10, max_components=100, 
                 gmm_params={"covariance_type": "full", 
                             "max_iter": 500, 
                             "init_params": "random",
                             "random_state": 666}):
        self.tol = tol
        self.max_components = max_components
        self.gmm_params = gmm_params
        self.gmm = None
        self.n_components = 1
        self.last_decreased = 0
        
    def fit(self, X):
        self.X = X
        decreased = None
        for i in range(1, self.max_components + 1):
            current_gmm = GaussianMixture(n_components=i)
            current_gmm.set_params(**self.gmm_params)
            current_gmm.fit(self.X)
            if self.gmm is None:
                self.gmm = current_gmm
            else:
                decreased = self._get_likelihood_change(current_gmm)
                if decreased >= self.last_decreased:
                    self.last_decreased = decreased
                    self.gmm = current_gmm
                    self.n_components = i
                else:
                     return 

    def predict(self):
        return np.dot(self.gmm.weights_, self.gmm.means_)

    def get_distribution(self):
        n_components = self.gmm.n_components
        means = self.gmm.means_
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_
        return GMMDist(n_components, weights, means, variances)

    def get_distribution_with_means(self, new_means):
        means = []
        for i in range(len(self.gmm.means_)):
            means.append(self.gmm.means_[i] + new_means * self.gmm.weights_[i])
        n_components = self.gmm.n_components
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_
        return GMMDist(n_components, weights, means, variances)
            
    def _get_likelihood_change(self, current_gmm):
        return (-self.gmm.score(self.X) - ( -current_gmm.score(self.X)))
        

class GmmModel:
    def __init__(self, tol=1e-10, max_components=100, 
                 gmm_params={"covariance_type": "full", 
                             "max_iter": 500, 
                             "init_params": "random",
                             "random_state": 666}):
        self.tol = tol
        self.max_components = max_components
        self.gmm_params = gmm_params
        self.gmm = None
        self.n_components = 1
        self.last_decreased = 0

    def set_params(self, **params):
        if "tol" in params:
            self.tol = params["tol"]
        if "max_components" in params:
            self.max_components = params["max_components"]

    def fit(self, X):
        self.X = np.array(X).reshape(-1, 1)
        decreased = None
        for i in range(1, self.max_components + 1):
            current_gmm = GaussianMixture(n_components=i)
            current_gmm.set_params(**self.gmm_params)
            current_gmm.fit(self.X)
            if self.gmm is None:
                self.gmm = current_gmm
            else:
                decreased = self._get_likelihood_change(current_gmm)
                if decreased >= self.last_decreased:
                    self.last_decreased = decreased
                    self.gmm = current_gmm
                    self.n_components = i
                else:
                     return 

    def predict(self):
        return np.dot(self.gmm.weights_, self.gmm.means_)

    def get_distribution(self):
        n_components = self.gmm.n_components
        means = self.gmm.means_.flatten()
        weights  = self.gmm.weights_
        variances = self.gmm.covariances_.flatten()
        return GMMDist(n_components, weights, means, variances)
            
    def _get_likelihood_change(self, current_gmm):
        return (-self.gmm.score(self.X) - ( -current_gmm.score(self.X)))