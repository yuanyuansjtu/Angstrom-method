import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import invgamma
from joblib import Parallel, delayed
import time
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

class Metropolis_Hasting_sampler:
    
    def __init__(self,params_init,prior_log_mu,prior_log_sigma,df_phase_diff_amp_ratio,exp_setting,N_sample,transition_sigma,result_name):
        
        self.params_init = params_init
        self.prior_log_mu = prior_log_mu
        self.prior_log_sigma = prior_log_sigma
        self.exp_setting = exp_setting
        self.df_phase_diff_amp_ratio = df_phase_diff_amp_ratio
        self.N_sample = N_sample
        self.transition_sigma = transition_sigma
        self.result_name = result_name
        #self.accepted = []

        #updated!
        self.alpha_posterior = 0
        self.h_posterior = 0
        
    def AL(self,x1,x2):
        
        return np.sqrt(x1**4/(np.sqrt(x1**4+x2**4)+x2**2))
    
    def BL(self,x1,x2):
        
        return np.sqrt(x2**2+np.sqrt(x1**4+x2**4))
    
    def lopze_wave_vector(self,x1,x2):
        
        return (self.AL(x1,x2)+1j*self.BL(x1,x2))

    def lopze_original(self,x1,x2,x3):
        
        amp = np.abs(np.cos(self.lopze_wave_vector(x1,x2)*(1-x3))/np.cos(self.lopze_wave_vector(x1,x2)))
        phase = np.angle(np.cos(self.lopze_wave_vector(x1,x2)*(1-x3))/np.cos(self.lopze_wave_vector(x1,x2)))
        
        return amp,phase
    
    def log_likelihood(self,params):
    
        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        px = self.exp_setting['px']
        r = self.exp_setting['r'] # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']
        gap = self.exp_setting['gap']

        alpha, h = 10**params[0],10**params[1]

        sigma_dA =10**params[2]
        sigma_dP = 10**params[3]
        z = params[4]
        rho_dA_dP = np.tanh(z)

        k = alpha*cp*rho

        x = self.df_phase_diff_amp_ratio['x']
        x1 = np.sqrt(f_heating*2*np.pi/(2*alpha))*L
        x2 = np.sqrt(h/r/k)*L
        x3 = x/L # x3 indicate different locations on the sample

        amp_ratio_measurement = self.df_phase_diff_amp_ratio['amp_ratio']
        phase_diff_measurement = self.df_phase_diff_amp_ratio['phase_diff']

        p_joint_dA_dP = np.zeros(len(x3))


        for i,x3_ in enumerate(x3):
            amp_ratio_theoretical_,phase_diff_theoretical_ = self.lopze_original(x1,x2,x3_)
            x_ = [amp_ratio_measurement[i],phase_diff_measurement[i]]
            mean=[amp_ratio_theoretical_,phase_diff_theoretical_]
            cov = [[sigma_dA**2,sigma_dA*sigma_dP*rho_dA_dP],[sigma_dA*sigma_dP*rho_dA_dP,sigma_dP**2]]

            p_joint_dA_dP[i] =  np.log(multivariate_normal.pdf(x_,mean,cov))

        return np.sum(p_joint_dA_dP)
    
    def acceptance(self,x, x_new):
        if x_new>x:
            return True
        else:
            accept=np.random.uniform(0,1)
            return (accept < (np.exp(x_new-x)))
    
    def manual_priors(self,params): # this is actually proposal distribution's probability
        
        p_log_alpha = norm.pdf(params[0], loc = self.prior_log_mu[0], scale = self.prior_log_sigma[0])
        p_log_h = norm.pdf(params[1], loc = self.prior_log_mu[1], scale = self.prior_log_sigma[1])

        p_log_sigma_dA = norm.pdf(params[2], loc = self.prior_log_mu[2], scale = self.prior_log_sigma[2])
        p_log_sigma_dP = norm.pdf(params[3], loc = self.prior_log_mu[3], scale = self.prior_log_sigma[3])

        return np.log(p_log_alpha)+np.log(p_log_h)+np.log(p_log_sigma_dA)+np.log(p_log_sigma_dP)
    
    def rw_proposal(self,params):
        [alpha,h] = np.random.multivariate_normal(params[0:2],[[self.transition_sigma[0]**2,-0.4*self.transition_sigma[0]*self.transition_sigma[1]],[-0.4*self.transition_sigma[0]*self.transition_sigma[1],self.transition_sigma[1]**2]])
        [sigma_dA,sigma_dP] = np.random.multivariate_normal(params[2:4],[[self.transition_sigma[2]**2,-0.6*self.transition_sigma[2]*self.transition_sigma[3]],[-0.6*self.transition_sigma[2]*self.transition_sigma[3],self.transition_sigma[3]**2]])
        rho = np.random.normal(params[4], scale=self.transition_sigma[4])
        #[sigma_dA,sigma_dP,rho] = np.random.normal(params[2:5],scale = self.transition_sigma[2:5])
        return [alpha,h,sigma_dA,sigma_dP,rho]
    
    def metropolis_hastings_rw(self):

        params = self.params_init
        accepted = []
        #rejected = []   
        n_sample = 0
        n_rej = 1

        transition_model_rw = lambda x: np.random.normal(x,scale = self.transition_sigma)
         
        while(n_sample<self.N_sample):

            #params_new =  transition_model_rw(params)
            params_new = self.rw_proposal(params)
            params_lik = self.log_likelihood(params)
            params_new_lik = self.log_likelihood(params_new) 
            #params_new[0:2] = transition_rw_alpha_h(params[0:2])
            jac = np.log(10**(np.sum(params[0:4])))+np.log((2+2*np.exp(4*params[4]))/(1+np.exp(2*params[4]))**2)
            jac_new = np.log(10**(np.sum(params_new[0:4])))+np.log((2+2*np.exp(4*params_new[4]))/(1+np.exp(2*params_new[4]))**2)
                        
            if (self.acceptance(params_lik + self.manual_priors(params)+jac,params_new_lik+self.manual_priors(params_new)+jac_new)): 
  
                params = params_new
                accepted.append(params_new)
                n_sample += 1
                print('iter '+str(n_sample)+', accepted: '+str(params_new)+', acceptance rate: '+"{0:.4g}".format(n_sample/n_rej))
            else:
                n_rej += 1           

        accepted = np.array(accepted)
        
        
        return accepted
    
    
    def least_square_regression(self,params):
        
        # calculate square error of measurement vs theoretical
        
        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        px = self.exp_setting['px']
        r = self.exp_setting['r'] # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']
        gap = self.exp_setting['gap']
        
        
        alpha, h = 10**params[0],10**params[1]
        
        w = 2*np.pi*f_heating
        k = alpha*cp*rho
        x = self.df_phase_diff_amp_ratio['x']
        x1 = np.sqrt(w/(2*alpha))*L
        x2 = np.sqrt(h/r/k)*L
        x3 = x/L

        amp_ratio_measurement = self.df_phase_diff_amp_ratio['amp_ratio']
        phase_diff_measurement = self.df_phase_diff_amp_ratio['phase_diff']

        amp_ratio_theoretical = []
        phase_diff_theoretical = []
        p_amp_ratio_list = np.zeros(len(x3))
        p_phase_diff_list = np.zeros(len(x3))
        err = np.zeros(len(x3))

        for i,x3_ in enumerate(x3):
            amp_ratio_theoretical_,phase_diff_theoretical_ = self.lopze_original(x1,x2,x3_)
            err_ = (amp_ratio_theoretical_-amp_ratio_measurement[i])**2+(phase_diff_theoretical_-phase_diff_measurement[i])**2
            #p_phase_diff_list[i] = np.log(p_phase_diff)
            err[i] = err_
        return np.sum(err)
    
    
    def minimize_regression(self):
        guess = self.params_init[:2]
        results = minimize(self.least_square_regression,guess, method = 'Nelder-Mead', options={'disp': True})
        return results
    


def multi_chain_Metropolis_Hasting(params_init,prior_log_mu,prior_log_sigma,df_phase_diff_amp_ratio,exp_setting,N_sample,transition_sigma,result_name,N_chains):
    # execute Metropolis-Hasting algorithm using parallel processing
    chains = [Metropolis_Hasting_sampler(params_init,prior_log_mu,prior_log_sigma,df_phase_diff_amp_ratio,exp_setting,N_sample,transition_sigma,result_name) for i in range(N_chains)]
    results = Parallel(n_jobs=N_chains)(delayed(chain.metropolis_hastings_rw)() for chain in chains)

    all_chain_results = np.reshape(results,(-1,5))
    chain_num = [int(i/N_sample)+1 for i in range(len(all_chain_results))]

    df_posterior = pd.DataFrame(data = all_chain_results)
    df_posterior.columns = ['alpha','h','sigma_dA','sigma_dP','corr']
    df_posterior['chain_num'] = chain_num
    return df_posterior