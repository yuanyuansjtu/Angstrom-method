import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from scipy import signal
import seaborn as sns

class post_processing_results:
    
    def __init__(self,df_posterior,processing_settings,exp_setting,df_amplitude_phase):
        self.alpha_posterior = 0
        self.h_posterior = 0
        self.N_burns = processing_settings['N_burns']
        self.N_chains = processing_settings['N_chains']
        self.theoretical_amp_ratio = []
        self.theoretical_phase_diff = []
        self.df_posterior = df_posterior
        self.exp_setting = exp_setting
        self.df_amplitude_phase = df_amplitude_phase
        
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
    
    def theoretical_Lopze_solution(self):
        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        px = self.exp_setting['px']
        r = self.exp_setting['r'] # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']
        gap = self.exp_setting['gap']

        k = self.alpha_posterior*cp*rho

        x = self.df_amplitude_phase['x'].unique()
        x1 = np.sqrt(f_heating*2*np.pi/(2*self.alpha_posterior))*L
        x2 = np.sqrt(self.h_posterior/r/k)*L
        x3 = x/L # x3 indicate different locations on the sample
        
        amp_ratio_list = np.zeros(len(x3))
        phase_diff_list = np.zeros(len(x3))

        for i,x3_ in enumerate(x3):
            amp_ratio_theoretical_,phase_diff_theoretical_ = self.lopze_original(x1,x2,x3_)
            amp_ratio_list[i] = amp_ratio_theoretical_
            phase_diff_list[i] = phase_diff_theoretical_
            
        self.theoretical_amp_ratio = amp_ratio_list
        self.theoretical_phase_diff = phase_diff_list
        return phase_diff_list, amp_ratio_list
 
    
    def burn_posterior(self):
        
        alpha_posterior = []
        h_posterior = []
        sigma_dA_posterior = []
        sigma_dP_posterior = []
        corr_posterior = []
        chain_num = []

        for i in range(self.N_chains):
            df = self.df_posterior.query('chain_num =='+str(i+1))
            alpha_posterior= alpha_posterior+list((10**(df['alpha'].iloc[self.N_burns:])))
            h_posterior = h_posterior+list(10**(df['h'].iloc[self.N_burns:]))
            sigma_dA_posterior = sigma_dA_posterior+list((10**(df['sigma_dA'].iloc[self.N_burns:])))
            sigma_dP_posterior = sigma_dP_posterior+list((10**(df['sigma_dP'].iloc[self.N_burns:])))
            corr_posterior = corr_posterior+list(np.tanh(df['corr'].iloc[self.N_burns:]))
            chain_num = chain_num+list(df['chain_num'].iloc[self.N_burns:])

        df_posterior_burn = pd.DataFrame(data = {'alpha_posterior':alpha_posterior,'h_posterior':h_posterior,'sigma_dA_posterior':sigma_dA_posterior,'sigma_dP_posterior':sigma_dP_posterior,'corr_posterior':corr_posterior,'chain_num':chain_num})
        self.alpha_posterior = df_posterior_burn['alpha_posterior'].mean()
        self.h_posterior = df_posterior_burn['h_posterior'].mean()
        return df_posterior_burn
    
    def acf(x, length=20):
        return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])
    
    def auto_correlation_function(self,trace,lags):
        autocorr_trace = self.acf(trace,lags)
        return autocorr_trace
    
    def obtain_fitting_using_posterior(self):
        alpha,h = self.theoretical_Lopze_solution()
        plt.plot(alpha,h,label = 'posterior theoretical fitting',color = 'red')
        plt.scatter(self.df_amplitude_phase['phase_diff'],self.df_amplitude_phase['amp_ratio'],label = 'measurement')
        plt.show()
        