from scipy.io import loadmat
import tables
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path
import time
import scipy.signal
from scipy import signal
from lmfit import minimize, Parameters
import scipy.optimize as optimization
import operator

class temperature_preprocessing_extract_phase_amplitude():
    
    def __init__(self,exp_setup,line_info,time_stamp):
        self.exp_setup = exp_setup
        # exp_setup = {'px':25/10**6,'f_heating':1,'gap':20}
        self.line_info = line_info 
        # line_info = {'N_line_groups':N_line_groups,'N_horizontal_lines':N_horizontal_lines,'N_files':N_files}
        self.time_stamp = time_stamp
        
    def butter_highpass(self,cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self,data, cutoff, fs, order=4):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def filter_signal(self,df_rec,f0):
        cutoff = f0*0.5

        fs = (df_rec.shape[0])/(max(df_rec['reltime'])-min(df_rec['reltime']))
        # Plot the frequency response for a few different orders.
        time = df_rec['reltime']
        N = df_rec.shape[1]-1

        df_filtered = pd.DataFrame(data = {'reltime':np.array(df_rec['reltime'])})

        for i in range(N):
            temp = (butter_highpass_filter(df_rec[i],cutoff,fs))
            df_filtered[i] = np.array(temp)
        return df_filtered
    
    def sin_func(self,x,amplitude,phase,bias,f_heating):
        return amplitude*np.sin(2*np.pi*f_heating*x + phase)+bias

    def residual(self,params, x, data, eps_data):
        amplitude = params['amplitude']
        phase = params['phase']
        bias = params['bias']
        freq = params['frequency']

        model = amplitude*np.sin(2*np.pi*freq*x + phase)+bias

        return (data-model) / eps_data

    def extract_phase_amplitude_sinusoidal_function(self,index,df_temperature,exp_setup):

        px = exp_setup['px']
        f_heating = exp_setup['f_heating']
        gap = exp_setup['gap']

        fitting_params_initial = {'amplitude':0.2,'phase':0.1,'bias':0.1}

        n_col = df_temperature.shape[1]
        tmin = df_temperature['reltime'][0]
        time = df_temperature['reltime']-tmin


#         A1 = df_temperature.iloc[:,index[0]+3]
#         A2 = df_temperature.iloc[:,index[1]+3]
        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]
        A1-= A1.mean()
        A2-= A2.mean()

        x0 = np.array([1,0,0]) # amplitude,phase,bias

        sigma = np.ones(len(time))

        params1 = Parameters()
        params1.add('amplitude', value=fitting_params_initial['amplitude'])
        params1.add('phase', value=fitting_params_initial['phase'])
        params1.add('bias', value=fitting_params_initial['bias'])
        params1.add('frequency', value=f_heating,vary=False)

        res1 = minimize(residual, params1, args=(time, A1, sigma))

        params2 = Parameters()
        params2.add('amplitude', value=fitting_params_initial['amplitude'])
        params2.add('phase', value=fitting_params_initial['phase'])
        params2.add('bias', value=fitting_params_initial['bias'])
        params2.add('frequency', value=f_heating,vary=False)
        res2 = minimize(residual, params2, args=(time, A2, sigma))

        amp1 = np.abs(res1.params['amplitude'].value)
        amp2 = np.abs(res2.params['amplitude'].value)

        p1 = res1.params['phase'].value
        p2 = res2.params['phase'].value


        amp_ratio = min(np.abs(amp1/amp2),np.abs(amp2/amp1))
        
        phase_diff = np.abs(p1-p2)
        if phase_diff>2*np.pi:
            phase_diff = phase_diff - 2*np.pi

        if phase_diff>np.pi/2:
            phase_diff = np.pi - phase_diff

        T_total = np.max(time)-np.min(time)

        df = 1/T_total

        L = abs(index[0]-index[1])*px*gap
        w = 2*np.pi*f_heating

        return L, phase_diff,amp_ratio
    
    def extract_phase_amplitude_Fourier_transform(self,index,df_temperature):
        
        
        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']
        
        n_col = df_temperature.shape[1]
        tmin = df_temperature['reltime'][0]
        time = df_temperature['reltime']-tmin

        fft_X1 = np.fft.fft(df_temperature.iloc[:,index[0]+3])
        fft_X2 = np.fft.fft(df_temperature.iloc[:,index[1]+3])

        T_total = np.max(time)-np.min(time)
        df = 1/T_total

        N_0 = int(f_heating/df)

        magnitude_X1 = np.abs(fft_X1)
        magnitude_X2 = np.abs(fft_X2)

        phase_X1 = np.angle(fft_X1)
        phase_X2 = np.angle(fft_X2)    

        N1, Amp1 = max(enumerate(magnitude_X1[N_0-5:N_0+5]), key=operator.itemgetter(1))
        N2, Amp2 = max(enumerate(magnitude_X2[N_0-5:N_0+5]), key=operator.itemgetter(1))

        Nf = N_0+N1-5
        amp_ratio = magnitude_X1[Nf]/magnitude_X2[Nf]
        phase_diff = phase_X1[Nf]-phase_X2[Nf]

        if phase_diff<0:
            phase_diff = phase_diff+np.pi*2

        L = abs(index[0]-index[1])*px*gap
    

        return L, phase_diff,amp_ratio
    
            
    def fit_amp_phase_one_batch(self,df_temperature,method):
    
        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']

        N_lines = df_temperature.shape[1]-1
        x_list = np.zeros(N_lines-1)
        phase_diff_list = np.zeros(N_lines-1)
        amp_ratio_list = np.zeros(N_lines-1)

        for i in range(N_lines):
            if i>0:
                index = [0,i]
                if method == 'fft':
                    x_list[i-1],phase_diff_list[i-1], amp_ratio_list[i-1] = self.extract_phase_amplitude_Fourier_transform(index,df_temperature)
                else:
                    x_list[i-1],phase_diff_list[i-1], amp_ratio_list[i-1] = self.extract_phase_amplitude_sinusoidal_function(index,df_temperature,exp_setup)
        return x_list,phase_diff_list,amp_ratio_list
    
    def extract_temperature_from_IR(self,X0,Y0,rec_name,N_avg):
        # this function takes the average of N pixels in Y0 direction, typically N = 100

        gap = self.exp_setup['gap']
        N_line_groups = self.line_info['N_line_groups']
        N_horizontal_lines = self.line_info['N_horizontal_lines']
        N_files = self.line_info['N_files']

        T = np.zeros((N_line_groups,N_horizontal_lines,N_files))
        for k in range(N_files):
            temp = pd.read_csv(data_path+rec_name+str(k)+'.csv')
            for j in range(N_line_groups):
                for i in range(N_horizontal_lines):
                    T[j,i,k] = temp.iloc[Y0-int(N_avg/2):Y0+int(N_avg/2),X0-j-gap*i].mean() # for T, first dim is line group, 2nd dimension is # of lines, 3rd dim is number of files 
        return T
    
    def batch_process_horizontal_lines(self,T,method):
        
        #T averaged temperature for N_lines and N_line_groups and N_frames
        
        x_list_all = []
        phase_diff_list_all = []
        amp_ratio_list_all = []
        
        N_horizontal_lines = self.line_info['N_horizontal_lines']

        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']
        time_stamp = self.time_stamp
        
        for j in range(N_line_groups):
            horinzontal_temp = T[j,:,:].T
            df = pd.DataFrame(horinzontal_temp)
            df['reltime'] = time_stamp['reltime']
            df_filtered = filter_signal(df,f_heating)
            x_list,phase_diff_list,amp_ratio_list = self.fit_amp_phase_one_batch(df_filtered,method)
            x_list_all = x_list_all+list(x_list)
            phase_diff_list_all = phase_diff_list_all+list(phase_diff_list)
            amp_ratio_list_all = amp_ratio_list_all+list(amp_ratio_list)

        df_result_IR = pd.DataFrame(data = {'x':x_list_all,'amp_ratio':amp_ratio_list_all,'phase_diff':phase_diff_list_all})

        return df_result_IR