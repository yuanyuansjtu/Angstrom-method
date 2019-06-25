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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from scipy import signal
import seaborn as sns

class temperature_preprocessing_extract_phase_amplitude():

    def __init__(self, exp_setup, line_info, time_stamp):
        self.exp_setup = exp_setup
        # exp_setup = {'px':25/10**6,'f_heating':1,'gap':20}
        self.line_info = line_info
        # line_info = {'N_line_groups':N_line_groups,'N_horizontal_lines':N_horizontal_lines,'N_files':N_files}
        self.time_stamp = time_stamp

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=4):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    def filter_signal(self, df_rec, f0):
        cutoff = f0 * 0.5

        fs = (df_rec.shape[0]) / (max(df_rec['reltime']) - min(df_rec['reltime']))
        # Plot the frequency response for a few different orders.
        time = df_rec['reltime']
        N = df_rec.shape[1] - 1

        df_filtered = pd.DataFrame(data={'reltime': np.array(df_rec['reltime'])})

        for i in range(N):
            temp = (self.butter_highpass_filter(df_rec[i], cutoff, fs))
            df_filtered[i] = np.array(temp)
        return df_filtered

    def sin_func(self, x, amplitude, phase, bias, f_heating):
        return amplitude * np.sin(2 * np.pi * f_heating * x + phase) + bias

    def residual(self, params, x, data, eps_data):
        amplitude = params['amplitude']
        phase = params['phase']
        bias = params['bias']
        freq = params['frequency']

        model = amplitude * np.sin(2 * np.pi * freq * x + phase) + bias

        return (data - model) / eps_data

    def extract_phase_amplitude_sinusoidal_function(self, index, df_temperature):

        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']

        fitting_params_initial = {'amplitude': 0.2, 'phase': 0.1, 'bias': 0.1}

        n_col = df_temperature.shape[1]
        tmin = df_temperature['reltime'][0]
        time = df_temperature['reltime'] - tmin

        A1 = df_temperature[index[0]]  # the column number is numerical integer number, 0,1,....N-1
        A2 = df_temperature[index[1]]
        A1 -= A1.mean()
        A2 -= A2.mean()

        x0 = np.array([1, 0, 0])  # amplitude,phase,bias

        sigma = np.ones(len(time))

        params1 = Parameters()
        params1.add('amplitude', value=fitting_params_initial['amplitude'])
        params1.add('phase', value=fitting_params_initial['phase'])
        params1.add('bias', value=fitting_params_initial['bias'])
        params1.add('frequency', value=f_heating, vary=False)

        res1 = minimize(self.residual, params1, args=(time, A1, sigma))

        params2 = Parameters()
        params2.add('amplitude', value=fitting_params_initial['amplitude'])
        params2.add('phase', value=fitting_params_initial['phase'])
        params2.add('bias', value=fitting_params_initial['bias'])
        params2.add('frequency', value=f_heating, vary=False)
        res2 = minimize(self.residual, params2, args=(time, A2, sigma))

        amp1 = np.abs(res1.params['amplitude'].value)
        amp2 = np.abs(res2.params['amplitude'].value)

        p1 = res1.params['phase'].value
        p2 = res2.params['phase'].value

        amp_ratio = min(np.abs(amp1 / amp2), np.abs(amp2 / amp1))

        phase_diff = np.abs(p1 - p2)

        if phase_diff > 2 * np.pi:
            phase_diff = phase_diff - 2 * np.pi

        if phase_diff > np.pi / 2:
            phase_diff = np.pi - phase_diff

        T_total = np.max(time) - np.min(time)

        df = 1 / T_total

        L = abs(index[0] - index[1]) * px * gap
        w = 2 * np.pi * f_heating

        return L, phase_diff, amp_ratio

    def extract_phase_amplitude_Fourier_transform(self, index, df_temperature):

        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']

        n_col = df_temperature.shape[1]
        tmin = df_temperature['reltime'][0]
        time = df_temperature['reltime'] - tmin

        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]
        A1 -= A1.mean()
        A2 -= A2.mean()

        fft_X1 = np.fft.fft(A1)
        fft_X2 = np.fft.fft(A2)

        T_total = np.max(time) - np.min(time)
        df = 1 / T_total

        N_0 = int(f_heating / df)

        magnitude_X1 = np.abs(fft_X1)
        magnitude_X2 = np.abs(fft_X2)

        phase_X1 = np.angle(fft_X1)
        phase_X2 = np.angle(fft_X2)

        N1, Amp1 = max(enumerate(magnitude_X1[N_0 - 5:N_0 + 5]), key=operator.itemgetter(1))
        N2, Amp2 = max(enumerate(magnitude_X2[N_0 - 5:N_0 + 5]), key=operator.itemgetter(1))

        Nf = N_0 + N1 - 5
        amp_ratio = magnitude_X2[Nf] / magnitude_X1[Nf]
        phase_diff = phase_X1[Nf] - phase_X2[Nf]

        if phase_diff < 0:
            phase_diff = phase_diff + np.pi * 2

        L = abs(index[0] - index[1]) * px * gap

        return L, phase_diff, amp_ratio

    def fit_amp_phase_one_batch(self, df_temperature, method):

        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']

        N_lines = df_temperature.shape[1] - 1
        x_list = np.zeros(N_lines - 1)
        phase_diff_list = np.zeros(N_lines - 1)
        amp_ratio_list = np.zeros(N_lines - 1)

        for i in range(N_lines):
            if i > 0:
                index = [0, i]
                if method == 'fft':
                    x_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[
                        i - 1] = self.extract_phase_amplitude_Fourier_transform(index, df_temperature)
                else:
                    x_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[
                        i - 1] = self.extract_phase_amplitude_sinusoidal_function(index, df_temperature)
        return x_list, phase_diff_list, amp_ratio_list

    def extract_temperature_from_IR(self, X0, Y0, rec_name, N_avg):
        # this function takes the average of N pixels in Y0 direction, typically N = 100

        gap = self.exp_setup['gap']
        N_line_groups = self.line_info['N_line_groups']
        N_horizontal_lines = self.line_info['N_horizontal_lines']
        data_path = self.line_info['data_path']
        N_files = len(os.listdir(data_path))


        T = np.zeros((N_line_groups, N_horizontal_lines, N_files))
        for k in range(N_files):
            temp = pd.read_csv(data_path + rec_name + str(k) + '.csv')
            for j in range(N_line_groups):
                for i in range(N_horizontal_lines):
                    T[j, i, k] = temp.iloc[Y0 - int(N_avg / 2):Y0 + int(N_avg / 2),
                                 X0 - j - gap * i].mean()  # for T, first dim is line group, 2nd dimension is # of lines, 3rd dim is number of files
        return T

    def batch_process_horizontal_lines(self, T, method):

        # T averaged temperature for N_lines and N_line_groups and N_frames

        x_list_all = []
        phase_diff_list_all = []
        amp_ratio_list_all = []

        N_horizontal_lines = self.line_info['N_horizontal_lines']
        N_line_groups = self.line_info['N_line_groups']
        px = self.exp_setup['px']
        f_heating = self.exp_setup['f_heating']
        gap = self.exp_setup['gap']
        time_stamp = self.time_stamp

        for j in range(N_line_groups):
            horinzontal_temp = T[j, :, :].T
            df = pd.DataFrame(horinzontal_temp)
            df['reltime'] = time_stamp['reltime']  # add relative time to the dataframe of averaged temperature
            df_filtered = self.filter_signal(df, f_heating)
            x_list, phase_diff_list, amp_ratio_list = self.fit_amp_phase_one_batch(df_filtered, method)
            x_list_all = x_list_all + list(x_list)
            phase_diff_list_all = phase_diff_list_all + list(phase_diff_list)
            amp_ratio_list_all = amp_ratio_list_all + list(amp_ratio_list)

        df_result_IR = pd.DataFrame(
            data={'x': x_list_all, 'amp_ratio': amp_ratio_list_all, 'phase_diff': phase_diff_list_all})

        return df_result_IR


class post_processing_results:

    def __init__(self, df_posterior, processing_settings, exp_setting, df_amplitude_phase):
        self.alpha_posterior = 0
        self.h_posterior = 0
        self.N_burns = processing_settings['N_burns']
        self.N_chains = processing_settings['N_chains']
        self.theoretical_amp_ratio = []
        self.theoretical_phase_diff = []
        self.df_posterior = df_posterior
        self.exp_setting = exp_setting
        self.df_amplitude_phase = df_amplitude_phase

    def AL(self, x1, x2):
        return np.sqrt(x1 ** 4 / (np.sqrt(x1 ** 4 + x2 ** 4) + x2 ** 2))

    def BL(self, x1, x2):
        return np.sqrt(x2 ** 2 + np.sqrt(x1 ** 4 + x2 ** 4))

    def lopze_wave_vector(self, x1, x2):
        return (self.AL(x1, x2) + 1j * self.BL(x1, x2))

    def lopze_original(self, x1, x2, x3):
        amp = np.abs(np.cos(self.lopze_wave_vector(x1, x2) * (1 - x3)) / np.cos(self.lopze_wave_vector(x1, x2)))
        phase = np.angle(np.cos(self.lopze_wave_vector(x1, x2) * (1 - x3)) / np.cos(self.lopze_wave_vector(x1, x2)))

        return amp, phase

    def theoretical_Lopze_solution(self):
        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        px = self.exp_setting['px']
        r = self.exp_setting[
            'r']  # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']
        gap = self.exp_setting['gap']

        k = self.alpha_posterior * cp * rho

        x = self.df_amplitude_phase['x'].unique()
        x1 = np.sqrt(f_heating * 2 * np.pi / (2 * self.alpha_posterior)) * L
        x2 = np.sqrt(self.h_posterior / r / k) * L
        x3 = x / L  # x3 indicate different locations on the sample

        amp_ratio_list = np.zeros(len(x3))
        phase_diff_list = np.zeros(len(x3))

        for i, x3_ in enumerate(x3):
            amp_ratio_theoretical_, phase_diff_theoretical_ = self.lopze_original(x1, x2, x3_)
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
            df = self.df_posterior.query('chain_num ==' + str(i + 1))
            alpha_posterior = alpha_posterior + list((10 ** (df['alpha'].iloc[self.N_burns:])))
            h_posterior = h_posterior + list(10 ** (df['h'].iloc[self.N_burns:]))
            sigma_dA_posterior = sigma_dA_posterior + list((10 ** (df['sigma_dA'].iloc[self.N_burns:])))
            sigma_dP_posterior = sigma_dP_posterior + list((10 ** (df['sigma_dP'].iloc[self.N_burns:])))
            corr_posterior = corr_posterior + list(np.tanh(df['corr'].iloc[self.N_burns:]))
            chain_num = chain_num + list(df['chain_num'].iloc[self.N_burns:])

        df_posterior_burn = pd.DataFrame(data={'alpha_posterior': alpha_posterior, 'h_posterior': h_posterior,
                                               'sigma_dA_posterior': sigma_dA_posterior,
                                               'sigma_dP_posterior': sigma_dP_posterior,
                                               'corr_posterior': corr_posterior, 'chain_num': chain_num})
        self.alpha_posterior = df_posterior_burn['alpha_posterior'].mean()
        self.h_posterior = df_posterior_burn['h_posterior'].mean()
        return df_posterior_burn

    def acf(self, x, length):
        return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

    def auto_correlation_function(self, trace, lags):
        autocorr_trace = self.acf(trace, lags)
        return autocorr_trace

    def plot_auto_correlation(self, trace, lags):
        f = plt.figure(figsize=(7, 6))
        autocorr_trace = self.auto_correlation_function(trace, lags)
        plt.plot(autocorr_trace)
        plt.xlabel('lags N', fontsize=14, fontweight='bold')
        plt.ylabel('auto corr', fontsize=14, fontweight='bold')

        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        # plt.legend(prop={'weight': 'bold', 'size': 12})
        return f

    def obtain_fitting_using_posterior(self):
        alpha, h = self.theoretical_Lopze_solution()
        f = plt.figure(figsize=(7, 6))
        plt.plot(alpha, h, label='posterior theoretical fitting', color='red')
        plt.scatter(self.df_amplitude_phase['phase_diff'], self.df_amplitude_phase['amp_ratio'], label='measurement')
        plt.xlabel('dP (rad)', fontsize=14, fontweight='bold')
        plt.ylabel('dA', fontsize=14, fontweight='bold')

        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize=12)
            tick.label.set_fontweight('bold')
        plt.legend(prop={'weight': 'bold', 'size': 12})

        # plt.show()

