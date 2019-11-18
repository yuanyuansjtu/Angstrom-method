from scipy.io import loadmat
import operator
import tables
from lmfit import minimize, Parameters


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, os.path
import time
import scipy.signal
from scipy import signal
import scipy.optimize as optimization


from scipy.stats import norm
from scipy.optimize import minimize as minimize2
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed


class temperature_preprocessing_extract_phase_amplitude:

    def __init__(self, analysis_region,time_stamp):

        self.analysis_region = analysis_region
        # line_info = {'N_line_groups':N_line_groups,'N_horizontal_lines':N_horizontal_lines,'N_files':N_files}
        self.temp_full = None
        self.N_line_groups = None
        self.N_line_each_group = None

        file_path = self.analysis_region['file_path']
        file_name = self.analysis_region['file_name']
        file_skip = int(self.analysis_region['file_skip'])

        list_file = os.listdir(file_path)  # dir is your directory path
        self.N_files = len(list_file)
        frame_index = [i * (file_skip + 1) for i in range(int(self.N_files / (file_skip + 1)))]
        N_frame_keep = len(frame_index)

        time_stamp_after_skipping = time_stamp.iloc[frame_index]

        self.time_stamp = time_stamp_after_skipping.reset_index(drop=True)

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

        fs = self.N_files / (max(df_rec['reltime']) - min(df_rec['reltime']))
        # Plot the frequency response for a few different orders.
        time = df_rec['reltime']
        N = df_rec.shape[1] - 1 # Number of lines in each line group

        df_filtered = pd.DataFrame(data={'reltime': np.array(df_rec['reltime'])})
        # df_filtered is indexed by column number, therefore it is fine, the first column reltime doesn't matter here
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

        return (data - model) / eps_data # before this is only data-model, this can't be right for certain situations

    def cal_phase(self,p):
        if abs(p)>np.pi/2 and abs(p)<np.pi:
            return np.pi-abs(p)
        elif abs(p)>2*np.pi:
            return abs(p)-2*np.pi
        elif abs(p)>3/2*(np.pi) and abs(p)<2*np.pi:
            return 2*np.pi-abs(p)

    def extract_phase_amplitude_sinusoidal_function(self, index, df_temperature):

        px = self.analysis_region['px']
        f_heating = self.analysis_region['f_heating']
        gap = self.analysis_region['gap']

        fitting_params_initial = {'amplitude': 0.2, 'phase': 0.1, 'bias': 0.1}

        n_col = df_temperature.shape[1]
        tmin = min(df_temperature['reltime'])
        time = df_temperature['reltime'] - tmin

        A1 = df_temperature[index[0]]
        A2 = df_temperature[index[1]]
        A1 -= A1.mean()
        A2 -= A2.mean()

        x0 = np.array([0.1, 0, 0])  # amplitude,phase,bias

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
        phase_diff = phase_diff%np.pi
        if phase_diff > np.pi/2:
            phase_diff = np.pi - phase_diff

        T_total = np.max(time) - np.min(time)

        df = 1 / T_total

        L = abs(index[0] - index[1]) * px * gap # here does not consider the offset between the heater and the region of analysis
        w = 2 * np.pi * f_heating

        return L, phase_diff, amp_ratio

    def extract_phase_amplitude_Fourier_transform(self, index, df_temperature):

        px = self.analysis_region['px']
        f_heating = self.analysis_region['f_heating']
        gap = self.analysis_region['gap']

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

        N_f = 2
        N1, Amp1 = max(enumerate(magnitude_X1[N_0 - N_f:N_0 + N_f]), key=operator.itemgetter(1))
        N2, Amp2 = max(enumerate(magnitude_X2[N_0 - N_f:N_0 + N_f]), key=operator.itemgetter(1))

        Nf = N_0 + N1 - N_f
        amp_ratio = magnitude_X2[Nf] / magnitude_X1[Nf]
        phase_diff = abs(phase_X1[Nf] - phase_X2[Nf])
        #phase_diff = np.abs(p1 - p2)
        phase_diff = phase_diff % np.pi
        if phase_diff > np.pi / 2:
            phase_diff = np.pi - phase_diff

        L = abs(index[0] - index[1]) * px * gap

        return L, phase_diff, amp_ratio

    def fit_amp_phase_one_batch(self,df_temperature,X0,Y0):

        px = self.analysis_region['px']
        direction = self.analysis_region['direction']

        N_lines = df_temperature.shape[1] - 1
        x_list = np.zeros(N_lines - 1)
        phase_diff_list = np.zeros(N_lines - 1)
        amp_ratio_list = np.zeros(N_lines - 1)
        x_ref_list = np.zeros(N_lines - 1)

        if direction == 'right-left' or direction == 'left-right':
            delta_line = abs(self.analysis_region['x_heater'] - X0)*px

        elif direction == 'bottom-up' or direction == 'up-bottom':
            delta_line = abs(self.analysis_region['y_heater'] - Y0)*px


        for i in range(N_lines):

            if i > 0:
                index = [0, i]

                if self.analysis_region['analysis_method'] == 'fft':
                    x_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[
                        i - 1] = self.extract_phase_amplitude_Fourier_transform(index, df_temperature)
                else:
                    x_list[i - 1], phase_diff_list[i - 1], amp_ratio_list[
                        i - 1] = self.extract_phase_amplitude_sinusoidal_function(index, df_temperature)

                x_list[i - 1] = x_list[i - 1] + delta_line
                x_ref_list[i-1] = delta_line

        return x_list,x_ref_list, phase_diff_list, amp_ratio_list

    def load_temperature_profiles(self):

        file_path = self.analysis_region['file_path']
        file_name = self.analysis_region['file_name']
        file_skip = int(self.analysis_region['file_skip'])


        N_files = self.N_files
        frame_index = [i*(file_skip+1) for i in range(int(N_files/(file_skip+1)))]
        N_frame_keep = len(frame_index)


        shape_single_frame = pd.read_csv(file_path + file_name+'_0.csv', header=None).shape # frame 0 must be there

        temp_full = np.zeros((shape_single_frame[0], shape_single_frame[1], N_frame_keep))

        for idx,frame_idx in enumerate(frame_index):
            temp = pd.read_csv(file_path + file_name+ '_' +str(frame_idx) + '.csv', header=None).values.tolist()
            #a = temp.values.tolist()
            if np.shape(temp)[0] == 480 and np.shape(temp)[1] == 640:
                temp_full[:, :, idx] = temp
            else:
                print('The input file at index '+str(frame_idx) +' is illegal!')

        self.temp_full = temp_full
        return temp_full

    def extract_temperature_from_IR(self):
        # this function takes the average of N pixels in Y0 direction, typically N = 100

        gap = self.analysis_region['gap']
        X0 =self.analysis_region['x_region_line_center']
        Y0 = self.analysis_region['y_region_line_center']
        direction = self.analysis_region['direction']

        N_frame_keep = self.time_stamp.shape[0]


        self.N_line_groups = gap - 1
        if direction == 'left-right' or direction == 'right-left':
            self.N_line_each_group = int(self.analysis_region['dx']/gap)-1
        elif direction == 'up-bottom' or direction == 'bottom-up':
            self.N_line_each_group = int(self.analysis_region['dy'] / gap) - 1

        T = np.zeros((self.N_line_groups, self.N_line_each_group, N_frame_keep))
        for k in range(N_frame_keep):
            for j in range(self.N_line_groups):
                for i in range(self.N_line_each_group):
                    if direction == 'right-left':
                        N_avg = self.analysis_region['dy']
                        T[j, i, k] = self.temp_full[Y0 - int(N_avg / 2):Y0 + int(N_avg / 2),
                                     X0 - j - gap * i, k].mean()
                    elif direction == 'left-right':
                        N_avg = self.analysis_region['dy']
                        T[j, i, k] = self.temp_full[Y0 - int(N_avg / 2):Y0 + int(N_avg / 2),
                                     X0 + j + gap * i, k].mean()
                    elif direction == 'bottom-up':
                        N_avg = self.analysis_region['dx']
                        T[j, i, k] = self.temp_full[Y0 - j - gap * i,
                                     X0 - int(N_avg / 2):X0 + int(N_avg / 2), k].mean()
                    elif direction == 'up-bottom':
                        N_avg = self.analysis_region['dx']
                        T[j, i, k] = self.temp_full[Y0 + j + gap * i,
                                     X0 - int(N_avg / 2):X0 + int(N_avg / 2), k].mean()
        return T

    def batch_process_horizontal_lines(self,apply_filter):

        # T averaged temperature for N_lines and N_line_groups and N_frames
        T = self.extract_temperature_from_IR()
        x_list_all = []
        x_ref_list_all = []
        phase_diff_list_all = []
        amp_ratio_list_all = []

        X0 = self.analysis_region['x_region_line_center']
        Y0 = self.analysis_region['y_region_line_center']

        f_heating = self.analysis_region['f_heating']
        time_stamp = self.time_stamp
        direction = self.analysis_region['direction']


        # N = df_rec.shape[1] - 1
        #
        # df_filtered = pd.DataFrame(data={'reltime': np.array(df_rec['reltime'])})
        #
        # for i in range(N):
        #     temp = (self.butter_highpass_filter(df_rec[i], cutoff, fs))
        #     df_filtered[i] = np.array(temp)
        # return df_filtered

        for j in range(self.N_line_groups):

            horinzontal_temp = T[j, :, :].T
            df = pd.DataFrame(horinzontal_temp)
            df['reltime'] = time_stamp['reltime']
            if apply_filter == True:
                df_filtered = self.filter_signal(df, f_heating)
            else:
                # Do not apply filter
                df_filtered = df

            if direction == 'right-left':
                X0 = X0 - 1 # everytime the moves one piexel only!
            elif direction == 'left-right':
                X0 = X0 + 1 #
            elif direction == 'up-bottom':
                Y0 = Y0 + 1
            elif direction == 'bottom-up':
                Y0 = Y0 - 1

            x_list, x_ref_list, phase_diff_list, amp_ratio_list = self.fit_amp_phase_one_batch(df_filtered,X0,Y0)

            x_list_all = x_list_all + list(x_list)
            x_ref_list_all = x_ref_list_all + list(x_ref_list)

            phase_diff_list_all = phase_diff_list_all + list(phase_diff_list)
            amp_ratio_list_all = amp_ratio_list_all + list(amp_ratio_list)

        df_result_IR = pd.DataFrame(
            data={'x': x_list_all, 'x_ref':x_ref_list_all,'amp_ratio': amp_ratio_list_all, 'phase_diff': phase_diff_list_all})

        return df_result_IR

    def load_phase_ampltude_csv(self):

        file_name = self.analysis_region['file_name']
        x_pos = self.analysis_region['x_region_line_center']
        y_pos = self.analysis_region['y_region_line_center']
        dx = self.analysis_region['dx']
        dy = self.analysis_region['dy']
        gap = self.analysis_region['gap']
        method = self.analysis_region['analysis_method']
        file_path = self.analysis_region['file_path']
        direction = self.analysis_region['direction']
        x_heater = self.analysis_region['x_heater']
        y_heater = self.analysis_region['y_heater']
        frequency = self.analysis_region['f_heating']
        directory_path = self.analysis_region['directory_path']

        phase_amp_loc_info = file_name + '_frequency_' + str(int(frequency * 1000)) + 'mHz' + '_xpos_' + str(
            x_pos) + '_ypos_' + str(
            y_pos) + '_dx_' + str(dx) + '_dy_' + str(dy) + '_xheater_' + str(x_heater) + '_yheater_' + str(
            y_heater) + '_gap_' + str(
            gap) + '_dir_' + direction + '_method_' + method + '.csv'

        df_phase_amplitude = pd.read_csv(directory_path + 'phase amplitude data//' + phase_amp_loc_info)
        return df_phase_amplitude

    def save_phase_amplitude_to_csv(self,df_amplitude_phase):
        file_name = self.analysis_region['file_name']
        x_pos  = self.analysis_region['x_region_line_center']
        y_pos = self.analysis_region['y_region_line_center']
        dx = self.analysis_region['dx']
        dy = self.analysis_region['dy']
        gap = self.analysis_region['gap']
        method = self.analysis_region['analysis_method']
        file_path = self.analysis_region['file_path']
        direction = self.analysis_region['direction']
        x_heater = self.analysis_region['x_heater']
        y_heater = self.analysis_region['y_heater']
        frequency = self.analysis_region['f_heating']
        directory_path = self.analysis_region['directory_path']

        phase_amp_loc_info = file_name + '_frequency_'+str(int(frequency*1000))+'mHz' +'_xpos_' + str(x_pos) + '_ypos_' + str(
            y_pos) + '_dx_' + str(dx) + '_dy_' + str(dy) + '_xheater_'+ str(x_heater)+ '_yheater_'+ str(y_heater) + '_gap_' + str(
            gap) + '_dir_' + direction + '_method_' + method + '.csv'
        df_amplitude_phase.to_csv(directory_path + 'phase amplitude data//' + phase_amp_loc_info)

class angstrom_conventional_regression:
    pass

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
        x3 = x / (L)

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

def AL(x1, x2):
    #x1 = np.sqrt(f_heating * 2 * np.pi / (2 * alpha)) * L
    #x2 = np.sqrt(h / r / k) * L
    return np.sqrt(x1 ** 4 / (np.sqrt(x1 ** 4 + x2 ** 4) + x2 ** 2))

def BL(x1, x2):

    return np.sqrt(x2 ** 2 + np.sqrt(x1 ** 4 + x2 ** 4))

def lopze_wave_vector(x1, x2):

    return (AL(x1, x2) + 1j * BL(x1, x2))

def lopze_original(x1, x2, x3):

    amp = np.abs(np.cos(lopze_wave_vector(x1, x2) * (1 - x3)) / np.cos(lopze_wave_vector(x1, x2)))
    phase = np.angle(np.cos(lopze_wave_vector(x1, x2) * (1 - x3)) / np.cos(lopze_wave_vector(x1, x2)))

    return amp, phase

def calculate_theoretical_results(material_properties, analysis_region, df_phase_diff_amp_ratio, params):
    #material_properties = {'L':3.76e-3,'f_heating':f_heating,'px':25/10**6,'r':5.15e-3,'cp':693,'rho':3120,'gap':5}

    L = material_properties['L']
    f_heating = analysis_region['f_heating']

    r = material_properties['r']  # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
    cp = material_properties['cp']
    rho = material_properties['rho']

    alpha, h = 10 ** params[0], 10 ** params[1]

    w = 2 * np.pi * f_heating
    k = alpha * cp * rho

    x1 = np.sqrt(w / (2 * alpha)) * L
    x2 = np.sqrt(h / r / k) * L

        # data={'x': x_list_all, 'x_ref':x_ref_list_all,'amp_ratio': amp_ratio_list_all, 'phase_diff': phase_diff_list_all})

    x = np.array(df_phase_diff_amp_ratio['x'])
    x0 = np.array(df_phase_diff_amp_ratio['x_ref'])


    theoretical_amp_ratio_array = np.zeros(len(x))
    theoretical_phase_diff_array = np.zeros(len(x))

    for i, x_ in enumerate(x):

        x0_ = x0[i]
        x3_ref = x0_/L

        x3_ = x_/L

        amp_ratio_theoretical_ref, phase_diff_theoretical_ref = lopze_original(x1, x2, x3_ref)
        amp_ratio_theoretical_x, phase_diff_theoretical_x = lopze_original(x1, x2, x3_)

        amp_ratio_theoretical_ = amp_ratio_theoretical_x/amp_ratio_theoretical_ref
        phase_diff_theoretical_ = phase_diff_theoretical_x- phase_diff_theoretical_ref

        theoretical_amp_ratio_array[i] = amp_ratio_theoretical_
        theoretical_phase_diff_array[i] = phase_diff_theoretical_

    return theoretical_amp_ratio_array, theoretical_phase_diff_array



class least_square_regression_Angstrom:

    def __init__(self, params_init, analysis_region, df_phase_diff_amp_ratio, material_properties):

        self.params_init = params_init
        self.df_phase_diff_amp_ratio = df_phase_diff_amp_ratio
        self.material_properties = material_properties
        self.analysis_region = analysis_region
        self.alpha_fitting = None
        self.h_fitting = None

    def least_square_regression(self, params,analyze_method):

        theoretical_amp_ratio_array, theoretical_phase_diff_array = calculate_theoretical_results(self.material_properties,self.analysis_region,
                                                                                                  self.df_phase_diff_amp_ratio,
                                                                                                  params)
        x = np.array(self.df_phase_diff_amp_ratio['x'])
        amp_ratio_measurement = np.array(self.df_phase_diff_amp_ratio['amp_ratio'])
        phase_diff_measurement = np.array(self.df_phase_diff_amp_ratio['phase_diff'])
        err = np.zeros(len(x))

        for i, x_ in enumerate(x):
            if analyze_method== 'phase-amplitude':
                err_ = (theoretical_amp_ratio_array[i] - amp_ratio_measurement[i]) ** 2 + (
                        theoretical_phase_diff_array[i] - phase_diff_measurement[i]) ** 2
            elif analyze_method == 'phase':
                err_ = (theoretical_phase_diff_array[i] - phase_diff_measurement[i]) ** 2
            elif analyze_method == 'amplitude':
                err_ = (theoretical_amp_ratio_array[i] - amp_ratio_measurement[i]) ** 2

            err[i] = err_
        return np.sum(err)

    def minimize_regression(self,analyze_method):
        guess = self.params_init[:2]
        results = minimize2(self.least_square_regression, x0 = guess,args = (analyze_method), method='Nelder-Mead', options={'disp': True})
        self.alpha_fitting = (results['final_simplex'][0][1][0])
        self.h_fitting = (results['final_simplex'][0][1][1])
        return results


    def show_fitting_results(self):
        #alpha, h = self.theoretical_Lopze_solution()
        theoretical_amp_ratio_array, theoretical_phase_diff_array = calculate_theoretical_results(self.material_properties,self.analysis_region,
                                                                                                  self.df_phase_diff_amp_ratio,
                                                                                                  [self.alpha_fitting,self.h_fitting])

        f = plt.figure(figsize=(7, 6))
        plt.scatter(self.df_phase_diff_amp_ratio['phase_diff'], self.df_phase_diff_amp_ratio['amp_ratio'], label='measurement', alpha = 0.4)
        plt.scatter(theoretical_phase_diff_array, theoretical_amp_ratio_array, label='posterior theoretical fitting', color='red')

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


class Metropolis_Hasting_sampler:

    def __init__(self, params_init, prior_log_mu, prior_log_sigma, df_phase_diff_amp_ratio, exp_setting, N_sample,
                 transition_sigma, result_name):

        self.params_init = params_init
        self.prior_log_mu = prior_log_mu
        self.prior_log_sigma = prior_log_sigma
        self.exp_setting = exp_setting
        self.df_phase_diff_amp_ratio = df_phase_diff_amp_ratio
        self.N_sample = N_sample
        self.transition_sigma = transition_sigma
        self.result_name = result_name
        # self.accepted = []

        # updated!
        self.alpha_posterior = 0
        self.h_posterior = 0


    def log_likelihood(self, params):

        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        px = self.exp_setting['px']
        r = self.exp_setting[
            'r']  # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']
        gap = self.exp_setting['gap']

        alpha, h = 10 ** params[0], 10 ** params[1]

        sigma_dA = 10 ** params[2]
        sigma_dP = 10 ** params[3]
        z = params[4]
        rho_dA_dP = np.tanh(z)

        k = alpha * cp * rho

        x = self.df_phase_diff_amp_ratio['x']
        x1 = np.sqrt(f_heating * 2 * np.pi / (2 * alpha)) * L
        x2 = np.sqrt(h / r / k) * L
        x3 = x / L  # x3 indicate different locations on the sample

        amp_ratio_measurement = self.df_phase_diff_amp_ratio['amp_ratio']
        phase_diff_measurement = self.df_phase_diff_amp_ratio['phase_diff']

        p_joint_dA_dP = np.zeros(len(x3))

        for i, x3_ in enumerate(x3):
            amp_ratio_theoretical_, phase_diff_theoretical_ = lopze_original(x1, x2, x3_)
            x_ = [amp_ratio_measurement[i], phase_diff_measurement[i]]
            mean = [amp_ratio_theoretical_, phase_diff_theoretical_]
            cov = [[sigma_dA ** 2, sigma_dA * sigma_dP * rho_dA_dP], [sigma_dA * sigma_dP * rho_dA_dP, sigma_dP ** 2]]

            p_joint_dA_dP[i] = np.log(multivariate_normal.pdf(x_, mean, cov))

        return np.sum(p_joint_dA_dP)

    def log_likelihood_updated(self,params):
        L = self.exp_setting['L']
        f_heating = self.exp_setting['f_heating']
        r = self.exp_setting[
            'r']  # note this is the parameter for cylinderical case, for rectangular situation, it is ab/(a+b)
        cp = self.exp_setting['cp']
        rho = self.exp_setting['rho']

        alpha, h = 10 ** params[0], 10 ** params[1]

        sigma_dA = 10 ** params[2]
        sigma_dP = 10 ** params[3]
        z = params[4]
        rho_dA_dP = np.tanh(z)

        k = alpha * cp * rho

        x1 = np.sqrt(f_heating * 2 * np.pi / (2 * alpha)) * L
        x2 = np.sqrt(h / r / k) * L

        x = self.df_phase_diff_amp_ratio['x'] # the absolute location on the sample
        x0 = self.df_phase_diff_amp_ratio['x_ref']

        amp_ratio_measurement = self.df_phase_diff_amp_ratio['amp_ratio']
        phase_diff_measurement = self.df_phase_diff_amp_ratio['phase_diff']

        p_joint_dA_dP = np.zeros(len(x))

        for i, x_ in enumerate(x):
            x0_ = x0[i]
            x3_ref = x0_ / L

            x3_ = x_ / L

            amp_ratio_theoretical_ref, phase_diff_theoretical_ref = lopze_original(x1, x2, x3_ref)
            amp_ratio_theoretical_x, phase_diff_theoretical_x = lopze_original(x1, x2, x3_)

            amp_ratio_theoretical_ = amp_ratio_theoretical_x / amp_ratio_theoretical_ref
            phase_diff_theoretical_ = phase_diff_theoretical_x - phase_diff_theoretical_ref

            mean_measured = [amp_ratio_measurement[i], phase_diff_measurement[i]]
            mean_theoretical = [amp_ratio_theoretical_, phase_diff_theoretical_]

            cov_errs = [[sigma_dA ** 2, sigma_dA * sigma_dP * rho_dA_dP], [sigma_dA * sigma_dP * rho_dA_dP, sigma_dP ** 2]]

            p_joint_dA_dP[i] = np.log(multivariate_normal.pdf(mean_measured, mean_theoretical, cov_errs))

        return np.sum(p_joint_dA_dP)


    def acceptance(self, x, x_new):
        if x_new > x:
            return True
        else:
            accept = np.random.uniform(0, 1)
            return (accept < (np.exp(x_new - x)))

    def manual_priors(self, params):  # this is actually proposal distribution's probability

        p_log_alpha = norm.pdf(params[0], loc=self.prior_log_mu[0], scale=self.prior_log_sigma[0])
        p_log_h = norm.pdf(params[1], loc=self.prior_log_mu[1], scale=self.prior_log_sigma[1])

        p_log_sigma_dA = norm.pdf(params[2], loc=self.prior_log_mu[2], scale=self.prior_log_sigma[2])
        p_log_sigma_dP = norm.pdf(params[3], loc=self.prior_log_mu[3], scale=self.prior_log_sigma[3])

        return np.log(p_log_alpha) + np.log(p_log_h) + np.log(p_log_sigma_dA) + np.log(p_log_sigma_dP)

    def rw_proposal(self, params):
        rho_alpha_h = -0.6
        rho_sigma_dA_dP = 0
        [alpha, h] = np.random.multivariate_normal(params[0:2], [
            [self.transition_sigma[0] ** 2, rho_alpha_h * self.transition_sigma[0] * self.transition_sigma[1]],
            [rho_alpha_h * self.transition_sigma[0] * self.transition_sigma[1], self.transition_sigma[1] ** 2]])
        #[sigma_dA, sigma_dP, rho] = np.random.normal(params[2:5], scale=self.transition_sigma[2:5])
        [sigma_dA,sigma_dP] = np.random.multivariate_normal(params[2:4],
                                                            [[self.transition_sigma[2]**2,rho_sigma_dA_dP*self.transition_sigma[2]*self.transition_sigma[3]],
                                                             [rho_sigma_dA_dP*self.transition_sigma[2]*self.transition_sigma[3],self.transition_sigma[3]**2]])
        rho = np.random.normal(params[4], scale=self.transition_sigma[4])

        return [alpha, h, sigma_dA, sigma_dP, rho]

    def metropolis_hastings_rw(self):

        params = self.params_init
        accepted = []
        # rejected = []
        n_sample = 0
        n_rej = 1
        temp_data = []

        #transition_model_rw = lambda x: np.random.normal(x, scale=self.transition_sigma)

        while (n_sample < self.N_sample):

            #params_new = transition_model_rw(params)
            params_new = self.rw_proposal(params)
            params_lik = self.log_likelihood_updated(params)
            params_new_lik = self.log_likelihood_updated(params_new)
            # params_new[0:2] = transition_rw_alpha_h(params[0:2])
            jac = np.log(10 ** (np.sum(params[0:4]))) + np.log(
                (2 + 2 * np.exp(4 * params[4])) / (1 + np.exp(2 * params[4])) ** 2)
            jac_new = np.log(10 ** (np.sum(params_new[0:4]))) + np.log(
                (2 + 2 * np.exp(4 * params_new[4])) / (1 + np.exp(2 * params_new[4])) ** 2)

            if (self.acceptance(params_lik + self.manual_priors(params) + jac,
                                params_new_lik + self.manual_priors(params_new) + jac_new)):

                n_sample += 1
                accept_rate = n_sample/(n_rej+n_sample)

                params = params_new
                temp_data = params_new + [time.time(),accept_rate]
                accepted.append(temp_data)
                print('iter ' + str(n_sample) + ', accepted: ' + str(
                    params_new) + ', acceptance rate: ' + "{0:.4g}".format(n_sample / n_rej)+', time:'+"{0:.7g}".format(time.time()))
            else:
                n_rej += 1

        accepted = np.array(accepted)

        return accepted




def multi_chain_Metropolis_Hasting(params_init, prior_log_mu, prior_log_sigma, df_phase_diff_amp_ratio, exp_setting,
                                   N_sample, transition_sigma, result_name, N_chains):
    # execute Metropolis-Hasting algorithm using parallel processing
    chains = [
        Metropolis_Hasting_sampler(params_init, prior_log_mu, prior_log_sigma, df_phase_diff_amp_ratio, exp_setting,
                                   N_sample, transition_sigma, result_name) for i in range(N_chains)]
    results = Parallel(n_jobs=N_chains)(delayed(chain.metropolis_hastings_rw)() for chain in chains)

    all_chain_results = np.reshape(results, (-1, 7))
    chain_num = [int(i / N_sample) + 1 for i in range(len(all_chain_results))]

    df_posterior = pd.DataFrame(data=all_chain_results)
    df_posterior.columns = ['alpha', 'h', 'sigma_dA', 'sigma_dP', 'corr','time','acceptance_rate']
    df_posterior['chain_num'] = chain_num
    return df_posterior