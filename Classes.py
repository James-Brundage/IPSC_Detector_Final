import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import find_peaks
from scipy import signal
import pyabf
from tqdm import tqdm
import pickle
from Preprocessing import preprocessing

test_file = '7_30_2019 264 Continuous Export.abf'


class RecordingFile:

    def __init__(self, file_path):
        self.abf_init = self.read_abf(file_path)
        self.guesses = self.recording_prep()
        self.processed = self.process_file()
        self.peak_info = self.get_info()
        self.predict = self.predictions()
        self.time = len(self.abf_init)

    def read_abf(self, path, slicer=True, s_val=0, e_val=1000000, plot_me=False):
        # Reads in the abf file
        abf = pyabf.ABF(path)

        # Pulls the time component (x) and the original current component (y)
        x = abf.sweepX
        y = abf.sweepY

        # Slicer function for testing the filter on smaller sizes
        if slicer == True:
            def slicer(x, y, s_val=0, e_val=10000):
                x = x[s_val:e_val]
                y = y[s_val:e_val]

                return x, y

            x, y = slicer(x, y, s_val, e_val)

        # Plots data if desired
        if plot_me == True:
            plt.plot(x, y)
            plt.show()

        # Creates a dataframe with the x as time and y as current.
        dff = pd.DataFrame({
            'Time': x,
            'Current': y
        })

        return dff

    def apply_filter (self, df):

        # Selects the current column for filtering
        y = df['Current']

        # Filter settings and filtering
        b, a = signal.butter(3, (0.0003, 0.07), btype='bp', analog=False)
        filt_trace = signal.filtfilt(b, a, y)

        dff = pd.DataFrame({
            'Time': df['Time'],
            'Current_F': filt_trace
        })

        return dff

    def invert_trace (self, df):

        cols = list(df.columns)
        trace = df[cols[1]]
        trc = [float(i) * (-1) for i in trace]

        dff = pd.DataFrame({
            cols[0]: df[cols[0]],
            cols[1]: trc
        })

        return dff

    def smooth_trace (self, df):

        def smooth(x, window_len=400, window='hanning'):
            """smooth the data using a window with requested size.

            This method is based on the convolution of a scaled window with the signal.
            The signal is prepared by introducing reflected copies of the signal
            (with the window size) in both ends so that transient parts are minimized
            in the begining and end part of the output signal.

            input:
                x: the input signal
                window_len: the dimension of the smoothing window; should be an odd integer
                window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.

            output:
                the smoothed signal

            example:

            t=linspace(-2,2,0.1)
            x=sin(t)+randn(len(t))*0.1
            y=smooth(x)

            see also:

            numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
            scipy.signal.lfilter

            TODO: the window parameter could be the window itself if an array instead of a string
            NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
            """

            if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")

            if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")

            if window_len < 3:
                return x

            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

            s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
            # print(len(s))
            if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
            else:
                w = eval('np.' + window + '(window_len)')

            y = np.convolve(w / w.sum(), s, mode='valid')
            return y

        cols = list(df.columns)
        trace = df[cols[1]]
        trace_s = smooth(trace)

        dff = pd.DataFrame({
            cols[0]: df[cols[0]],
            cols[1]: trace_s[:len(trace)]
        })

        return dff

    def find_pot_peaks (self, df, plot_me=False):

        # Gets right cols
        cols = list(df.columns)
        trace = df[cols[1]]

        # Determines filter height threshold as 60% of the overall Standard Deviation
        sd = np.std(trace)
        threshold = sd * 0.6

        # Finds possible peaks
        peak_x, properties = list(find_peaks(trace, threshold=0.00000006, height=threshold, width=0.000000000001))
        peak_y = []
        for idx, i in enumerate(peak_x):
            val = trace[i]
            peak_y.append(val)

        if plot_me == True:
            fig, axs = plt.subplots(2, 1)
            fig.set_figwidth(20)
            fig.set_figheight(8)
            axs[0].plot(trace, color='blue')
            axs[0].set_title('Filtering')

            axs[1].set_title('Find Peaks')
            axs[1].plot(trace)
            axs[1].scatter(peak_x, peak_y, color='red')
            plt.show()

        peak_guesses = pd.DataFrame({
            'Time':peak_x,
            'Current':peak_y
        })

        return peak_guesses

    def process_file (self):

        df = self.abf_init

        df = self.apply_filter(df)
        df = self.smooth_trace(df)
        # df = self.invert_trace(df)

        return df

    def process_file_inv (self):

        df = self.abf_init

        df = self.apply_filter(df)
        df = self.smooth_trace(df)
        df = self.invert_trace(df)

        return df

    def recording_prep(self):

        df = self.abf_init

        df = self.apply_filter(df)
        df = self.smooth_trace(df)
        df = self.invert_trace(df)
        df = self.find_pot_peaks(df)

        return df

    def get_info (self):

        peak_x = self.guesses['Time']
        trace_s = self.process_file_inv()['Current_F']
        peak_y = list(self.guesses['Current'])
        peak_y = [(i*-1) for i in peak_y]

        test = peak_y

        def get_decay_feats(peak_xt, plot_me=False):
            '''
            Using the locations for each peak, extracts features for model. It will calculate when the peak returns to
            baseline and use it and the peak to calculate an exponential fit. This will be used to generate Tau and AUC.
            It will also return the time domain for ea ch peak as well as the percent baseline used to determine what
            the peak must return to to be considered baseline.

            :param peak_xt: Array of currrent values surrounding the peak in question.
            :param plot_me: When this is true, will plot the exponential fit of each peak to the calculated baseline.
            :return:
            tau: The negative inverse of the dampening coefficeint fo the exponential fit (-1/b)
            r2_decay: The r2 value for the exponential fit
            auc: Area under the curve of the fitted exponential equation. NOT the AUC for the whole peak,
            just after the peak.
            perc: The percent peak height used to find the baseline.
            '''

            # Sets starting values for determining a return to baseline
            val = 1000
            perc = 0.001

            # Finds the return to baseline index for the peak. If the initial range is unsuccessful, it will expand
            # potential trace by 10% and increase the allowed percent return to double. This will occur until a minimum is
            # found.
            y_mins = []
            while len(y_mins) == 0:
                search_area = list(trace_s[peak_xt:peak_xt + val])
                # print(search_area)
                y_mins = [(abs(x), search_area.index(x)) for x in search_area if x <= peak_y[1] * perc]
                # print(y_mins)
                perc = 2 * perc
                val = val + int(np.round(val * 1.1))
            perc = (perc / 2)
            y_min = y_mins[0]

            def exp_func(x, a, b):
                '''
                Exponential function for the fit.
                :param x: Input varibale
                :param a: Coefficient for flexibility, not indicitive (amplitude)
                :param b: Dampening coefficient
                :return: Function output.
                '''
                return a * np.exp(b * x)

            # print(y_min)

            # Grabs the appropriate portion of the trace for the exponential fit for a paticular peak.
            ydata = search_area[:y_min[1]]
            xdata = [ydata.index(x) for x in ydata]

            # Performs the exponential fit.
            initial_guess = [-0.1, -0.1]
            popt, pcov = curve_fit(exp_func, xdata, ydata, initial_guess)

            # Calculates tau
            tau = -1 / popt[1]

            # Calculates r2 for exponential fit
            xFit = np.arange(0, len(ydata), 1)
            residuals = ydata - exp_func(xFit, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
            r2_decay = 1 - (ss_res / ss_tot)

            # Fitted model for getting AUC
            def func(x):
                return popt[0] * np.exp(popt[1] * x)

            # Calculates AUC
            auc = integrate.quad(func, 0, len(ydata))[0]

            # Plots the fit and decay if plot_me is true.
            if plot_me == True:
                plt.plot(xdata, ydata, 'b-', label='Peak Data')
                plt.plot(xFit, exp_func(xFit, *popt), 'g--', label='Fitted Curve')
                plt.title(peak_xt)
                plt.legend()
                plt.show()

            return tau, r2_decay, auc, perc

        def get_start_feats(peak_xt, plot_me=False):

            # Sets starting values for determining a return to baseline
            val = 1000
            perc = 0.001

            # Finds the return to baseline index for the peak. If the initial range is unsuccessful, it will expand
            # potential trace by 10% and increase the allowed percent return to double. This will occur until a minimum is
            # found.
            y_mins = []
            while len(y_mins) == 0:
                search_area = list(trace_s[peak_xt - val:peak_xt])
                y_mins = [(abs(x), search_area.index(x)) for x in search_area if x <= peak_y[1] * perc]
                perc = 2 * perc
                val = val + int(np.round(val * 1.1))
            perc = (perc / 2)
            y_min = y_mins[len(y_mins) - 1]

            # Grabs the appropriate portion of the trace for the exponential fit for a paticular peak.
            ydata = search_area[y_min[1]:]
            xdata = [ydata.index(x) for x in ydata]

            rise_time = len(xdata)

            # Plots the fit and decay if plot_me is true.
            if plot_me == True:
                plt.plot(xdata, ydata, 'b-', label='Peak Data')
                plt.title(peak_xt)
                plt.legend()
                plt.show()

            return rise_time, perc

        # Iterates through the possible peaks and gets the necessary features
        taus = []
        r2_decays = []
        aucs = []
        iss = []
        percs = []
        rise_times = []
        rise_percs = []
        # print('Extracting features...')
        for i in range(0, len(peak_x)):
            peak = peak_x[i]
            try:
                tau, r2_decay, auc, perc = get_decay_feats(peak, plot_me=False)
                rise_time, rise_perc = get_start_feats(peak, plot_me=False)
                rise_times.append(rise_time)
                rise_percs.append(rise_perc)
                taus.append(tau)
                r2_decays.append(r2_decay)
                aucs.append(auc)
                percs.append(perc)
            except:
                print('Error returning peak ' + str(i + 1))
                iss.append(i)
                continue

        # Adjusts the dataframe size if there are errors
        peak_x = list(peak_x)
        for i in iss:
            del peak_x[i]

        # Recalls the peak heights
        peaks = trace_s[peak_x]

        # Creates the dataframe with the extracted features.
        df = pd.DataFrame(
            {
                'Time': peak_x,
                'Percent Basleine': percs,
                'Peak Heights': peaks,
                'Tau': taus,
                'r2 Decay': r2_decays,
                'AUC': aucs,
                'Rise Time': rise_times,
                'Rise Time Percent Baseline': rise_percs
            }
        )
        # print(type(trace_s))
        # print(trace_s)
        return df

    def predictions (self):

        from joblib import load
        model = load('ML_model.joblib')
        df = self.peak_info
        preds = model.predict(df)
        preds = list(preds)
        return preds

    def create_display_df (self):

        # Creates the prediction df
        df = pd.DataFrame()
        df['Time'] = list(self.peak_info['Time'])
        df['Current'] = list(self.guesses['Current'])
        df['Current'] = [(i*-1) for i in df['Current']]
        df['Prediction'] = list(self.predict)

        trace = self.processed
        trace['Time'] = trace['Time'] * 10000

        # plt.plot(trace['Time'],trace['Current_F'])

        # colors = ['r','g']
        # mask = [0,1]
        #
        # for i in range(0,len(colors)):
        #     dff = df[df['Prediction'] == mask[i]]
        #     plt.scatter(dff['Time'], dff['Current'], c=colors[i])

        return df

    def get_freq (self, plotme=False):
        '''
        Estimates the frequency per minute
        :param plotme: When true plots the frequency
        :return: Returns the dataframe with the frequency and time domain
        '''

        # Imports the traces for estimation
        trace = self.abf_init
        trace['Time Min'] = trace['Time'] / 60

        # Gets the max minute
        max_int = int(np.max(trace['Time Min']))

        # Imports the predictions and adjusts to the correct time domain
        preds = self.create_display_df()
        preds['Time Min'] = preds['Time'] / (10000 * 60)

        # Creates the bins for each minute
        freqs = []
        for i in range(1, max_int + 1):
            mx = i
            mn = (i - 1)
            filt = [t for t in preds['Time Min'] if mn < t <= mx]
            freq_m = len(filt)
            freqs.append(freq_m)

        # Creates frequency dataframe
        freq_df = pd.DataFrame({
            'Time Min': list(range(1, max_int + 1)),
            'Freq': freqs
        })

        # Plots when true
        if plotme == True:
            sns.lineplot(x='Time Min', y='Freq', data=freq_df)
            plt.show()

        return freq_df


class PeakChunk:

    def __init__(self, trace_df, time_index):
        self.time_index = time_index
        self.trace_df = trace_df
        self.chunk = self.get_chunk()
        self.drug_name = None

    def get_chunk (self, val=2000):
        peak = self.time_index

        cols = list(self.trace_df.columns)

        chunk = self.trace_df[cols[1]][(peak - val):(peak + val)]
        return chunk


class BatchAnalysis:


    def __init__(self, folder_path=None, drg_name=None):
        self.folder_path = folder_path
        if type(folder_path) != type(None):
            self.file_paths = self.get_file_paths()
        self.drg_name = drg_name

    def get_file_paths (self):
        '''
        Takes the folder and returns the path names of each .abf file within it. Otherwise returns an error.
        :return: List of path names for each .abf file within a folder
        '''

        # Ceraets list of all files at this folder
        files = os.listdir(self.folder_path)

        # Filters only the .abf files and gets path names
        paths = [os.path.join(self.folder_path,f) for f in files if '.abf' in f]

        return paths

    def iter_preds(self, drug_name=None, alt_paths=None):
        '''
        Iterates through each file in the paths, generates predictions, returns various results.
        :return: Returns various results from multiple predictions
        '''

        # Defines the paths to be iterated
        if type(alt_paths) != type(None):
            paths = alt_paths
        else:
            paths = self.file_paths

        # Iterates through the paths to generate predictions
        preds_lst = []
        for p in paths:
            rec = RecordingFile(p)
            peak_info = rec.peak_info
            peak_info.reset_index(inplace=True)
            preds = rec.create_display_df()
            preds.reset_index(inplace=True)

            peak_info = peak_info.drop(['Time'], axis=1)
            dfp = pd.concat([preds, peak_info], axis=1)

            # Gets the file name to update the dataframe in batch
            pkey = p.split('/')
            k = pkey[len(pkey)-1]

            # Adds the File Name column
            dfp['File Name'] = k
            preds_lst.append(dfp)

        # Concatenates the final data frames
        dff_preds = pd.concat(preds_lst, axis=0)
        dff_preds.drop('index',inplace=True, axis=1)

        # Adds drug name if there is one given and returns dfs. First checks the local name, then the class drug name
        if type(drug_name) != type(None):
            dff_preds['Drug Applied'] = drug_name
            return dff_preds

        elif type(self.drg_name) != type(None):
            dff_preds['Drug Applied'] = self.drg_name
            return dff_preds

        else :
            return dff_preds



