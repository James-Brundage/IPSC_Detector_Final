"""
From the raw .abf files, creates the dataset for CNN training. WHat each unit looks like is demonstrated in the Example
jupyter notebook. I would start there.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import find_peaks
import pyabf
from tqdm import tqdm
import pickle
import os

path = '/Datasets/7_30_2019 264 Continuous Export.abf'
cd = pd.read_excel('/Users/jamesbrundage/Box/James data (Hillary Wadsworth).xlsx')

def preprocessing(path, chunk_size=2000 ,slicer=False, plot=False):
    '''
    Preprocessing steps for the IPSC spon finder. This is optimized for IPSC spons from the Yorgason Lab at BYU.
    Currently, many of the variables that could be optimized for other factors are static in the code. As time goes on
    further customization tools will be created.
    :param path: .abf path for analysis
    :param slicer: Will decide whether or not to slice the signal for analysis
    :param plot: Decides whether to plot filtering or not.
    :return: A dataframe with the extracted parameters for the model training and evaluation.
    '''
    # Reads in the apf file
    print('Reading in file: ' + str(path))
    abf = pyabf.ABF(path)

    # Pulls the time component (x) and the original current component (y)
    x = abf.sweepX
    y = abf.sweepY

    # Slicer function for testing the filter on smaller sizes
    if slicer == True:
        print('Slicing dataframe')
        def slicer(x, y, s_val=0, e_val=100000):
            x = x[s_val:e_val]
            y = y[s_val:e_val]

            return x, y

        x, y = slicer(x, y, 0, 800000)

    # Applies a butterworth filter to smooth out the signal and straighten it. Butter object is the y component only.
    print('Applying butterworth filter...')
    b, a = signal.butter(3, (0.0003, 0.07), btype='bp', analog=False)
    trace = signal.filtfilt(b, a, y)
    trace = trace * -1

    # Running average smoothing function
    print('Smoothing...')
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

    trace_s = smooth(trace)

    # Determines filter height threshold as 60% of the overall Standard Deviation
    print('Determining threshold height...')
    sd = np.std(trace_s)
    threshold = sd * 0.6

    # Finds possible peaks
    print('Finding peaks...')
    peak_x, properties = list(find_peaks(trace_s, threshold=0.00000006, height=threshold, width=0.000000000001))
    peak_y = []
    for idx, i in enumerate(peak_x):
        val = trace_s[i]
        peak_y.append(val)

    # Plots the overall filtering and peak guesses when plot is True
    if plot == True:
        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(20)
        fig.set_figheight(8)
        axs[0].plot(y * -1, color='red')
        axs[0].plot(trace, color='green')
        axs[0].plot(trace_s, color='blue')
        axs[0].set_title('Filtering')
        axs[0].legend(['Unfiltered', 'Butterworth Filter', 'Hanning Smoothing'])

        axs[1].set_title('Find Peaks')
        axs[1].plot(trace_s)
        axs[1].vlines(x=peak_x, ymin=peak_y - properties["prominences"], ymax=trace_s[peak_x], color="C1")
        axs[1].hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"],
                      color="C1")
        axs[1].scatter(peak_x, peak_y, color='red')
        plt.show()

    def get_chunk (peak, val):
        chunk = trace_s[(peak - val):(peak + val)]
        return chunk

    lst = []
    for i in range(0, len(peak_x)):
        pk = peak_x[i]
        chnk = get_chunk(pk, chunk_size)
        if len(chnk) != 2 * chunk_size:
            continue
        lst.append(chnk)

    dff = pd.DataFrame(lst)
    dff['Time'] = peak_x
    print(dff)

    return dff
    # def get_decay_feats(peak_xt, plot_me=False):
    #     '''
    #     Using the locations for each peak, extracts features for model. It will calculate when the peak returns to
    #     baseline and use it and the peak to calculate an exponential fit. This will be used to generate Tau and AUC.
    #     It will also return the time domain for ea ch peak as well as the percent baseline used to determine what
    #     the peak must return to to be considered baseline.
    #
    #     :param peak_xt: Array of currrent values surrounding the peak in question.
    #     :param plot_me: When this is true, will plot the exponential fit of each peak to the calculated baseline.
    #     :return:
    #     tau: The negative inverse of the dampening coefficeint fo the exponential fit (-1/b)
    #     r2_decay: The r2 value for the exponential fit
    #     auc: Area under the curve of the fitted exponential equation. NOT the AUC for the whole peak,
    #     just after the peak.
    #     perc: The percent peak height used to find the baseline.
    #     '''
    #
    #     # Sets starting values for determining a return to baseline
    #     val = 1000
    #     perc = 0.001
    #
    #     # Finds the return to baseline index for the peak. If the initial range is unsuccessful, it will expand
    #     # potential trace by 10% and increase the allowed percent return to double. This will occur until a minimum is
    #     # found.
    #     y_mins = []
    #     while len(y_mins) == 0:
    #         search_area = list(trace_s[peak_xt:peak_xt + val])
    #         y_mins = [(abs(x), search_area.index(x)) for x in search_area if x <= peak_y[1] * perc]
    #         perc = 2 * perc
    #         val = val + int(np.round(val * 1.1))
    #     perc = (perc / 2)
    #     y_min = y_mins[0]
    #
    #     def exp_func(x, a, b):
    #         '''
    #         Exponential function for the fit.
    #         :param x: Input varibale
    #         :param a: Coefficient for flexibility, not indiciative (amplitude)
    #         :param b: Dampening coefficient
    #         :return: Function output.
    #         '''
    #         return a * np.exp(b * x)
    #
    #     # Grabs the appropriate portion of the trace for the exponential fit for a paticular peak.
    #     ydata = search_area[:y_min[1]]
    #     xdata = [ydata.index(x) for x in ydata]
    #
    #     # Performs the exponential fit.
    #     initial_guess = [-0.1, -0.1]
    #     popt, pcov = curve_fit(exp_func, xdata, ydata, initial_guess)
    #
    #     # Calculates tau
    #     tau = -1 / popt[1]
    #
    #     # Calculates r2 for exponential fit
    #     xFit = np.arange(0, len(ydata), 1)
    #     residuals = ydata - exp_func(xFit, *popt)
    #     ss_res = np.sum(residuals ** 2)
    #     ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    #     r2_decay = 1 - (ss_res / ss_tot)
    #
    #     # Fitted model for getting AUC
    #     def func(x):
    #         return popt[0] * np.exp(popt[1] * x)
    #
    #     # Calculates AUC
    #     auc = integrate.quad(func, 0, len(ydata))[0]
    #
    #     # Plots the fit and decay if plot_me is true.
    #     if plot_me == True:
    #         plt.plot(xdata, ydata, 'b-', label='Peak Data')
    #         plt.plot(xFit, exp_func(xFit, *popt), 'g--', label='Fitted Curve')
    #         plt.title(peak_xt)
    #         plt.legend()
    #         plt.show()
    #
    #     return tau, r2_decay, auc, perc
    #
    #
    # def get_start_feats(peak_xt, plot_me=True):
    #
    #     # Sets starting values for determining a return to baseline
    #     val = 1000
    #     perc = 0.001
    #
    #     # Finds the return to baseline index for the peak. If the initial range is unsuccessful, it will expand
    #     # potential trace by 10% and increase the allowed percent return to double. This will occur until a minimum is
    #     # found.
    #     y_mins = []
    #     while len(y_mins) == 0:
    #         search_area = list(trace_s[peak_xt-val:peak_xt])
    #         y_mins = [(abs(x), search_area.index(x)) for x in search_area if x <= peak_y[1] * perc]
    #         perc = 2 * perc
    #         val = val + int(np.round(val * 1.1))
    #     perc = (perc / 2)
    #     y_min = y_mins[len(y_mins)-1]
    #
    #     # Grabs the appropriate portion of the trace for the exponential fit for a paticular peak.
    #     ydata = search_area[y_min[1]:]
    #     xdata = [ydata.index(x) for x in ydata]
    #
    #     rise_time = len(xdata)
    #
    #     # Plots the fit and decay if plot_me is true.
    #     if plot_me == True:
    #         plt.plot(xdata, ydata, 'b-', label='Peak Data')
    #         plt.title(peak_xt)
    #         plt.legend()
    #         plt.show()
    #
    #     return rise_time, perc
    #
    # # Iterates through the possible peaks and gets the necessary features
    # taus = []
    # r2_decays = []
    # aucs = []
    # iss = []
    # percs = []
    # rise_times = []
    # rise_percs = []
    # print('Extracting features...')
    # for i in range(0, len(peak_x)):
    #     peak = peak_x[i]
    #     try:
    #         tau, r2_decay, auc, perc = get_decay_feats(peak, plot_me=False)
    #         rise_time, rise_perc = get_start_feats(peak, plot_me=False)
    #         rise_times.append(rise_time)
    #         rise_percs.append(rise_perc)
    #         taus.append(tau)
    #         r2_decays.append(r2_decay)
    #         aucs.append(auc)
    #         percs.append(perc)
    #     except:
    #         print('Error returning peak ' + str(i + 1))
    #         iss.append(i)
    #         continue
    #
    # # Adjusts the dataframe size if there are errors
    # peak_x = list(peak_x)
    # for i in iss:
    #     del peak_x[i]
    #
    # # Recalls the peak heights
    # peaks = trace_s[peak_x]
    #
    # # Creates the dataframe with the extracted features.
    # df = pd.DataFrame(
    #     {
    #         'Time': peak_x,
    #         'Percent Basleine': percs,
    #         'Peak Heights': peaks,
    #         'Tau': taus,
    #         'r2 Decay': r2_decays,
    #         'AUC': aucs,
    #         'Rise Time': rise_times,
    #         'Rise Time Percent Baseline':rise_percs
    #     }
    # )
    #
    # return df

def batch_preproc(abfs = '/Users/jamesbrundage/Box/Current 10k', cd = pd.read_excel('/Users/jamesbrundage/Box/James data (Hillary Wadsworth).xlsx')):

    abf_paths = list(set(cd['ABF File']))

    dfs = []
    for i in tqdm(range(0, len(abf_paths))):

        file = abf_paths[i]

        path = os.path.join(abfs, file)
        try:
            df_p = preprocessing(path)
        except:
            print('Problem with ' + file)
            continue
        df_p['Original File'] = file
        dfs.append(df_p)

    df = pd.concat(dfs)
    print(df)
    return df

def label_peaks (guess, verified):

    # Gets list of file name
    files = list(set(guess['Original File']))

    # Iterates through each file to create the labels column and concatenates them
    dfs_lst = []
    for f in tqdm(files):

        # Isolates data opnly for the specific ABF file in the iteration
        guess_df = guess[guess['Original File'] == f].copy()
        ver_df = verified[verified['ABF File'] == f].copy()

        # Gets the tome domain for the guesses and the verified (verified peaks are adjusted to match the time domain
        # in
        gt = guess_df['Time']
        vt = ver_df['Time (ms)'] * 10

        # Iterates through each verified peak finding the guessed peak closest to it and creates labels
        ver_gs = []
        for t in vt:

            # Finds the guessed peak closest to each verified peak and adds to ver_gs
            given_value = t
            absolute_difference_function = lambda list_value: abs(list_value - given_value)
            closest_value = min(gt, key=absolute_difference_function)
            ver_gs.append(closest_value)

        # Creates dataframe with the verified time and closest guess for comparison of the time domains
        dff = pd.DataFrame({
            'Verified Time':vt,
            'Closest Guess':ver_gs
        })

        # Actually creates the labels column from the verified guesses (ver_gs) and adds to dfs_lst
        def assign_mask (x):
            '''
            Creates the labels column using the verified guesses (ver_gs)
            :param x:
            :return:
            '''
            if x in ver_gs:
                return 1
            else :
                return 0
        guess_df['Labels'] = guess_df['Time'].apply(assign_mask).copy()
        dfs_lst.append(guess_df)

    # Finalizes returns
    dff = pd.concat(dfs_lst)
    dff.to_pickle('Labels_mac.pkl')
    return dff

df = batch_preproc(cd=cd)
df = label_peaks(df, cd)
print(df)
print(df.head())
df.to_csv('CNN_Labels.csv')