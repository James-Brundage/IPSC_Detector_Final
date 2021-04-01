"""

"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Preprocessing import preprocessing
from tqdm import tqdm
import pyabf
from scipy import signal

# df = pd.read_pickle('Subset_Labels.pkl')
abfs = '/Users/jamesbrundage/Box/Current 10k'
print('Reading Files')
cd = pd.read_excel('/Users/jamesbrundage/Box/James data (Hillary Wadsworth).xlsx')
print('Next')
files = [x for x in os.listdir(abfs) if x.find('.abf') > -1]
abf_paths = list(set(cd['ABF File']))

def batch_preproc (test):
    dfs_lst = []
    for i in tqdm(range(0,len(test))):

        test_file = test[i]

        # Verify that test file is in the directory
        if test_file in files:
            print('File is in there')
        else:
            print('Trouble finding '+ str(test_file))
            continue
        test_path = os.path.join(abfs, test_file)
        try:
            df_p = preprocessing(test_path, plot=False)
        except:
            print('Problem with '+ test_file)
            continue
        df_p['Original File'] = test_file

        dfs_lst.append(df_p)

    dff = pd.concat(dfs_lst)
    return dff

# vers = pd.read_pickle('Subset_ver_data.pkl')
vers = cd
files_v = list(set(vers['ABF File']))

# Gets subset of guesses
abf_paths = [f for f in abf_paths if f in files_v]
guesses = batch_preproc(abf_paths)

# Compares time domain of guesses and verified peaks for each thing
def label_peaks (guess, verified):

    # Gets list of file name
    files = list(set(guess['Original File']))

    # Iterates through each file to create the labels column and concatenates them
    dfs_lst = []
    for f in files:

        # Isolates data opnly for the specific ABF file in the iteration
        guess_df = guess[guess['Original File'] == f].copy()
        ver_df = verified[verified['ABF File'] == f].copy()

        # Gets the tome domain for the guesses and the verified (verified peaks are adusted to match the time domain
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

# Calls label peaks to actually create a labelled dataset
lab_ds = label_peaks(guesses, vers)

# Plots the predicted peaks vs the actual peaks
def plot_check():
    # Gets list of files in the labelled set and sorts
    fs = list(set(lab_ds['Original File']))
    fs.sort()

    for i in range(0, len(fs)):
        test_file = fs[i]
        test_df = lab_ds[lab_ds['Original File'] == test_file]

        path = os.path.join(abfs, test_file)
        abf = pyabf.ABF(path)

        # Pulls the time component (x) and the original current component (y)
        x = abf.sweepX
        y = abf.sweepY

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

        fig = plt.figure(figsize=(10,5))
        # ax = fig.add_axes()
        plt.plot(trace_s, color='blue')
        plt.scatter(test_df[test_df['Labels']==0]['Time'], test_df[test_df['Labels']==0]['Peak Heights'], color='r')
        plt.scatter(test_df[test_df['Labels']==1]['Time'], test_df[test_df['Labels']==1]['Peak Heights'], color='g')
        plt.ylim(-0.01,0.05)
        plt.title('Filtering')

        plt.show()

# Compares distributions of peaks
# sns.countplot(x='Labels', data=lab_ds)
# plt.show()












