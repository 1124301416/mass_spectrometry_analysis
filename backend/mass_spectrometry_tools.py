import csv
import re
import warnings
from os import listdir, path, rename
from statistics import mean, stdev
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import savetxt
from scipy import integrate
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

def linear_plateau(m, a, x):
    return float(m) * (1 - np.exp(-float(a) * x))


def is_float(obj):
    try:
        float(obj)
        return True
    except ValueError:
        return False

def integration(x, y):
    # interpolation
    fn = UnivariateSpline(x, y, k=1)

    results = quad(fn, x[0], x[-1])[0]
    return results

def to_number(obj):
    try:
        return int(obj)
    except ValueError:
        return float(obj)


def least_square(param, *args):
    m, a = param[0], param[1]
    x, obs = args[0], args[1]
    pred = linear_plateau(m, a, x)
    print(m, a, np.sum((obs - pred) ** 2))
    return np.sum((obs - pred) ** 2)


def rename_file(name: str):
    """
    Rename file if concentration is not followed with a space
    :param name:
    :return: rename
    """
    reached_number = False
    for i in range(len(name)):
        if name[i].isdigit():
            reached_number = True
        if reached_number:
            if name[i] == ' ':
                return name
            elif name[i].isdigit():
                pass
            else:
                return name[:i] + ' ' + name[i:]


## https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def refactor_name(abs_path):
    dir_old = listdir(abs_path)
    dir = [rename_file(name) for name in listdir(abs_path)]
    ## rename file for further analysis if needed
    names = [path.join(abs_path, name) for name in dir]
    names_old = [path.join(abs_path, name) for name in dir_old]
    for i in range(len(names)):
        rename(names_old[i], names[i])
    names.sort(key=natural_keys)
    return names


def get_linear_plateau_parameters(x, y, initial_guess=np.array([16, 0.001])):
    res = minimize(least_square, initial_guess, args=(x, y))
    print(res.x[0], res.x[1])
    return res.x[0], res.x[1]


class MassSpectrometryTools:
    def __init__(self, filenames: List[str], polynomial_degree: int = 6, initial_guess=np.array([16, 0.001])):
        """
        Analyze the sample_data by give names
        Must have "# ?M" or "# ?L" in filenames. Do not use full path
        :param filenames: file names
        :param polynomial_degree: fitting degree
        :param initial_guess: initial guess for plateau approximation.
        """
        self.filenames = filenames
        self.polynomial_degree = polynomial_degree
        self.legends = self.get_concentration_legends(self.filenames)
        self.time = None
        self.signal = None
        self.index = None
        self.ratio = None
        self.raw_signal = None
        self.polynomial_fitting_function = None
        self.initial_guess = initial_guess
        self.STARTING_INDEX = 32
        self.load_signal()

    def load_signal(self):
        """
        load the signal and time data, with the index to cut.
        crop the data to the uniform minimum size
        :return: None
        """
        self.generate_trendlines(plot=False)
        # crop signal
        time_lengths = [len(time) for time in self.time]
        signal_lengths = [len(signal) for signal in self.signal]
        minimum = min(time_lengths + signal_lengths)
        self.signal = [s[:minimum] for s in self.signal]
        self.time = [t[:minimum] for t in self.time]
        self.raw_signal = [s[:minimum] for s in self.raw_signal]

    def is_starting_index(self, data_points: List[float], base_line: float, stdev: float):
        """
        :param data_points:
        :param base_line:
        :param stdev:
        :return: true if it is the starting index
        """
        minimum = min(data_points)
        if minimum - base_line > stdev:
            return True
        else:
            return False

    def get_concentration_legends(self, filenames: List[str]):
        """
        Get the legends for plotting. Finds with M
        :param filenames: names of file
        :return: legends
        """
        concentration_names = []
        for names in filenames:
            end = int(names.find('M'))
            if end == -1:
                end = int(names.find('L'))
            start = end
            while start > 0:
                if names[start] == '/':
                    break
                start -= 1
            concentration_names.append(names[start + 1:end ])
        return concentration_names

    def subtract_value(self, list, value):
        # modify the list by subtracting everything
        for i in range(len(list)):
            list[i] -= value

    def get_a_trendline(self, time: list, signal: list, plot: bool = False):
        """
        :param time:
        :param signal:
        :param plot:
        :return:
        """
        ## Takes a time sample_data and signal sample_data to visualize a trendline for the sample_data
        ## returns the polynomial function and a sample_data starting point

        noise_signal = signal[:self.STARTING_INDEX]
        noise_average = mean(noise_signal)
        noise_standard_deviation = stdev(noise_signal)

        while not self.is_starting_index(signal[self.STARTING_INDEX:], noise_average, noise_standard_deviation):
            self.STARTING_INDEX += 1
        time = [x for x in time[self.STARTING_INDEX:] if str(x) != 'nan']
        signal = [x for x in signal[self.STARTING_INDEX:] if str(x) != 'nan']
        z = np.polyfit(time, signal, self.polynomial_degree)
        p = np.poly1d(z)
        if plot:
            plt.plot(time[self.STARTING_INDEX:], signal[self.STARTING_INDEX:],
                     time[self.STARTING_INDEX:], p(time[self.STARTING_INDEX:]))
            plt.show()
        return p, self.STARTING_INDEX

    def plot_ratio_raw_data(self, time_range=None, ratio_range=None):

        min_signals = self.raw_signal[0]

        for i in range(len(self.raw_signal)):
            ratio_signal = []
            for j in range(len(self.raw_signal[i])):
                signal = abs(self.raw_signal[i][j])
                try:
                    signal = abs(float(signal) / float(min_signals[j]))
                except ZeroDivisionError:
                    k = 1
                    while min_signals[j - k] == 0:
                        k -= 1
                    signal = signal / min_signals[j - k]

                ratio_signal.append(signal)
            avg = mean(ratio_signal)
            std = stdev(ratio_signal)
            z = np.polyfit(self.time[i], ratio_signal, self.polynomial_degree)
            p = np.poly1d(z)
            plt.plot(self.time[i], ratio_signal)

        if self.legends is not None:
            plt.legend(self.legends, loc='upper center', ncol=int(len(self.filenames) / 2))
        if time_range is not None:
            plt.xlim(time_range)
        if ratio_range is not None:
            plt.ylim(ratio_range)
        plt.xlabel('Time (min)')
        plt.ylabel('Ratio')
        plt.show()

    def generate_trendlines(self, plot: bool = True, normalize_all=False):

        time_data = []
        signal_data = []
        raw_signal_data = []
        max_cut_index = 0
        for name in self.filenames:
            db = pd.read_csv(name, header=None, sep=None, encoding='utf-16', engine='python')
            time = list(db.iloc[:, 0])
            signal = list(db.iloc[:, 1])

            p, index = self.get_a_trendline(time, signal)

            time = [t - time[index] for t in time[index:]]

            raw_signal = signal[:len(time)]
            signal = [(s - p(time)[0]) for s in p(time)]
            if normalize_all:
                signal = [s / max(signal) for s in signal]
            time_data.append(time)
            signal_data.append(signal)
            raw_signal_data.append(raw_signal)
            if index > max_cut_index:
                max_cut_index = index
            if plot:
                plt.plot(time, signal)
        if plot:
            plt.xlabel('Time (min)')
            plt.ylabel('Signal (A.U.)')
            if self.legends is not None:
                plt.legend(self.legends)
            plt.show()
        self.time, self.signal, self.index, self.raw_signal = time_data, signal_data, max_cut_index, raw_signal_data

    def plot_ratio(self):
        if self.time is None or max(self.signal) < 2:
            self.generate_trendlines(plot=False)

        assert len(self.time) == len(self.signal)
        signal_copy = self.signal
        max_point = None
        min_index = 0
        i = 0
        for signal in self.signal:
            if max_point is not None:
                if max(signal) < max_point:
                    max_point = max(signal)
                    min_index = i

            else:
                max_point = max(signal)
            i += 1

        data_amount = len(self.signal[min_index])
        i = 0
        for time in self.time:
            self.time[i] = time[:data_amount]
            self.time[i] = time[:-self.index]
            i += 1
        for i in range(len(self.signal)):
            self.signal[i] = self.signal[i][:data_amount]
        for i in range(len(self.signal)):
            if i is not min_index:
                self.signal[i] = [self.signal[i][j] / self.signal[min_index][j] for j in range(len(self.signal[i]))]
        self.signal[min_index] = [1 for i in range(len(self.signal[min_index]))]
        self.ratio = self.signal
        self.signal = signal_copy
        for i in range(len(self.signal)):
            self.signal[i] = self.signal[i][:-self.index]
        for i in range(len(self.time)):
            min_len = min(len(self.time[i]), len(self.signal[i]))
            plt.plot(self.time[i][:min_len], self.signal[i][:min_len])
        if self.legends is not None:
            plt.legend(self.legends, loc='upper center', ncol=int(len(self.filenames) / 2))
        plt.xlabel('Time (min)')
        plt.ylabel('Ratio')
        plt.show()

    def plot_concentration_vs_ratio(self, time_point: float = 0):
        if self.time is None or max(self.signal) < 2:
            self.plot_ratio()
        x = []
        y = []
        for name in self.legends:
            x.append([to_number(s) for s in name.split()
                      if is_float(s) or s.isdigit()][0])  # Find numbers/floats in legend
        time_index = 0
        for time in self.time:
            if time_point >= self.time[0][time_index]:
                break
            time_index += 1

        for s in self.ratio:
            if time_index == 0:
                y.append(s[time_index + 1])
            else:
                y.append(s[time_index + 1])
        plt.scatter(x, y)

        m, a = get_linear_plateau_parameters(np.array(x), np.array(y), initial_guess=self.initial_guess)
        trend_x = np.linspace(min(x), max(x), 100)
        trend_y = linear_plateau(m, a, trend_x)
        plt.plot(trend_x, trend_y)
        plt.legend(['y = M(1-exp(ax))', 'sample_data'])
        plt.xlabel('Concentration (' + self.legends[0][-2:] + ")")
        plt.ylabel('Ratio')
        plt.show()

    def plot_concentration_vs_mass_intensity(self, time_point: float = 0):
        if self.time is None or max(self.signal) < 2:
            self.generate_trendlines(plot=False)
        x = []
        y = []
        for name in self.legends:
            x.append([to_number(s) for s in name.split()
                      if is_float(s) or s.isdigit()][0])  # Find numbers/floats in legend
        time_index = 0
        for time in self.time:
            if time_point >= self.time[0][time_index]:
                break
            time_index += 1

        for s in self.signal:
            if time_index == 0:
                y.append(s[time_index + 1])
            else:
                y.append(s[time_index + 1])
        plt.scatter(x, y)

        m, a = get_linear_plateau_parameters(np.array(x), np.array(y), initial_guess=self.initial_guess)
        trend_x = np.linspace(min(x), max(x), 100)
        trend_y = linear_plateau(m, a, trend_x)
        plt.plot(trend_x, trend_y)
        plt.xlabel('Concentration (' + self.legends[0][-2:] + ")")
        plt.ylabel('Mass Intensity')
        plt.show()

    def integration(self, integration_range=None, is_poly: bool = False ):
        """
        Integrate the fitted signal
        :param range: range of integration
        :return: integration of the given range of interest
        """
        results = []
        for i in range(len(self.time)):
            if is_poly == True:
                z = np.polyfit(self.time[i], self.raw_signal[i], self.polynomial_degree)
                p = np.poly1d(z)
            else:
                p = UnivariateSpline(self.time[i], self.raw_signal[i], k=1)

            if integration_range is None:
                result, uncertainty = integrate.quad(p, self.time[i][0], self.time[i][-1])[0], \
                                      integrate.quad(p, self.time[i][0], self.time[i][-1])[1]
                print('The data of', self.legends[i], 'has an integration of', result, 'in full range')
            else:
                result, uncertainty = integrate.quad(p, integration_range[0], integration_range[1])[0], \
                                      integrate.quad(p, integration_range[0], integration_range[1])[1],
                print('The data of', self.legends[i], 'has an integration of', result,
                      ' in range from', integration_range[0], 'min to', integration_range[1], 'min.')
            number = [int(s) for s in self.legends[i].split() if s.isdigit()][0]
            results.append([float(number),float(result)])
        savetxt('results.csv',results,delimiter=',')