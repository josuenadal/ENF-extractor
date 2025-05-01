import os
import sys
import subprocess
import math
import time
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.widgets import Button, Slider, RangeSlider
from scipy.signal import butter, filtfilt, sosfilt, resample, savgol_filter
from scipy.interpolate import make_smoothing_spline
from audio2numpy import open_audio
import pylab
from datetime import datetime

def extract_audio(video_path, output_format="mp3"):
    """Extract audio from video."""
    extracted_audio_folder = "./audio/"
    try:
        if os.path.exists(extracted_audio_folder) == False:
            os.mkdir(extracted_audio_folder)

        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = f"{extracted_audio_folder}{filename}.{output_format}"

        if os.path.exists(output_file):
            print(f"Will not extract audio since audio file already exists, will use that one.")
            return output_file
        
        subprocess.call(["ffmpeg", "-y", "-i", video_path, output_file], 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        
        return output_file
    except Exception as e:
        raise Exception("Error when extracting audio", e)

def audio_to_numpy(audio_file):
    signal, sampling_rate = open_audio(audio_file)
    return (signal, sampling_rate)

@dataclass
class Signal:
    """Class for signals"""
    signal: list[float]
    sample_rate: int
    sample_count: int
    duration: float
    is_stereo: bool
    # If signal is stereo, pick a channel
    L_channel: list[float]
    R_channel: list[float]
    # Filtered
    Filtered_signal: list[float]
    Filtered_L_channel: list[float]
    Filtered_R_channel: list[float]

    def __init__(self, sig_tup):
        signal = sig_tup[0]
        sample_rate = sig_tup[1]

        self.Filtered_signal = signal
        if type(signal[0]) is np.ndarray:
            self.signal = map(lambda x: x.item(), signal)
        else:
            self.signal = signal
        self.sample_rate = sample_rate
        self.sample_count = len(signal)
        self.duration = self.sample_count/self.sample_rate
        self.is_stereo = False
        self.L_channel = []
        self.R_channel = []
        if type(signal[0]) is np.ndarray:
            self.is_stereo = True
            for s in signal:
                self.L_channel.append(s[0].item())
                self.R_channel.append(s[1].item())
        self.Filtered_L_channel = self.L_channel
        self.Filtered_R_channel = self.R_channel

    def create_butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def  butter_bandpass_filter(self, low, high, order=5):
        sos = self.create_butter_bandpass(low, high, self.sample_rate, order=order)
        # Apply the filter
        if self.is_stereo:
            self.Filtered_L_channel = sosfilt(sos,  self.Filtered_L_channel)
            self.Filtered_R_channel = sosfilt(sos,  self.Filtered_R_channel)
        else:
            self.Filtered_signal = sosfilt(sos, self.Filtered_signal)

    def resample(self, num):
        print()
        # if self.is_stereo:
        #             resample(self.Filtered_L_channel, 120)
        #             self.Filtered_L_channel = sosfilt(sos, self.Filtered_L_channel)
        #             self.Filtered_R_channel = sosfilt(sos, self.Filtered_R_channel)
        #         else:
        #             self.Filtered_signal = sosfilt(sos, self.Filtered_signal)
        
class FrequencySelector:
    """FrequencySelector"""
    # Properties
    default_window_length = 250000

    signal = plot_title = None
    figure = spectogram_Ax = colorbar = None
    y0 = y1 = x0 = x1 = None
    spec_data = spec_time = spec_frequencies = spec_image = None

    slider_Ax_1 = slider_Ax_2 = slider_Ax_3 = slider_Ax_3 = None
    NFFT_slider = noverlap_slider = padto_slider = vminmax_slider = None

    graph_Ax = None

    reset_Btn_Ax = reset_Btn = plot_Btn_Ax = plot_Btn = None
    vmin = vmax = ENF_x_points = ENF_y_points = None
    spec_ylim = (None, None)

    # Color bar init values
    vmin_init = 500
    vmax_init = 0

    y_mean = None

    resolution_Ax = resolution_Test = None

    resample = True
    Filter = True
    target_frequency = 58

    vmin = -241
    vmax = 0

    # Visible range
    # spec_ylim = (55,61)
    # spec_ylim = (57,59)
    # spec_ylim = (119,121)

    NFFT = 30000
    noverlap = 29000
    pad_to = 650000

    cmap = pylab.get_cmap('Greys')
    scale_by_freq = True

    cmap = pylab.get_cmap('Greys')

    def reset_widgets(self, event):
        self.NFFT_slider.reset()
        self.padto_slider.reset()
        self.noverlap_slider.reset()
        self.vminmax_slider.reset()
        self.plot_spectogram(None)

    def get_vmin_vmax(self):
        if (self.vmin is None) or (self.vmax is None):
            return (self.vmin_init, self.vmax_init)
        else:
            return (self.vmin, self.vmax)
    
    def create_layout(self):
        self.figure = plt.figure(figsize=(10,5))
        self.spectogram_Ax = plt.axes((0.085, 0.3, 1, 0.6))

        slider_max = int(self.signal.sample_rate*10)

        self.slider_Ax_1 = plt.axes((0.085,0.225,0.8,0.02))
        self.NFFT_slider = Slider(
            ax = self.slider_Ax_1,
            label = 'NFFT',
            valmin = 0,
            valmax = slider_max,
            valinit = self.NFFT,
            valfmt="%i"
        )

        self.slider_Ax_2 = plt.axes((0.085,0.2,0.8,0.02))
        self.noverlap_slider = Slider(
            ax = self.slider_Ax_2,
            label = 'noverlap',
            valmin = 0,
            valmax = slider_max,
            valinit = self.noverlap,
            valfmt="%i"
        )

        self.slider_Ax_3 = plt.axes((0.085,0.175,0.8,0.02))
        self.padto_slider = Slider(
            ax = self.slider_Ax_3,
            label = 'pad_to',
            valmin = 0,
            valmax =  slider_max*4,
            valinit = self.pad_to,
            valfmt="%i"
        )

        self.slider_Ax_4 = plt.axes((0.085,0.15,0.8,0.02))
        self.vminmax_slider = RangeSlider(
            ax = self.slider_Ax_4,
            label = 'vmin_vmax',
            valmin = -550,
            valmax = 0,
            valinit = self.get_vmin_vmax(),
            valfmt="%i"
        )

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        self.reset_Btn_Ax = plt.axes((0.085,0.1,0.390,0.04))
        self.reset_Btn = Button(self.reset_Btn_Ax, 'Reset', hovercolor='0.975')

        self.plot_Btn_Ax = plt.axes((0.495,0.1,0.390,0.04))
        self.plot_Btn = Button(self.plot_Btn_Ax, 'Plot All', hovercolor='0.975')
        
        self.NFFT_slider.on_changed(self.redraw_view)
        self.padto_slider.on_changed(self.redraw_view)
        self.noverlap_slider.on_changed(self.redraw_view)
        self.vminmax_slider.on_changed(self.set_spec_cb_lims)

        self.reset_Btn.on_clicked(self.reset_widgets)
        self.plot_Btn.on_clicked(self.plot_chunk_lines)

        cax, kw = matplotlib.colorbar.make_axes(self.spectogram_Ax, orienation="vertical")
        norm = matplotlib.colors.Normalize(vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])
        self.colorbar = matplotlib.colorbar.ColorbarBase(cax, cmap=self.cmap, norm=norm)

        self.spectogram_Ax.set_title(self.plot_title)
        self.spectogram_Ax.set_xlabel("Time (s)")
        self.spectogram_Ax.set_ylabel('Frequency (Hz)')
        self.spectogram_Ax.set_xlim(0, self.signal.duration)

        self.set_axis_ylim()

    def plot_spectogram(self, val=None):

        if self.signal.is_stereo:
            data = self.signal.Filtered_L_channel
        else:
            data = self.signal.Filtered_signal
        
        self.spec_data, self.spec_frequencies, self.spec_time, self.spec_image = self.spectogram_Ax.specgram(data[:self.default_window_length], 
                NFFT=int(self.NFFT_slider.val), 
                noverlap=int(self.noverlap_slider.val), 
                Fs=self.signal.sample_rate, 
                pad_to=int(self.padto_slider.val), cmap=self.cmap,
                vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])

        self.spectogram_Ax.set_title(self.plot_title)
        self.spectogram_Ax.set_xlabel("Time (s)")
        self.spectogram_Ax.set_ylabel('Frequency (Hz)')

    def redraw_view(self, event=None):
        self.save_viewlims()
        self.redraw_spectogram()

    def redraw_spectogram(self, event=None):
        self.spectogram_Ax.clear()

        self.plot_spectogram()

        self.spectogram_Ax.set_ylim(self.y0,self.y1)
        self.spectogram_Ax.set_xlim(self.x0,self.x1)

        self.ENF_y_points = []
        self.ENF_x_points = []

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def set_spec_cb_lims(self, val):
        for i in self.spectogram_Ax.images:
            i.set_clim(self.vminmax_slider.val[0], self.vminmax_slider.val[1])
            self.colorbar.update_normal(i)
    
    def get_min_w_index(self, arr):
        min_val = 0
        max_freq_index = 0
        for i in range(len(arr)):
            if self.y_mean is None:
                if (arr[i] < min_val):
                    min_val = arr[i]
                    max_freq_index = i
            elif (arr[i] < min_val) and (arr[i] < (self.y_mean + 0.6)) and (arr[i] > (self.y_mean - 0.6)):
                min_val = arr[i]
                max_freq_index = i
        return (min_val, max_freq_index)

    def has_acceptable_deviation(self, prev_freq, curr_freq):
        dev = 0.2
        if prev_freq is not None:
            diff = abs(prev_freq - curr_freq)
            if (diff <= dev):
                return True
            else:
                return False
        return True

    def get_max_w_index(self, time_slice, wndw_min_freq_index):
        """ Get the highest frequency and its index from a single slice of time of the spectogram. """
        max_val = -1
        max_val_index = -1
        prev_freq = None
        if len(self.ENF_y_points) > 0:
            prev_freq = self.ENF_y_points[-1]
        for i in range(len(time_slice)):
            
            curr_freq = self.spec_frequencies[wndw_min_freq_index + i]

            if (time_slice[i] > max_val) and self.has_acceptable_deviation(prev_freq, curr_freq):
                max_val = time_slice[i]
                max_val_index = i

        return (max_val, max_val_index)
    
    def trace_ENF(self, start_index):
        """ Plot a line composed of the largest frequencies """
        start_sec = 0
        if start_index > 0:
            start_sec = start_index/self.signal.sample_rate

        y0, y1 = self.spectogram_Ax.get_ylim()
        x0, x1 = self.spectogram_Ax.get_xlim()
        
        freq_indexes = []
        for fi in range(len(self.spec_frequencies)):
            if (self.spec_frequencies[fi] >= y0) and (self.spec_frequencies[fi] <= y1):
                freq_indexes.append(fi)

        curr_x_points = []
        curr_y_points = []
        spec_x_points = []
        
        # The index of the lowest frequency in the window, all checked frequencies will be above this one.
        window_bottom_freq_index = min(freq_indexes)
        window_top_freq_index = max(freq_indexes)

        for ti in range(len(self.spec_time)):
            current_time = self.spec_time[ti]
            if (current_time >= x0) and (current_time <= x1):

                max_freq_index = self.get_max_w_index(
                                self.spec_data[ window_bottom_freq_index : window_top_freq_index, ti],
                                window_bottom_freq_index)[1]

                max_frequency = self.spec_frequencies[window_bottom_freq_index + max_freq_index]

                curr_y_points.append(max_frequency)
                self.ENF_y_points.append(max_frequency)

                curr_x_points.append(current_time + start_sec)
                self.ENF_x_points.append(current_time + start_sec)

                spec_x_points.append(current_time)
                print(f"({current_time + start_sec}s, {max_frequency}Hz)")
        
        self.spectogram_Ax.plot(spec_x_points, curr_y_points, 'r-')

        self.graph_Ax.clear()
        self.graph_Ax.plot(self.ENF_x_points, self.ENF_y_points, 'r-', label="raw")
        self.graph_Ax.plot(self.ENF_x_points, savgol_filter(self.ENF_y_points, 20, 2), "g-", label="smoothed")

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        time.sleep(20)

    def create_top_and_bottom_layout(self):
        if self.graph_Ax is not None:
            self.graph_Ax.remove()
        self.graph_Ax = plt.axes((0.085, 0.3, 0.8, 0.3))
        self.graph_Ax.grid()
        self.graph_Ax.set_xlabel("Time (s)")

        self.spectogram_Ax.remove()
        self.spectogram_Ax = plt.axes((0.085, 0.6, 0.8, 0.3))
        self.spectogram_Ax.set_xlabel("")
        self.spectogram_Ax.set_title(f"Sample for {self.plot_title}")

    def save_viewlims(self):
        self.y0, self.y1 = self.spectogram_Ax.get_ylim()
        self.x0, self.x1 = self.spectogram_Ax.get_xlim()

    def plot_chunk_lines(self, event):

        self.save_viewlims()

        self.create_top_and_bottom_layout()

        self.redraw_spectogram(None)

        # x overlap acounts for the shrinkage during the plotting of the spectogram.
        # x overlap makes sure that no points are skipped.
        x0_overlap = int((self.spec_time[0]*self.signal.sample_rate)-0)
        graph_end_point_index = int(((self.default_window_length/self.signal.sample_rate)-(self.spec_time[-1])) * self.signal.sample_rate)
        x1_overlap = x0_overlap + graph_end_point_index
        
        print(f"overlaps {x0_overlap}, {x1_overlap}")
        
        if self.signal.is_stereo:
            # Choose left channel (arbitrary)
            data = self.signal.Filtered_L_channel
        else:
            data = self.signal.Filtered_signal
        
        # Starting steps for chunk calculation
        x0 = 0
        x1 = self.default_window_length

        # Get initial y values and keep them for the duration of the plotting.        
        y0, y1 = self.spectogram_Ax.get_ylim()
        print(f"ylims {(y0,y1)}")

        # Clear plot array.
        # This is what will be output as CSV.
        self.ENF_y_points = []
        self.ENF_x_points = []
        while (x1 < self.signal.sample_count):

            # Calculate spectogram in current window.
            print(f"Moving window to {x0/self.signal.sample_rate}s, {x1/self.signal.sample_rate}s")
            self.spectogram_Ax.clear()
            self.spectogram_Ax.set_title(f"Sample for {self.plot_title}")
            self.spec_data, self.spec_frequencies, self.spec_time, self.spec_image = self.spectogram_Ax.specgram(data[x0:x1],
                NFFT=int(self.NFFT_slider.val),
                noverlap=int(self.noverlap_slider.val),
                Fs=self.signal.sample_rate,
                pad_to=int(self.padto_slider.val), cmap=self.cmap,
                vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1]
            )
            
            self.spectogram_Ax.plot(self.spectogram_Ax.get_xlim(),[self.y_mean,self.y_mean],'b-')
            # Refresh image.
            self.spectogram_Ax.set_ylim(y0, y1)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.spectogram_Ax.set_ylim(y0, y1)
            
            # Plot the line.
            self.trace_ENF(x0)

            if self.y_mean is None:
                self.y_mean = np.mean(self.ENF_y_points)

            # Move window.
            x0 = x1 - x1_overlap
            x1 = x1 + self.default_window_length - x1_overlap
        
        # When done create "save" button and refresh image.
        self.save_enf_plot_ax = plt.axes((0.495,0.05,0.390,0.04))
        self.save_enf_plot_Btn = Button(self.save_enf_plot_ax, 'Save ENF plot', hovercolor='0.975')
        self.save_enf_plot_Btn.on_clicked(self.save_enf_plot)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def resample_signal(self):
        if self.resample_signal:
            self.signal.resample(120)
    
    def filter_signal(self):
        if self.target_frequency is not None:
            self.signal.butter_bandpass_filter(self.target_frequency-1, self.target_frequency+1, 5)
    
    def set_plot_title(self, title):
        self.plot_title = title
        self.spectogram_Ax.set_title(f"Sample for {self.plot_title}")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save_enf_plot(self, event):
        if (self.ENF_x_points is None) or (self.ENF_y_points is None): 
            print("No data to print.")
        date = datetime.today().strftime('%Y-%m-%d')
        fname = f"{os.path.splitext(os.path.basename(self.plot_title))[0]}_{date}.csv"
        # Smooth
        self.ENF_y_points = savgol_filter(self.ENF_y_points, 20, 2)
        with open(fname, "w") as f:
            for i in range(len(self.ENF_x_points)):
                f.write(f"{self.ENF_x_points[i].item()}, {self.ENF_y_points[i].item()}\n")
        print(f"printed to {fname}")

    def set_axis_ylim(self):
        """ Set the y limits to the init values """
        if self.target_frequency is None:
            self.spectogram_Ax.set_ylim(auto=True)
        else:
            self.spec_ylim = (self.target_frequency - 5, self.target_frequency + 5)
            self.spectogram_Ax.set_ylim(self.spec_ylim)

    def __init__(self, signal, name, target_freq):
        self.signal = signal
        self.plot_title = name
        self.target_frequency = target_freq
        self.create_layout()
        self.filter_signal()
        # self.spectogram_Ax.magnitude_spectrum(data, Fs = self.signal.sample_rate, color='C1',window=mlab.window_none)
        # self.spectogram_Ax.set_xlim(auto=True)
        # self.spectogram_Ax.set_ylim(auto=True)
        # self.resample_signal()
        self.plot_spectogram()
        self.set_axis_ylim()
        plt.show()

#
# MAIN
#
def main() -> int:

    target_frequency = None
    file = sys.argv[1]
    if len(sys.argv) > 2:
        target_frequency = int(sys.argv[2])

    if not os.path.exists(file):
        print("Not a real file")
        quit()

    path, ext = os.path.splitext(file)
    if ext.upper() in [".MP4", ".MKV", ".WEBM", ".M4A"]:
        file = extract_audio(file)

    print(f"file to process = {file}")
    signal = Signal(audio_to_numpy(file))

    print(f"sampling rate = {signal.sample_rate}")
    print(f"number of sample_count = {signal.sample_count}")
    print(f"audio length = {signal.duration}")
    print(f"stereo = {signal.is_stereo} ({type(signal.signal)})")
    print(f"target frequency = {target_frequency}\n")

    # Bring up spectogram with sliders to select frequency
    analyzer = FrequencySelector(signal, file, target_frequency)

if __name__ == '__main__':
    sys.exit(main())
