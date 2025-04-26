import os
import sys
import subprocess
import math
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RangeSlider
from scipy.signal import butter, filtfilt, sosfilt
from audio2numpy import open_audio
import pylab

# matplotlib.use('Agg')

def extract_audio(video_path, output_format="mp3"):

    try:
        filename, ext = os.path.splitext(video_path)
        output_file = f"{filename}.{output_format}"
        if os.path.exists(output_file):
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
    l_channel: list[float]
    r_channel: list[float]

    F_signal: list[float]
    F_l_channel: list[float]
    F_r_channel: list[float]

    def __init__(self, sig_tup):
        signal = sig_tup[0]
        sample_rate = sig_tup[1]

        self.F_signal = signal
        if type(signal[0]) is np.ndarray:
            self.signal = map(lambda x: x.item(), signal)
        else:
            self.signal = signal
        self.sample_rate = sample_rate
        self.sample_count = len(signal)
        self.duration = self.sample_count/self.sample_rate
        self.is_stereo = False
        self.l_channel = []
        self.r_channel = []
        if type(signal[0]) is np.ndarray:
            self.is_stereo = True
            for s in signal:
                self.l_channel.append(s[0].item())
                self.r_channel.append(s[1].item())
        self.F_l_channel = self.l_channel
        self.F_r_channel = self.r_channel

    def  low_pass_filter(self, max):
        # Filter parameters
        cutoff_freq = max  # Cutoff frequency in Hz
        fs = self.sample_rate  # Sampling rate in Hz
        order = 1  # Filter order

        # Design the low-pass filter
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        sos = butter(order, normal_cutoff, btype='lowpass', output="sos")

        # Apply the filter
        if self.is_stereo:
            self.F_l_channel = sosfilt(sos, self.F_l_channel)
            self.F_r_channel = sosfilt(sos, self.F_r_channel)
        else:
            self.F_signal = sosfilt(sos, self.F_signal)

    def high_pass_filter(self, min):
        # Filter parameters
        cutoff_freq = min  # Cutoff frequency in Hz
        fs = self.sample_rate  # Sampling rate in Hz
        order = 1  # Filter order

        # Design the high-pass filter (change btype to 'highpass')
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        sos = butter(order, normal_cutoff, btype='highpass', output="sos")  # <-- modify here

        # Apply the filter
        if self.is_stereo:
            self.F_l_channel = sosfilt(sos, self.F_l_channel)
            self.F_r_channel = sosfilt(sos, self.F_r_channel)
        else:
            self.F_signal = sosfilt(sos, self.F_signal)

    def  band_pass_filter(self, low, high):

        nyq = self.signal.sample_rate * 0.5
        lowcut = low / nyq
        highcut = high / nyq
        sos = butter(2, [lowcut, highcut], 'bandpass', fs=self.signal.sample_rate, output='sos')

        # Apply the filter
        if self.is_stereo:
            self.F_l_channel = sosfilt(sos,  self.F_l_channel)
            self.F_r_channel = sosfilt(sos,  self.F_r_channel)
        else:
            self.F_signal = sosfilt(sos, self.F_signal)

class Analyzer:
    """Analyzer"""

    signal = None
    plot_title = None
    Filter = True

    figure = None
    ax1 = None
    sl1 = None
    sl2 = None
    sl3 = None
    spc = None
    cb = None
    NFFT_slider = None
    NFFT_diff_slider = None
    padto_slider = None
    vminmax_slider = None
    resetax = None
    button = None
    plot_btn_ax = None
    plot_btn = None
    vmin_init = -548
    vmax_init = 0
    vmin = None
    vmax = None
    spec_ylim = (None, None)

    NFFT = 127000
    # NFFT = 213661

    noverlap = 128
    # noverlap = 201846

    pad_to = 650000
    # pad_to = 1193353
    
    # spec_ylim = (119,121)
    # spec_ylim = (119.8,120.3)

    # vmin = -58
    # vmax = -30

    filter_freq = 120

    cmap = pylab.get_cmap('Greys')
    scale_by_freq = True

    cmap = pylab.get_cmap('Greys')

    def reset(self, event):
        self.NFFT_slider.reset()
        self.padto_slider.reset()
        self.NFFT_diff_slider.reset()
        self.vminmax_slider.reset()
        self.plot_spec(None)

    def set_ylim(self):
        if self.spec_ylim is None:
            self.ax1.set_ylim(auto=True)
        else:
            self.ax1.set_ylim(self.spec_ylim)

    def get_specgram(self, ax, sig, NFFT, noverlap, Fs, pad_to, cmap, vmin, vmax):
        return ax.specgram(sig, NFFT=NFFT, noverlap=noverlap,
                    Fs=Fs, pad_to=pad_to, cmap=cmap, vmin=vmin, vmax=vmax, 
                    scale_by_freq=self.scale_by_freq)

    def get_vmin_vmax(self):
        if (self.vmin is None) or (self.vmax is None):
            return (self.vmin_init, self.vmax_init)
        else:
            return (self.vmin, self.vmax)
        
    def get_max(self, arr):
        max_val = 0
        max_val_index = 0
        for i in range(len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
                max_val_index = i
        return (max_val, max_val_index)
    
    def get_min(self, arr):
        min_val = 0
        min_val_index = 0
        for i in range(len(arr)):
            if arr[i] < min_val:
                min_val = arr[i]
                min_val_index = i
        return (min_val, min_val_index)
    
    def plot_line(self, event):
        y0, y1 = self.ax1.get_ylim()
        x0, x1 = self.ax1.get_xlim()
        spec, freqs, time, im = self.spc
        
        print(f"freqs length = {len(freqs)}")
        print(f"time length = {len(time)}")
        print(f"spec length = {len(spec)}")

        freq_indexes = []
        for fi in range(len(freqs)):
            if (freqs[fi] >= y0) and (freqs[fi] <= y1):
                freq_indexes.append(fi)
        
        time_indexes = []
        for ti in range(len(time)):
            if (time[ti] >= x0) and (time[ti] <= x1):
                time_indexes.append(ti)

        print(f"{min(time_indexes)}:{max(time_indexes)}, {min(freq_indexes)}:{max(freq_indexes)}")

        # Extract the area of frequencies we want.
        spectrum = spec[min(freq_indexes):max(freq_indexes),min(time_indexes):max(time_indexes)]
        
        print(f"freq")

        min_freq = min(freq_indexes)
        for j in range(len(time_indexes)):
            min_val = self.get_max(spectrum[:,j])[1]

            print(f"will plot at {(time[time_indexes[j]], freqs[min_freq+min_val])}" )

            self.ax1.plot(time[time_indexes[j]], freqs[min_freq+min_val], 'ro', linestyle="-")
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
    
    def create_layout(self):
        self.figure = plt.figure(figsize=(10,5))
        self.ax1 = plt.axes((0.085, 0.3, 1, 0.6))

        slider_max = int(self.signal.sample_rate*10)

        self.sl1 = plt.axes((0.085,0.225,0.8,0.02))
        self.NFFT_slider = Slider(
            ax = self.sl1,
            label = 'NFFT',
            valmin = 0,
            valmax = slider_max,
            valinit = self.NFFT,
            valfmt="%i"
        )

        self.sl2 = plt.axes((0.085,0.2,0.8,0.02))
        self.NFFT_diff_slider = Slider(
            ax = self.sl2,
            label = 'noverlap',
            valmin = 0,
            valmax = slider_max,
            valinit = self.noverlap,
            valfmt="%i"
        )

        self.sl3 = plt.axes((0.085,0.175,0.8,0.02))
        self.padto_slider = Slider(
            ax = self.sl3,
            label = 'pad_to',
            valmin = 0,
            valmax =  slider_max*4,
            valinit = self.pad_to,
            valfmt="%i"
        )

        self.sl4 = plt.axes((0.085,0.15,0.8,0.02))
        self.vminmax_slider = RangeSlider(
            ax = self.sl4,
            label = 'vmin_vmax',
            valmin = -550,
            valmax = 0,
            valinit = self.get_vmin_vmax(),
            valfmt="%i"
        )

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        self.resetax = plt.axes((0.085,0.1,0.390,0.04))
        self.button = Button(self.resetax, 'Reset', hovercolor='0.975')

        self.plot_btn_ax = plt.axes((0.495,0.1,0.390,0.04))
        self.plot_btn = Button(self.plot_btn_ax, 'Plot', hovercolor='0.975')
        
        self.NFFT_slider.on_changed(self.redraw_plot)
        self.padto_slider.on_changed(self.redraw_plot)
        self.NFFT_diff_slider.on_changed(self.redraw_plot)
        self.vminmax_slider.on_changed(self.set_spec_ylims)

        self.button.on_clicked(self.reset)
        self.plot_btn.on_clicked(self.plot_line)

        cax, kw = matplotlib.colorbar.make_axes(self.ax1, orienation="vertical")
        norm = matplotlib.colors.Normalize(vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])
        self.cb = matplotlib.colorbar.ColorbarBase(cax, cmap=self.cmap, norm=norm)

        self.ax1.set_title(self.plot_title)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel('Frequency (Hz)')
        self.ax1.set_xlim(0, self.signal.duration)

        self.set_ylim()

    def plot_spec(self, val):
        self.ax1.clear()
        if self.signal.is_stereo:
            data = self.signal.F_l_channel
        else:
            data = self.signal.signal

        self.spc = self.get_specgram(self.ax1, data, 
                NFFT=int(self.NFFT_slider.val), 
                noverlap=int(self.NFFT_diff_slider.val), 
                Fs=self.signal.sample_rate, 
                pad_to=int(self.padto_slider.val), cmap=self.cmap,
                vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])

        self.ax1.set_title(self.plot_title)
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel('Frequency (Hz)')
        self.ax1.set_xlim(0, self.signal.duration)

        self.y0, self.y1 = self.ax1.get_ylim()
        self.x0, self.x1 = self.ax1.get_xlim()

        # with open("spectrum.txt", "w") as f:
        #     l = map(lambda x: x.tolist(), self.spc[0])
        #     for n in l:
        #         f.write(str(n) + "\n")
        # with open("freqs.txt", "w") as f:
        #     for n in self.spc[1]:
        #         f.write(str(n) + "\n")
        # with open("time.txt", "w") as f:
        #     for n in self.spc[2]:
        #         f.write(str(n) + "\n")

    def redraw_plot(self, event):
        y0, y1 = self.ax1.get_ylim()
        x0, x1 = self.ax1.get_xlim()
        self.plot_spec(None)
        self.ax1.set_ylim(y0,y1)
        self.ax1.set_xlim(x0,x1)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def set_spec_ylims(self, val):
        for i in self.ax1.images:
            i.set_clim(self.vminmax_slider.val[0], self.vminmax_slider.val[1])
            self.cb.update_normal(i)
    
    def filter(self):
        if self.Filter:
            self.signal.low_pass_filter(self.filter_freq-1)
            self.signal.high_pass_filter(self.filter_freq+1)
            # self.signal.band_pass_filter(freq-3,freq+3)
    
    def set_plot_title(self, title):
        self.plot_title = title
        self.ax1.set_title(self.plot_title)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def __init__(self, signal, name):
        self.signal = signal
        self.plot_title = name
        self.create_layout()
        self.filter()
        self.plot_spec(None)
        self.set_ylim()

        # self.figure.savefig('Spectrogram.jpg')
        # self.figure.clear()
        # plt.cla()
        # plt.clf()
        # plt.close('all')
        # del self.ax1
        # del self.figure
        plt.show()
    
    

#
# MAIN
#
def main() -> int:
    file = sys.argv[1]

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

    analyzer = Analyzer(signal, file)

if __name__ == '__main__':
    sys.exit(main())