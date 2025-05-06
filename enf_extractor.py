import os
import sys
import pylab
import librosa
import argparse
import matplotlib
import subprocess
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from audio2numpy import open_audio
from scipy.signal import butter, sosfilt, savgol_filter
from matplotlib.widgets import Button, Slider, RangeSlider


def extract_audio(video_path, output_format="wav"):
    """Extract audio from video."""
    extracted_audio_folder = "./audio/"
    try:
        if os.path.exists(extracted_audio_folder) == False:
            os.mkdir(extracted_audio_folder)

        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = f"{extracted_audio_folder}{filename}.{output_format}"

        if os.path.exists(output_file):
            print(
                f"Will not extract audio since audio file already exists, will use that one.")
            return output_file

        subprocess.call(["ffmpeg", "-y", "-i", video_path, output_file],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)

        return output_file
    except Exception as e:
        raise Exception("Error when extracting audio", e)


def audio_to_numpy(audio_file):
    """ Convert audio file to a numpy array for processing. """
    signal, sampling_rate = open_audio(audio_file)
    return (signal, sampling_rate)


@dataclass
class Signal:
    """Class for signals, with filter and resampling methods. """
    signal: list[float]
    sample_rate: int
    sample_count: int
    duration: float

    # If signal is stereo, pick a channel
    is_stereo: bool
    L_channel: list[float]
    R_channel: list[float]

    # Filtered
    resampled_rate = None
    Filtered_signal: list[float]
    Filtered_L_channel: np.ndarray
    Filtered_R_channel: np.ndarray

    # Smoothing parameters.
    savgol_w_length = 50
    savgol_polyorder = 4

    def __init__(self, sig_tup):

        signal = sig_tup[0]
        sample_rate = sig_tup[1]
        self.Filtered_signal = signal

        self.sample_rate = sample_rate
        self.sample_count = len(signal)
        self.duration = self.sample_count/self.sample_rate
        self.is_stereo = False
        self.L_channel = []
        self.R_channel = []

        # If stereo, split channels.
        if type(signal[0]) is np.ndarray:
            self.is_stereo = True
            for s in signal:
                self.L_channel.append(s[0])
                self.R_channel.append(s[1])

        self.Filtered_L_channel = np.array(self.L_channel)
        self.Filtered_R_channel = np.array(self.R_channel)

    def create_butter_bandpass(self, lowcut, highcut, fs, order=5):
        """ Create butter filter. """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False,
                     btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, low, high, order=5):
        """ Filter useful signals only. """
        samples = self.sample_rate
        if self.resampled_rate is not None:
            samples = self.resampled_rate

        sos = self.create_butter_bandpass(low, high, samples, order=order)
        # Apply the filter
        if self.is_stereo:
            self.Filtered_L_channel = sosfilt(sos,  self.Filtered_L_channel)
            self.Filtered_R_channel = sosfilt(sos,  self.Filtered_R_channel)
        else:
            self.Filtered_signal = sosfilt(sos, self.Filtered_signal)

    def librosa_resample(self, new_sample):
        """ Resample audio to save copmutation. """
        if self.is_stereo:
            self.Filtered_L_channel = librosa.resample(
                self.Filtered_L_channel, orig_sr=self.sample_rate, target_sr=int(new_sample))
            self.Filtered_R_channel = librosa.resample(
                self.Filtered_R_channel, orig_sr=self.sample_rate, target_sr=int(new_sample))
        else:
            self.Filtered_signal = librosa.resample(
                self.Filtered_signal, orig_sr=self.sample_rate, target_sr=int(new_sample))
        self.resampled_rate = int(new_sample)


class FrequencySelector:
    """FrequencySelector"""
    # Properties
    segment_length = None
    vmin_init = -250
    vmax_init = 0
    y_mean = None
    target_frequency = None
    NFFT = 4096
    noverlap = int(NFFT/2)
    pad_to = NFFT
    spec_ylim = (None, None)
    max_line = mean_line = smooth_line = None

    # Matplotlib layout properties.
    signal = plot_title = figure = spectrogram_Ax = colorbar = None
    y0 = y1 = x0 = x1 = graph_Ax = None
    spec_data = spec_time = spec_frequencies = spec_image = None
    slider_Ax_1 = slider_Ax_2 = slider_Ax_3 = slider_Ax_3 = None
    NFFT_slider = noverlap_slider = padto_slider = vminmax_slider = None
    save_enf_plot_ax = reset_Btn_Ax = reset_Btn = plot_Btn_Ax = plot_Btn = None
    ENF_x_points = ENF_y_points = None

    # Layout colors
    cmap = pylab.get_cmap('Greys')
    button_color = "black"
    hovercolor = "0.2"

    def create_layout(self):
        """ Create layout. """
        plt.style.use('dark_background')

        if self.figure is None:
            self.figure = plt.figure(figsize=(10, 5))
        else:
            self.figure.clf()
        
        # Create slider axis.
        self.spectrogram_Ax = plt.axes((0.085, 0.3, 1, 0.6))
        slider_max = 25000

        # Create sliders.
        self.slider_Ax_1 = plt.axes((0.085, 0.225, 0.8, 0.02))
        self.NFFT_slider = Slider(
            ax=self.slider_Ax_1,
            label='NFFT',
            valmin=2,
            valmax=slider_max,
            valinit=self.NFFT,
            valfmt="%i"
        )
        self.NFFT_slider.on_changed(self.redraw_view)

        self.slider_Ax_2 = plt.axes((0.085, 0.2, 0.8, 0.02))
        self.noverlap_slider = Slider(
            ax=self.slider_Ax_2,
            label='noverlap',
            valmin=0,
            valmax=slider_max - 2,
            valinit=int(self.NFFT/2),
            valfmt="%i"
        )
        self.noverlap_slider.on_changed(self.redraw_view)

        self.slider_Ax_3 = plt.axes((0.085, 0.175, 0.8, 0.02))
        self.padto_slider = Slider(
            ax=self.slider_Ax_3,
            label='pad_to',
            valmin=0,
            valmax=slider_max*4,
            valinit=self.pad_to,
            valfmt="%i"
        )
        self.padto_slider.on_changed(self.redraw_view)

        self.slider_Ax_4 = plt.axes((0.085, 0.15, 0.8, 0.02))
        self.vminmax_slider = RangeSlider(
            ax=self.slider_Ax_4,
            label='vmin_vmax',
            valmin=-550,
            valmax=0,
            valinit=(self.vmin_init, self.vmax_init),
            valfmt="%i"
        )
        self.vminmax_slider.on_changed(self.set_spec_cb_lims)

        # Create reset button.
        self.reset_Btn_Ax = plt.axes((0.085, 0.1, 0.390, 0.04))
        self.reset_Btn = Button(self.reset_Btn_Ax, 'Reset All Parameters',
                                hovercolor=self.hovercolor, color=self.button_color)
        self.reset_Btn.on_clicked(self.reset_all)
        
        # Create plot button.
        self.plot_Btn_Ax = plt.axes((0.495, 0.1, 0.390, 0.04))
        self.plot_Btn = Button(self.plot_Btn_Ax, 'Plot All',
                               hovercolor=self.hovercolor, color=self.button_color)
        self.plot_Btn.on_clicked(self.plot_windows)

        # Create spectrogram color properties.
        cax, kw = matplotlib.colorbar.make_axes(
            self.spectrogram_Ax, orienation="vertical")
        norm = matplotlib.colors.Normalize(
            vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])
        self.colorbar = matplotlib.colorbar.ColorbarBase(
            cax, cmap=self.cmap, norm=norm)

        # Set spectrogram axis properties.
        self.spectrogram_Ax.set_title(self.plot_title)
        self.spectrogram_Ax.set_xlabel("Time (s)")
        self.spectrogram_Ax.set_ylabel('Frequency (Hz)')
        self.spectrogram_Ax.set_xlim(auto=True)
        self.set_axis_ylim()

    def plot_spectrogram(self):
        """ Plot spectrogram. """

        # n_overlap cannot be greater than NFFT.
        if self.noverlap_slider.val > self.NFFT_slider.val:
            self.spectrogram_Ax.set_title("noverlap cannot be greater than NFFT-1")
            return

        # Always use left channel if stereo. This is arbitrary.
        if self.signal.is_stereo:
            data = self.signal.Filtered_L_channel
        else:
            data = self.signal.Filtered_signal

        # Use full signal length or segment length given by user.
        end_index = self.signal.sample_count
        if self.segment_length is not None:
            end_index = self.segment_length

        # Use original signal sample rate or resampled rate.
        sr = self.signal.sample_rate
        if self.signal.resampled_rate is not None:
            sr = self.signal.resampled_rate

        # Plot spectrogram.
        self.spec_data, self.spec_frequencies, self.spec_time, self.spec_image = self.spectrogram_Ax.specgram(data[:end_index],
                                                                                                              NFFT=int(self.NFFT_slider.val),
                                                                                                              noverlap=int(self.noverlap_slider.val),
                                                                                                              Fs=sr,
                                                                                                              pad_to=int(self.padto_slider.val), cmap=self.cmap,
                                                                                                              vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1])
        # Set graph properties.
        self.spectrogram_Ax.set_title(self.plot_title)
        self.spectrogram_Ax.set_xlabel("Time (s)")
        self.spectrogram_Ax.set_ylabel('Frequency (Hz)')

    def has_acceptable_deviation(self, prev_freq, curr_freq):
        """ Check if previous point is within reasonable range. """
        # This idea should really use more statistical analysis.
        dev = 0.06
        if prev_freq is not None:
            diff = abs(prev_freq - curr_freq)
            if (diff <= dev):
                return True
            else:
                return False
        return True

    def get_max_w_index(self, time_slice, wndw_min_freq_index):
        """ Get the highest frequency and its index from a single slice of time in the spectrogram. """
        max_val = -1
        max_val_index = -1
        prev_freq = None
        # Get last frequency.
        if len(self.ENF_y_points) > 0:
            prev_freq = self.ENF_y_points[-1]
        # Select highest frequency.
        for i in range(len(time_slice)):
            curr_freq = self.spec_frequencies[wndw_min_freq_index + i]
            # Compare with last frequency and check if it's within range.
            if (time_slice[i] > max_val) and self.has_acceptable_deviation(prev_freq, curr_freq):
                max_val = time_slice[i]
                max_val_index = i
        return (max_val, max_val_index)

    def plot_windows(self, event):
        """ Plot the ENF for the current window. Window can be full spectrograph or segment. """
        print("Will start plotting the ENF...")

        # If segment is enabled create top/bottom view.
        end_index = self.signal.sample_count
        if self.segment_length is not None:
            end_index = self.segment_length
            self.create_top_and_bottom_layout()
            self.spectrogram_Ax.set(xticklabels=[])

        self.redraw_view()

        # Get the range of frequencies that are currently visible within the plot.
        # Only these will be considered for the ENF.
        self.freq_indexes = []
        for fi in range(len(self.spec_frequencies)):
            if (self.spec_frequencies[fi] >= self.y0) and (self.spec_frequencies[fi] <= self.y1):
                self.freq_indexes.append(fi)

        # Get sample count and rate if resampled.
        sr = self.signal.sample_rate
        sc = self.signal.sample_count
        if self.signal.resampled_rate is not None:
            sr = self.signal.resampled_rate
            sc = (self.signal.resampled_rate * self.signal.duration)

        # X overlap acounts for the shrinkage during the plotting of the spectrogram.
        # It makes sure that no points are skipped when plotting segments.
        x0_overlap = int((self.spec_time[0] * sr))
        graph_end_point_index = int(
            ((end_index / sr)-(self.spec_time[-1])) * sr)
        x1_overlap = x0_overlap + graph_end_point_index

        print(f"x extents of data: {x0_overlap}, {x1_overlap}")

        if self.signal.is_stereo:
            data = self.signal.Filtered_L_channel
        else:
            data = self.signal.Filtered_signal

        # Get initial y values and keep them for the duration of the plotting.
        self.y0, self.y1 = self.spectrogram_Ax.get_ylim()
        print(f"y extents of data: {(self.y0, self.y1)}")

        # Clear arrays, this is what will be output as CSV.
        self.ENF_y_points = []
        self.ENF_x_points = []

        # Starting segment.
        x0 = 0
        x1 = end_index

        # Calculate segments.
        while (True):

            # Calculate spectrogram in current window.
            self.save_viewlims()
            self.spectrogram_Ax.clear()
            self.spectrogram_Ax.set_title(f"Sample for {self.plot_title}")
            self.spec_data, self.spec_frequencies, self.spec_time, self.spec_image = self.spectrogram_Ax.specgram(data[x0:x1],
                                                                                                                  NFFT=int(self.NFFT_slider.val),
                                                                                                                  noverlap=int(self.noverlap_slider.val),
                                                                                                                  Fs=sr,
                                                                                                                  pad_to=int(self.padto_slider.val), cmap=self.cmap,
                                                                                                                  vmin=self.vminmax_slider.val[0], vmax=self.vminmax_slider.val[1]
                                                                                                                  )
            self.spectrogram_Ax.set_ylim(self.y0, self.y1)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            self.spectrogram_Ax.set_ylim(self.y0, self.y1)

            # Plot the line.
            self.graph_ENF(x0)

            if x1 >= sc:
                print(f"Done plotting")
                break
            else:
                print(f"Moving window to {x0/sr}s, {x1/sr}s : x1 = {x1}")

            # Move window.
            x0 = x1 - x1_overlap
            if self.segment_length is None:
                x1 = int(x1 + x1_overlap)
            else:
                x1 = int(x1 + self.segment_length - x1_overlap)

        self.y_mean = np.mean(self.ENF_y_points)

        self.mean_line = self.spectrogram_Ax.plot(self.spectrogram_Ax.get_xlim(), [
                                                  self.y_mean, self.y_mean], 'b-', label='Mean')
        if self.segment_length is not None:
            savgol_win_length = int(len(self.ENF_y_points)/10)
            self.graph_Ax.clear()
            self.graph_Ax.plot(self.ENF_x_points,
                               self.ENF_y_points, 'r-', label="Max")
            self.graph_Ax.plot(self.ENF_x_points, savgol_filter(
                self.ENF_y_points, savgol_win_length, self.signal.savgol_polyorder), "g-", label="Smoothed")
            self.graph_Ax.plot(self.graph_Ax.get_xlim(), [
                               self.y_mean, self.y_mean], 'b-', label='Mean')

        # When done create "save" button and refresh image.
        self.spectrogram_Ax.set_ylabel('Frequency (Hz)')
        self.spectrogram_Ax.set_xlabel("Time (s)")
        self.save_enf_plot_ax = plt.axes((0.495, 0.05, 0.390, 0.04))
        self.save_enf_plot_Btn = Button(
            self.save_enf_plot_ax, 'Save ENF plot', hovercolor=self.hovercolor, color=self.button_color)
        self.save_enf_plot_Btn.on_clicked(self.save_enf_plot)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def graph_ENF(self, start_index):
        """ Plot a line composed of the largest frequencies """
        start_sec = 0
        if start_index > 0:
            if self.signal.resampled_rate is None:
                start_sec = start_index/self.signal.sample_rate
            else:
                start_sec = start_index/self.signal.resampled_rate

        # Get current plot extent.
        x0, x1 = self.spectrogram_Ax.get_xlim()

        curr_x_points = []
        curr_y_points = []
        spec_x_points = []

        # The index of the lowest frequency in the window, all checked frequencies will be above this one.
        window_bottom_freq_index = min(self.freq_indexes)
        window_top_freq_index = max(self.freq_indexes)

        for ti in range(len(self.spec_time)):
            current_time = self.spec_time[ti]
            if (current_time >= x0) and (current_time <= x1):

                max_freq_index = self.get_max_w_index(
                    self.spec_data[window_bottom_freq_index: window_top_freq_index, ti],
                    window_bottom_freq_index)[1]

                max_frequency = self.spec_frequencies[window_bottom_freq_index + max_freq_index]

                curr_y_points.append(max_frequency)
                self.ENF_y_points.append(max_frequency)

                curr_x_points.append(current_time + start_sec)
                self.ENF_x_points.append(current_time + start_sec)

                spec_x_points.append(current_time)
                print(f"({current_time + start_sec}s, {max_frequency}Hz)")

        savgol_win_length = self.signal.savgol_w_length
        if savgol_win_length > len(self.ENF_y_points):
            savgol_win_length = len(self.ENF_y_points)-2

        # Show plots.
        self.spectrogram_Ax.plot(
            spec_x_points, curr_y_points, 'r-', label="Max")
        if self.segment_length is None:
            self.spectrogram_Ax.plot(self.ENF_x_points, savgol_filter(
                self.ENF_y_points, savgol_win_length, self.signal.savgol_polyorder), "g-", label="Smoothed")
        else:
            self.graph_Ax.clear()
            self.graph_Ax.plot(self.ENF_x_points,
                               self.ENF_y_points, 'r-', label="Max")
            # self.graph_Ax.plot(self.ENF_x_points, savgol_filter(self.ENF_y_points, savgol_win_length, self.signal.savgol_polyorder), "g-", label="Smoothed")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def filter_signal(self):
        """ Filter signal if target_frequency is set. """
        if self.target_frequency is not None:
            self.signal.butter_bandpass_filter(
                self.target_frequency-1, self.target_frequency+1, 4)

    def save_enf_plot(self, event):
        """ Save ENF data. """
        if (self.ENF_x_points is None) or (self.ENF_y_points is None):
            print("No data to print.")
        date = datetime.today().strftime('%Y-%m-%d')
        fname = f"{os.path.splitext(os.path.basename(self.plot_title))[0]}_{date}.csv"
        # Smooth
        smoothed_ENF_y_points = savgol_filter(self.ENF_y_points, int(
            self.signal.savgol_w_length), self.signal.savgol_polyorder)
        with open(fname, "w") as f:
            f.write(f"Time(s), Frequency(Hz), Smoothed_Frequency(Hz)\n")
            for i in range(len(self.ENF_x_points)):
                f.write(
                    f"{self.ENF_x_points[i].item()}, {self.ENF_y_points[i].item()}, {smoothed_ENF_y_points[i]}\n")
        print(f"printed to {fname}")

    # Suplemental functions.
    def create_top_and_bottom_layout(self):
        """ Create view for plotting segments. """
        if self.graph_Ax is None:
            self.graph_Ax = plt.axes((0.085, 0.3, 0.8, 0.3))
        self.graph_Ax.clear()
        self.graph_Ax.grid()
        self.graph_Ax.set_xlabel("Time (s)")
        self.save_viewlims()
        self.spectrogram_Ax.remove()
        self.spectrogram_Ax = plt.axes((0.085, 0.6, 0.8, 0.3))
        self.spectrogram_Ax.set_xlabel("")
        self.spectrogram_Ax.set_title(f"Sample for {self.plot_title}")
        self.spectrogram_Ax.set_ylim(self.y0, self.y1)
        self.spectrogram_Ax.set_xlim(auto=True)

    def save_viewlims(self):
        """ Save the visible limits of the axes."""
        self.y0, self.y1 = self.spectrogram_Ax.get_ylim()
        self.x0, self.x1 = self.spectrogram_Ax.get_xlim()

    def reset_all(self, event):
        """ Reset application to initial parameters. """
        print("Resetting...", end="")
        self.NFFT_slider.reset()
        self.padto_slider.reset()
        self.noverlap_slider.reset()
        self.vminmax_slider.reset()
        if self.save_enf_plot_ax is not None:
            self.save_enf_plot_ax.remove()
        if self.graph_Ax is not None:
            self.graph_Ax.remove()
        self.create_layout()
        self.ENF_x_points = []
        self.ENF_x_points = []
        self.plot_spectrogram()
        self.set_axis_ylim()
        self.spectrogram_Ax.set_xlim(auto=True)
        print("Done.")

    def redraw_view(self, event=None):
        """ Redraw spectrogram with the current ranges. """
        self.save_viewlims()
        self.redraw_spectrogram()
        self.spectrogram_Ax.set_ylim(self.y0, self.y1)
        self.spectrogram_Ax.set_xlim(auto=True)

    def redraw_spectrogram(self):
        """ Redraw spectrogram in the same position as before and erase ENF."""
        self.spectrogram_Ax.clear()
        self.plot_spectrogram()
        self.spectrogram_Ax.set_ylim(self.y0, self.y1)
        self.spectrogram_Ax.set_xlim(self.x0, self.x1)
        self.ENF_y_points = []
        self.ENF_x_points = []
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def set_spec_cb_lims(self, val):
        """ Draw colorbar when the vmin/vmax slider changes. """
        for i in self.spectrogram_Ax.images:
            i.set_clim(self.vminmax_slider.val[0], self.vminmax_slider.val[1])
            self.colorbar.update_normal(i)

    def set_axis_ylim(self):
        """ Set height of graph to center on target frequency or automatic. """
        if self.target_frequency is None:
            self.spectrogram_Ax.set_ylim(auto=True)
        else:
            self.spec_ylim = (self.target_frequency - 1,
                              self.target_frequency + 1)
            self.spectrogram_Ax.set_ylim(self.spec_ylim)

    def __init__(self, signal, name, target_freq = None, segment_length_seconds = None):
        self.signal = signal
        self.plot_title = name
        self.target_frequency = target_freq

        # Fixed resample targets 60, 120Hz and 180Hz harmonics.
        resample_target = 420
        
        self.filter_signal()
        self.filter_signal()
        self.signal.librosa_resample(resample_target)
        if segment_length_seconds is not None:
            self.segment_length = int(
                segment_length_seconds * self.signal.resampled_rate)
        else:
            self.segment_length = None

        self.create_layout()

        # If you choose a really small segment size change the slider values. Not advisable. 
        if (self.segment_length is not None) and (self.segment_length < self.NFFT):
            self.NFFT = int(self.segment_length/2)
            self.NFFT_slider.set_val(self.NFFT)
            self.NFFT_slider.valmax = self.segment_length
            self.noverlap_slider.set_val(int(self.NFFT/2))
            self.noverlap_slider.valmax = self.segment_length - 2
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        print(f"length={self.segment_length}\nnfft={self.NFFT_slider.val}\nnoverlap={self.noverlap_slider.val}")
        self.plot_spectrogram()

        # Set y limits to auto or, if given, centered around target frequency.
        self.set_axis_ylim()
        plt.show()

#
# MAIN
#


def main() -> int:

    # Arguments.
    parser = argparse.ArgumentParser(
        prog='ENF-Extractor',
        description='Extracts ENF from audio and video.')
    parser.add_argument(
        'filename', help='File to be processed, can be an audio or video file.')
    parser.add_argument(
        '-f', '--frequency', help='Target frequency to filter and center the plot on. Dont add one if you havent checked its there first.', type=int)
    parser.add_argument(
        '-s', '--seconds', help='Segments of plot to iteratively calculate.', type=int)
    args = parser.parse_args()

    file = args.filename

    if not os.path.exists(file):
        print("File does not exist.")
        quit()
    
    # Check type and extract if necesary.
    path, ext = os.path.splitext(file)
    if ext.upper() in [".MP4", ".MKV", ".WEBM", ".M4A"]:
        file = extract_audio(file)

    print(f"file to process = {file}")
    signal = Signal(audio_to_numpy(file))

    print(f"sampling rate = {signal.sample_rate}")
    print(f"number of sample_count = {signal.sample_count}")
    print(f"audio length = {signal.duration}")
    print(f"stereo = {signal.is_stereo}")
    print(f"target frequency = {args.frequency}\n")

    # Bring up spectrogram with sliders to select frequency.
    analyzer = FrequencySelector(
        signal, file, args.frequency, args.seconds)


if __name__ == '__main__':
    sys.exit(main())
