import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
import random
from scipy.io import wavfile
from scipy.signal import resample
from scipy import signal
import numpy as np

# Calculate and plot spectrogram for a wav audio file
def stft(wav_file, plot=False):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    data = np.fromstring(data, np.int16)
    data = data.astype(np.float64).reshape((-1,2))
    data -= data.min()
    data /= data.max() / 2.
    data -= 1.
    if nchannels == 1:
        if(plot):
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
            plt.figure()
            plt.plot(data)
            plt.ylabel('Amp')
            plt.show()
        _, _, Zxx = signal.stft(data[:,1], fs=fs, nperseg = nfft)
        
    elif nchannels == 2:
        if(plot):
            Zxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs)
            plt.figure()
            plt.plot(data)
            plt.ylabel('Amp')
            plt.show()
        _, _, Zxx = signal.stft(data[:,1], fs=fs, nperseg = nfft)
        
        

    return Zxx           


def istft(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    _, Zxx = signal.istft(data, fs=fs, nperseg = nfft)
    plt.figure()
    plt.plot(Zxx)
    plt.ylabel('Amp')
    plt.show()
    return Zxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Fetch data from training_samples Folder :
def fetch_dataset(X, Y):
    dataset_len = len(os.listdir("./training_samples/%s" % X))
    random_number = random.randint(0, dataset_len - 1)
    x = stft("./training_samples/%s/train_%d.wav" % (X, random_number), False).T
    y = stft("./training_samples/%s/train_%d.wav" % (Y, random_number), False).T
    return x, y
    
       
    
