# Fungsi untuk menghitung amplitudo spektrum yang sudah dinormalisasi
import numpy as np 
def fft_sinyal(signal):
    sample_rate = 20
    
     # Melakukan FFT pada sinyal
    fft_data = np.fft.fft(signal)
    
    # Menghitung magnitudo spektrum FFT
    fft_magnitude = np.abs(fft_data)
    # Normalisasi amplitudo
    normalized_amplitude = 2.0 / len(signal) * fft_magnitude

    freq_axis = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Ambil hanya data frekuensi positif
    freq_axis = freq_axis[:len(freq_axis)//2]
    normalized_amplitude = normalized_amplitude[:len(normalized_amplitude)//2]
    
    return freq_axis, normalized_amplitude
