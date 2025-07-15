import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
from sklearn.preprocessing import PowerTransformer
from scipy import signal

fileName ='data_capture.bin'

def read_raw(fileName):
    numADCSamples = 200 # number of ADC samples per chirp
    numADCBits = 16 # number of ADC bits per sample
    numRX = 4 # number of receivers
    numLanes = 2 # do not change. number of lanes is always 2
    isReal = 0 # set to 1 if real only data, 0 if complex data0
    fid = open(fileName,'rb')
    adcData = np.fromfile(fid, dtype='int16')
    if numADCBits != 16:
        l_max = 2**(numADCBits-1)-1
        adcData[adcData > l_max] = adcData[adcData > l_max] - 2**numADCBits
    fid.close()
    fileSize = adcData.shape[0]

    if isReal:
        numChirps = fileSize//numADCSamples//numRX
        LVDS = np.zeros((fileSize,), dtype=np.int16)
        LVDS = adcData.reshape(numADCSamples*numRX, numChirps, order='F').T
    else:
        numChirps = fileSize//2//numADCSamples//numRX
        LVDS = np.zeros((fileSize//2,), dtype=np.complex64)
        counter = 0
        for i in range(0, fileSize-1, 4):
            LVDS[counter] = adcData[i] + 1j*adcData[i+2]
            LVDS[counter+1] = adcData[i+1] + 1j*adcData[i+3]
            counter += 2
        LVDS = LVDS.reshape(numADCSamples*numRX, numChirps, order='F').T

    adcData = np.zeros((numRX,numChirps*numADCSamples), dtype=np.complex64)
    for row in range(numRX):
        for i in range(numChirps):
            adcData[row, i*numADCSamples:(i+1)*numADCSamples] = LVDS[i, row*numADCSamples:(row+1)*numADCSamples]

    retVal = adcData
    return retVal

from scipy.signal import butter, sosfiltfilt, filtfilt,lfilter
def _butter_bandpass1(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    # b, a = butter(order, [low, high], btype='band')
    return sos
def _butter_bandpass2(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # sos = butter(order, [low, high], btype="band", output="sos")
    b, a = butter(order, [low, high], btype='band')
    return b,a
def butter_bandpass_filter(signal, lowcut, highcut, fs, order):
    sos = _butter_bandpass1(lowcut, highcut, fs, order=order)
    b,a = _butter_bandpass2(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, signal)
    y2 = filtfilt(b,a, signal)
    return y
import numpy as np
import scipy.fftpack as fftpack
# Temporal bandpass filter with Fast-Fourier Transform
def fft_filter(data, freq_min, freq_max, fs):
    fft = fftpack.fft(data, axis=0)
    frequencies = fftpack.fftfreq(data.shape[0], d=1.0 / fs)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=0)
    result = np.abs(iff)
    result *= 100  # Amplification factor

    return result, fft, frequencies

from scipy import signal
# Calculate heart rate from FFT peaks
def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60

def bpm_hr(data):
    hasil =[]
    for i in range(data.shape[0]):
        freq_min = 1.3
        freq_max = 1.8
        fs = 20
        result, fft, frequencies = fft_filter(data[i], freq_min, freq_max, fs)
        heart_rate = find_heart_rate(fft, frequencies, freq_min, freq_max)
        hasil.append(heart_rate)
    return hasil 
def bpm_rr(data):
    hasil =[]
    for i in range(data.shape[0]):
        freq_min = 0.2
        freq_max = 0.5
        fs = 20
        result, fft, frequencies = fft_filter(data[i], freq_min, freq_max, fs)
        heart_rate = find_heart_rate(fft, frequencies, freq_min, freq_max)
        hasil.append(heart_rate)
    return hasil 
        

def slide_win_rr(data):
    isi_dataVT =[]
    i = 0
    while(True):
        if i == 0:
            index_1 = i
        else :
            index_1 = index_2 - 380
        index_2 = index_1 + 400
        if index_2 > data.shape[0]:
            break
        vt_20s_1 = data[index_1:index_2]
    #print(index_1,index_2)
        isi_dataVT.append(vt_20s_1)
        i+=1

    return isi_dataVT

def slide_win_hr(data):
    isi_dataVT =[]
    i = 0
    while(True):
        if i == 0:
            index_1 = i
        else :
            index_1 = index_2 - 2470
        index_2 = index_1 + 2600
        if index_2 > data.shape[0]:
            break
        vt_20s_1 = data[index_1:index_2]
    #print(index_1,index_2)
        isi_dataVT.append(vt_20s_1)
        i+=1

    return isi_dataVT

def processing_data_bin():
    Radar_data = read_raw(fileName)

    # Setting Radar
    n_rx = 4 
    n_fft = 1024
    n_adc = 256
    n_chirp = 2
    # Reshape Data 
    raw_data = Radar_data
    raw_new =raw_data.reshape((n_rx,n_adc*n_chirp,-1), order='F')
    data_len= raw_new.shape[2]

    def Range_FFT(InputData):
        OutputData = np.zeros((n_rx, n_fft, data_len), dtype=complex)
        dc_removal = np.zeros((n_rx, n_fft//2, data_len), dtype=complex)
        for i in range(n_rx):
            for j in range(data_len):
                window = np.hamming(n_fft)
                dc_removal[i,:,j]= InputData[i,:,j] - np.mean(InputData[i,:,j])
                OutputData[i,:,j]= fft((dc_removal[i,:,j]), n=1024)
        return OutputData

    def MTI(hasil_fft):
        clutter = np.zeros((hasil_fft.shape[0], n_fft, hasil_fft.shape[-1]),dtype=complex)
        OutputData = np.zeros((hasil_fft.shape[0],n_fft//2, hasil_fft.shape[-1]),dtype=complex)
        alpha = 0.01
        for i in np.arange(hasil_fft.shape[0]):
            for j in np.arange(1,hasil_fft.shape[-1]):
                clutter[i,:,j] = np.dot(alpha,hasil_fft[i,:,j])+ np.dot((1-alpha),clutter[i,:,j-1])
                OutputData[i,:,j] = hasil_fft[i,np.arange(n_fft//2),j]-(clutter[i,range(0, n_fft//2),j])
        return OutputData
    # Map Lokasinya 
    def map_loc(data_mti):
        Range_Map_An1 = []
        index_max = []
        for i in range(1, data_len):
            # Input data From MTI each Antenna 
            awal_an1 = data_mti[:, i] 
            abs_mti_1 = abs(awal_an1)
            peak_loc_an1 = np.argmax(abs_mti_1[25:60]) #pilih batas jarak
            rslt_peak_1 = 25 + peak_loc_an1
            if i > 1 and abs(rslt_peak_1 - index_max[-1]) == 10:
                # Jika perbedaan adalah 10, pilih peak tertinggi kedua
                abs_mti_1[peak_loc_an1] = 0  # Hilangkan peak pertama
                peak_loc_an1 = np.argmax(abs_mti_1[25:60])
                rslt_peak_1 = 25 + peak_loc_an1
            map_loc_an1 = awal_an1[rslt_peak_1]
            index_max.append(abs(rslt_peak_1))
            Range_Map_An1.append(map_loc_an1)
        return index_max, Range_Map_An1

    # Lakukan proses FFT
    FFT_hasil = Range_FFT(raw_new) # hasil FFT
    Hasil_mti = MTI(FFT_hasil)

    # Hasil MTI untuk setiap antena 
    Antena1 = Hasil_mti[0,:,:]
    Antena2 = Hasil_mti[1,:,:]
    Antena3 = Hasil_mti[2,:,:]
    Antena4 = Hasil_mti[3,:,:]



    # Contoh penggunaan
    map_locA1, Map_loc_An1 = np.array(map_loc(Antena1))
    map_locA2, Map_loc_An2 = np.array(map_loc(Antena2))
    map_locA3, Map_loc_An3 = np.array(map_loc(Antena3))
    map_locA4, Map_loc_An4 = np.array(map_loc(Antena4))

    # Phase Extraction
    from f_dacm import dacm
    from f_noise_RM import remove_noise

    DACM_An1 = np.array(dacm(np.array(Map_loc_An1)))
    DACM_An2 = np.array(dacm(np.array(Map_loc_An2)))
    DACM_An3 = np.array(dacm(np.array(Map_loc_An3)))
    DACM_An4 = np.array(dacm(np.array(Map_loc_An4)))

    from f_noise_RM import remove_noise
    NR_1 = remove_noise(DACM_An1)
    NR_2 = remove_noise(DACM_An2)
    NR_3 = remove_noise(DACM_An3)
    NR_4 = remove_noise(DACM_An4)



    low = 0.1
    high = 0.5
    fs = 20
    BPF_RR1= np.array(butter_bandpass_filter(NR_1,low,high,fs,order=3))
    BPF_RR2 = np.array(butter_bandpass_filter(NR_2,low,high,fs,order=3))
    BPF_RR3 = np.array(butter_bandpass_filter(NR_3,low,high,fs,order=3))
    BPF_RR4 = np.array(butter_bandpass_filter(NR_4,low,high,fs,order=3))

    lowhr =0.8
    highhr = 2.0
    fs = 20
    BPF_HR1 = np.array(butter_bandpass_filter(NR_1,lowhr,highhr,fs,order=6))
    BPF_HR2 = np.array(butter_bandpass_filter(NR_2,lowhr,highhr,fs,order=6))
    BPF_HR3 = np.array(butter_bandpass_filter(NR_3,lowhr,highhr,fs,order=6))
    BPF_HR4 = np.array(butter_bandpass_filter(NR_4,lowhr,highhr,fs,order=6))

    rr_predict = load_model("rr_model_tuner.h5", custom_objects={'mse': 'mean_squared_error'}, compile=False)
    hr_predict = load_model("hr_model_tuner.h5", custom_objects={'mse': 'mean_squared_error'}, compile=False)


    slide_an1_rr = np.array(slide_win_rr(BPF_RR1),dtype=object)
    slide_an2_rr = np.array(slide_win_rr(BPF_RR2),dtype=object)
    slide_an3_rr = np.array(slide_win_rr(BPF_RR3),dtype=object)
    slide_an4_rr = np.array(slide_win_rr(BPF_RR4),dtype=object)

    slide_an1_hr = np.array(slide_win_rr(BPF_HR1),dtype=object)
    slide_an2_hr = np.array(slide_win_rr(BPF_HR2),dtype=object)
    slide_an3_hr = np.array(slide_win_rr(BPF_HR3),dtype=object)
    slide_an4_hr = np.array(slide_win_rr(BPF_HR4),dtype=object)
    
    resample1 = []
    resample2 = []
    resample3 = []
    resample4 = []
    
    for i in range(len(slide_an1_hr)):
        pt = PowerTransformer()
        
        hr_norm1 = pt.fit_transform(slide_an1_hr[i].reshape(-1, 1))
        hr_norm2 = pt.fit_transform(slide_an2_hr[i].reshape(-1, 1))
        hr_norm3 = pt.fit_transform(slide_an3_hr[i].reshape(-1, 1))
        hr_norm4 = pt.fit_transform(slide_an4_hr[i].reshape(-1, 1))
        
        hr_resample1 = signal.resample(hr_norm1, 2600)
        hr_resample2 = signal.resample(hr_norm2, 2600)
        hr_resample3 = signal.resample(hr_norm3, 2600)
        hr_resample4 = signal.resample(hr_norm4, 2600)
        
        resample1.append(hr_resample1)
        resample2.append(hr_resample2)
        resample3.append(hr_resample3)
        resample4.append(hr_resample4)
        
    resample1 = np.array(resample1)
    resample2 = np.array(resample2)
    resample3 = np.array(resample3)
    resample4 = np.array(resample4)
    
    
    slide_an1_rr = np.array([np.array(rr_predict(s.reshape(1, 400, 1))).squeeze() for s in slide_win_rr(BPF_RR1)])
    slide_an2_rr = np.array([np.array(rr_predict(s.reshape(1, 400, 1))).squeeze() for s in slide_win_rr(BPF_RR2)])
    slide_an3_rr = np.array([np.array(rr_predict(s.reshape(1, 400, 1))).squeeze() for s in slide_win_rr(BPF_RR3)])
    slide_an4_rr = np.array([np.array(rr_predict(s.reshape(1, 400, 1))).squeeze() for s in slide_win_rr(BPF_RR4)])


    slide_an1_hr = []
    slide_an2_hr = []
    slide_an3_hr = []
    slide_an4_hr = []
    for i in range(len(resample1)):
        slide_an1_hr.append(hr_predict(resample1[i].reshape(1, -1, 1).astype(np.float64))[0,:,0])
        slide_an2_hr.append(hr_predict(resample2[i].reshape(1, -1, 1).astype(np.float64))[0,:,0])
        slide_an3_hr.append(hr_predict(resample3[i].reshape(1, -1, 1).astype(np.float64))[0,:,0])
        slide_an4_hr.append(hr_predict(resample4[i].reshape(1, -1, 1).astype(np.float64))[0,:,0])
    
    slide_an1_hr = np.array(slide_an1_hr)
    slide_an2_hr = np.array(slide_an2_hr)
    slide_an3_hr = np.array(slide_an3_hr)
    slide_an4_hr = np.array(slide_an4_hr)
    
    
    savgol_hr1 = []
    savgol_hr2 = []
    savgol_hr3 = []
    savgol_hr4 = []
    savgol_rr1 = []
    savgol_rr2 = []
    savgol_rr3 = []
    savgol_rr4 = []
    for i in range(len(slide_an1_hr)):
        savgol_hr1.append(savgol_filter(slide_an1_hr[i].squeeze(), window_length=101, polyorder=4))
        savgol_hr2.append(savgol_filter(slide_an2_hr[i].squeeze(), window_length=101, polyorder=4))
        savgol_hr3.append(savgol_filter(slide_an3_hr[i].squeeze(), window_length=101, polyorder=4))
        savgol_hr4.append(savgol_filter(slide_an4_hr[i].squeeze(), window_length=101, polyorder=4))
        savgol_rr1.append(savgol_filter(slide_an1_rr[i].squeeze(), window_length=85, polyorder=3))
        savgol_rr2.append(savgol_filter(slide_an2_rr[i].squeeze(), window_length=85, polyorder=3))
        savgol_rr3.append(savgol_filter(slide_an3_rr[i].squeeze(), window_length=85, polyorder=3))
        savgol_rr4.append(savgol_filter(slide_an4_rr[i].squeeze(), window_length=85, polyorder=3))
    
    slide_an1_rr = np.array(savgol_rr1)
    slide_an2_rr = np.array(savgol_rr2)
    slide_an3_rr = np.array(savgol_rr3)
    slide_an4_rr = np.array(savgol_rr4)

    slide_an1_hr = np.array(savgol_hr1)
    slide_an2_hr = np.array(savgol_hr2)
    slide_an3_hr = np.array(savgol_hr3)
    slide_an4_hr = np.array(savgol_hr4)
    
    
    pt_hr = PowerTransformer()
    pt_rr = PowerTransformer()

    # Gabungkan semua HR dan RR untuk fitting transformer
    all_hr = np.concatenate([slide_an1_hr, slide_an2_hr, slide_an3_hr, slide_an4_hr], axis=0)
    all_rr = np.concatenate([slide_an1_rr, slide_an2_rr, slide_an3_rr, slide_an4_rr], axis=0)

    pt_hr.fit(all_hr)
    pt_rr.fit(all_rr)

    slide_an1_hr = pt_hr.transform(slide_an1_hr)
    slide_an2_hr = pt_hr.transform(slide_an2_hr)
    slide_an3_hr = pt_hr.transform(slide_an3_hr)
    slide_an4_hr = pt_hr.transform(slide_an4_hr)

    slide_an1_rr = pt_rr.transform(slide_an1_rr)
    slide_an2_rr = pt_rr.transform(slide_an2_rr)
    slide_an3_rr = pt_rr.transform(slide_an3_rr)
    slide_an4_rr = pt_rr.transform(slide_an4_rr)
    
    hr_1 = np.array(bpm_hr(slide_an1_hr))
    hr_2 = np.array(bpm_hr(slide_an2_hr))
    hr_3 = np.array(bpm_hr(slide_an3_hr))
    hr_4 = np.array(bpm_hr(slide_an4_hr))
    rr_1 = np.array(bpm_rr(slide_an1_rr))
    rr_2 = np.array(bpm_rr(slide_an2_rr))
    rr_3 = np.array(bpm_rr(slide_an3_rr))
    rr_4 = np.array(bpm_rr(slide_an4_rr))
    
    
    return slide_an1_hr, slide_an2_hr, slide_an3_hr, slide_an4_hr, slide_an1_rr, slide_an2_rr, slide_an3_rr, slide_an4_rr, hr_1, hr_2, hr_3, hr_4, rr_1, rr_2, rr_3, rr_4

