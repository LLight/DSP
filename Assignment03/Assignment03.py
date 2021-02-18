# Assignment 3: Fourier Analysis


import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import math


# Question 1: Generating sinusoids

# 1.1

def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, 
                       length_secs, phase_radians):
    t = np.linspace(0,length_secs,math.ceil(sampling_rate_Hz*length_secs))
    x = amplitude * np.sin(2*np.pi*frequency_Hz*t + phase_radians)
    return t, x


# 1.2

t1, x1 = generateSinusoidal(amplitude = 1.0, sampling_rate_Hz = 44100, 
                   frequency_Hz = 400, length_secs = 0.5, phase_radians = np.pi/2)


# 1.3.

t1subset = t1[t1 <= .005]
x1subset = x1[0: len(t1subset)]

plt.plot(t1subset,x1subset)
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.title('Sinusoid')
plt.savefig('results/1-3_sinusoid.png')
plt.close()


# Question 2. Combining Sinusoids to generate waveforms with complex spectra 

# 2.1

def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    sum_sines = np.zeros(math.ceil(length_secs*sampling_rate_Hz))
    for k in range(1,10):
        t, sin_k = generateSinusoidal(amplitude, sampling_rate_Hz, (2*k-1)*frequency_Hz, length_secs, phase_radians) 
        sum_sines += sin_k / (2*k-1)

    x = 4 / np.pi * sum_sines
    
    return t, x


# 2.2

t2, x2 = generateSquare(amplitude = 1.0, sampling_rate_Hz = 44100,
                      frequency_Hz = 400, length_secs = 0.5, phase_radians = 0)


# 2.3

t2subset = t2[t2 <= .005]
x2subset = x2[0: len(t2subset)]

plt.plot(t2subset,x2subset)
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.title('Approximation of Square Wave as Sum of 10 Sinusoidals')
plt.savefig('results/2-3_square.png')
plt.show()
plt.close()


# Question 3. Fourier Transform 

# 3.1

def computeSpectrum(x, sample_rate_Hz):
    #fourier transform
    fourier=np.fft.fft(x)
    XAbs_all=np.abs(fourier)
    XRe_all=np.real(fourier)
    XIm_all=np.imag(fourier)
    XPhase_all=np.angle(fourier)
    
    n=x.size
    #list of frequencies
    f=np.arange(0,sample_rate_Hz/2,sample_rate_Hz/n)
    #keep non-redundant part without symmetry
    XAbs = XAbs_all[0:len(f)]
    XPhase = XPhase_all[0:len(f)]
    XRe = XRe_all[0:len(f)]
    XIm = XIm_all[0:len(f)]
    
    return (f, XAbs, XPhase, XRe, XIm)


# 3.2

f1, XAbs1, XPhase1, XRe1, XIm1 = computeSpectrum(x=x1, sample_rate_Hz=44100)
f2, XAbs2, XPhase2, XRe2, XIm2 = computeSpectrum(x=x2, sample_rate_Hz=44100)

# 3.3

plt.subplot(2,1,1)
plt.plot(f1,XAbs1)
plt.title('Magnitude and Phase of Sinusoidal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.subplot(2,1,2)
plt.plot(f1,XPhase1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.savefig('results/3-3_sinusoidal.png')
plt.show()
plt.close()

plt.subplot(2,1,1)
plt.plot(f2,XAbs2)
plt.title('Magnitude and Phase of Sinusoidal Approximation of Square Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.subplot(2,1,2)
plt.plot(f2,XPhase2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (Radians)')
plt.savefig('results/3-3_square.png')
plt.show()
plt.close()


# 3.4 - 3.5 answers in text file

# Question 4. Spectrogram (30 points)

# 4.1

def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    #number of blocks
    N=math.ceil(x.size/hop_size-1)
    #initialize X
    X = np.zeros((block_size,N))
    t = np.zeros(N)
    #populate blocks
    for n in range(0,N):
        start=n*hop_size
        stop=start+block_size
        if n<N-1:
            block=x[start : stop]  
        else: #zero pad the last block to make it the correct length
            lastBlock=x[start:]
            zeroPadding=np.zeros(block_size-len(lastBlock))
            block=np.concatenate((lastBlock,zeroPadding),axis=0) 
        X[:,n]=block
        t[n]=start/sample_rate_Hz
    return (t, X)

#t, X = generateBlocks(x=x1,sample_rate_Hz=44100,block_size=2048,hop_size=1024)


# 4.2

def mySpecgram(x, block_size, hop_size, sampling_rate_Hz, window_type):
    #block the input signal
    time_vector, X = generateBlocks(x,sampling_rate_Hz,block_size,hop_size)
    
    #compute the FFT for each block
    
    #number of blocks
    N = X.shape[1]
    magnitude_spectrogram=np.zeros((math.ceil(block_size/2),N))
    for n in range(0,N):    
        block=X[:,n]
        t=time_vector[n]

        if window_type=='hann':
            w = np.hanning(block_size)
            xw=np.multiply(w,block)
        else: #rectangular window 
            xw=block
        
        freq_vector, XAbs, XPhase, XRe, XIm = computeSpectrum(x=xw, sample_rate_Hz=sampling_rate_Hz)
        magnitude_spectrogram[:,n]=XAbs
    
    #plt.pcolormesh(time_vector, freq_vector, magnitude_spectrogram,shading='auto')
    #plt.colorbar()
    #plt.show()

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if window_type=='rect':
        plt.title('Spectrogram for Sinusoidal Approximation of Square Wave: Rectangular Window')
        plt.specgram(x,window=matplotlib.mlab.window_none,Fs=sampling_rate_Hz)
        plt.savefig('results/4-3_spectrogram_rect.png')
    elif window_type=='hann':
        plt.title('Spectrogram for Sinusoidal Approximation of Square Wave: Hanning Window')
        plt.specgram(x,NFFT=block_size,Fs=sampling_rate_Hz) #default is hanning window
        plt.savefig('results/4-3_hann.png')
    plt.show()
       
    return (freq_vector, time_vector, magnitude_spectrogram)
    
freq_vector, time_vector, magnitude_spectrogram = mySpecgram(x=x2, block_size=2048, hop_size=1024, sampling_rate_Hz=44100, window_type='rect')


# Question 5. BONUS: Sine-Sweep 

def sineSweep(startFreq,endFreq,length_secs,sampling_rate_Hz):
    t = np.linspace(0,length_secs,math.ceil(sampling_rate_Hz*length_secs))
    #print(len(t))
    x = np.zeros(len(t))
    for i in range(0,len(t)-1):
        #linear interpolation between starting and ending frequency
        freq=startFreq+(endFreq-startFreq)*i/len(t)
        x[i] = np.sin(2*np.pi*freq*t[i])
    return t, x
tsweep, xsweep = sineSweep(100,1000,1,44100)

plt.title('Sine Sweep from 100 to 1000 Hz')
plt.specgram(xsweep,Fs=44100) 
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.savefig('results/Bonus.png')
plt.show()

#write as audio file to test
from scipy.io.wavfile import write
write('audio/sinesweep.wav',44100,xsweep.astype(np.float32))





