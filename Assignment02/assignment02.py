#Assignment 2: convolution

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import time as tm

#Question 1: Time Domain Convolution
# 1a. Write a python function y = myTimeConv(x,h) that computes the sample by sample time domain convolution of two signals.
# 'x' and 'h' are the signal and impulse response respectively and must be NumPy arrays.
# 'y' is the convolution output and also must be a NumPy array (single channel signals only)

def myTimeConv(x,h):
    #length of convolution
    y=np.zeros(len(x)+len(h)-1)
    #pad signals with zeros
    x0=np.append(x,np.zeros(len(h)-1))
    h0=np.append(h,np.zeros(len(x)-1))
    #calculate convolution
    for n in range(0,len(y)-1):
        for m in range(0,len(h)-1):
            y[n]+=x0[n-m]*h0[m]
    return y

#1b. If the length of 'x' is 200 and the length of 'h' is 100, the length of y is 199

#1c. In your main script define 'x' as a DC signal of length 200 (constant amplitude of 1) and 'h' as a symmetric triangular signal of length 51
# Add a function call to myTimeConv() in your script to compute 'y_time' as the time-domain convolution of 'x' and 'h' as defined above.
# Plot the result (label the axes appropriately) and save in the results folder

def main():
    #x=signal of length 200, constant amplitude of 1
    x=np.ones(200)
    #h=symmetric triangular signal of length 51 (0 at first and last sample, 1 in the middle)
    h1=np.linspace(0,1,26)
    h2=np.flip(h1)
    h=np.append(h1,h2[1:])
    y_time = myTimeConv(x,h)
    plt.plot(y_time)
    plt.xlabel('t')
    plt.ylabel('Convolution')
    plt.title('Convolution of DC signal and Symmetric Triangular Signal')
    plt.savefig('results/02-convolution.png')
    #plt.show()

    #return y_time

main()

#Question 2. Compare with SciPy convolve()
#Write a function (m, mabs, stdev, time) = CompareConv(x, h) that compares the output of the convolution from both myTimeConv() with the built-in SciPy convolve() function.
#

def loadSoundFile(filename):
    samplerate, data = wavfile.read('./'+filename)
    #if there is more than one channel, keep the left
    if data.ndim>1:
        L = data[:,0]
    else:
        L=data
    #scale to [-1,1]
    x = L/max(L)
    return x
    return data

impulse=loadSoundFile('impulse-response.wav')
piano=loadSoundFile('piano.wav')


def CompareConv(x, h):
    myStart = tm.perf_counter()
    my_conv = myTimeConv(x, h)
    myEnd = tm.perf_counter()
    myTime = myEnd - myStart

    spStart = tm.perf_counter()
    sp_conv = signal.convolve(x, h)
    spEnd = tm.perf_counter()
    spTime = spEnd - spStart

    # time: 2-lengthed array containing the running time of each method (seconds)
    time = np.array([myTime, spTime])

    diff = my_conv - sp_conv
    # m: float of the mean difference of the output compared to convolve()
    m = np.mean(diff)
    # mabs: float of the mean absolute difference of the output compared to convolve()
    mabs = np.mean(abs(diff))
    # stdev: float standard deviation of the difference of the output compared to convolve()
    stdev = np.std(diff)

    return m, mabs, stdev, time

m, mabs, stdev, time = CompareConv(piano,impulse)

print(m)
print(mabs)
print(stdev)
print(time)