#!/usr/bin/env python
# coding: utf-8

# Q1: Correlation

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


#Write a python function x = loadSoundFile(filename) that takes a string and outputs a numpy array of floats - if the file is multichannel you should grab just the left channel.
def loadSoundFile(filename):
    samplerate, data = wavfile.read('./'+filename)
   # print(filename + ': ' + str(data.shape[1]) + ' channels')
   # print('data type' + str(type(data)))
   # print(data)
    #left channel
    L = data[:,0]
    #scale to [-1, 1]
    x = L/max(L)
    #make sure output is floating point 1 dimensional array
   # print('rescaled data' + str(type(x)) + ' dimensions=' + str(data.shape[1]) + ' type=' + str(type(x[0])))
   # print(x)
    return x



#Write a python function z = crossCorr(x, y) where x, y and z are numpy arrays of floats.
def crossCorr(x,y):
    z = signal.correlate(x, y, method='direct')
    return z   


#Create a main function that uses these functions to load the following sound files and compute the correlation between them, plotting the result to file results/01-correlation.png
def runCorrelation(file1,file2):
    x=loadSoundFile(file1)
    y=loadSoundFile(file2)
    z=crossCorr(x,y)
    lags=np.arange(-len(x),len(y)-1)
    
    #plot original signals
    plt.plot(x)
    plt.ylabel(file1)
    plt.show()
    
    plt.plot(y)
    plt.ylabel(file2)
    plt.show()
    
    #plot correlation and save to png
    plt.plot(lags,z)
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.title('Correlation Function for ' + file1 + ' and ' + file2)
    plt.savefig('results/01-correlation.png')
    plt.show()


runCorrelation('drum_loop.wav','snare.wav')


# Q2: Finding snare location

def findSnarePosition(snareFilename,drumloopFilename):
    y=loadSoundFile(snareFilename)
    x=loadSoundFile(drumloopFilename)
    z=crossCorr(x,y)
    #x, y, z, lags = runCorrelation(snareFilename, drumloopFilename)
    lags=np.arange(-len(x),len(y)-1)
    pos=[]
    #find the local maxima above a fixed threshold, identify corresponding sample numbers in drum loop file
    for i in range (1,len(z)-1):
        if (z[i-1] < z[i]) & (z[i] > z[i+1]) & (z[i] > 100):
            pos.append(-lags[i]+len(y))
    pos.sort()
   # print(type(snarePositionList))
    np.savetxt('results/02-snareLocation.txt', pos, fmt='%i')
    return pos
    


findSnarePosition('snare.wav','drum_loop.wav')






