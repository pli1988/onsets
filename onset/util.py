"""
Utility functions for dealing with audio and metadata files 
Author: Peter Li
"""

from __future__ import division

import glob
import os

import librosa
import medleydb as mdb

import numpy as np
import scipy

import itertools
import jams


def loadAudio(mixedAudioPath,stemsPathList, sr = 44100):
    ''' Function to load mix and stem audio for a song.
    
    Input: 
        - mixedAudioPath (str): file path for mixed audio
        - stemsPathList (list): list of file path for stems
        - sr (int): sample rate 
        
    Returns:
        - mixAudio (np.Array): (,length of audio)
        - stemsAudio (np.Array): (num stems, length of audio)
       
    '''

    mixAudio, sr = librosa.load(mixedAudioPath, sr)
    mixAudio = np.array(mixAudio).T

    # load stems

    stems = []

    for path in stemsPathList:

        stems.append(librosa.load(path, sr)[0])

    stemsAudio = np.array(stems).T
    
    return mixAudio, stemsAudio

def estimateGain(mixAudio, stemsAudio, window, stride = None):
    ''' Function to compute gain coefficients using NNLS
    TODO: don't need to return whole matrix
    
    Input: 
        - mixAudio (np.Array): (,length of audio)
        - stemsAudio (np.Array): (num stems, length of audio)
        - window (int): size of window in frames
        
    Returns:
        - gain (np.array): gain coefficients (num stems, length of audio)
       
    '''
    
    if stride == None:
        stride = window/2
        
    gain = np.zeros(stemsAudio.shape)

    gainStart = 0
    gainEnd = window

    regStart = 0
    regEnd = window

    stemTmp = stemsAudio[regStart:regEnd]
    mixTmp = mixAudio[regStart:regEnd]

    gain[gainStart:gainEnd] = scipy.optimize.nnls(stemTmp,mixTmp)[0]

    while regEnd < len(mixAudio):

        gainStart = gainEnd
        gainEnd = gainStart + stride

        regStart = regStart + stride
        regEnd = regEnd + stride

        stemTmp = stemsAudio[regStart:regEnd]
        mixTmp = mixAudio[regStart:regEnd]

        gain[gainStart:gainEnd] = scipy.optimize.nnls(stemTmp,mixTmp)[0]
    
    return gain

def getSurrounding(X,idx, window = 44100*.04):
    ''' Function to get audio segment centered at idx with length window      
    '''
    
    start = max(0, idx - int(window/2))
    end = min(len(X), idx + int(window/2))

    if len(X.shape) == 1:
        return X[start:end]
    else:
        return X[start:end,:]
    
def power_db(S):
    ''' Function to compute power of a signal in dB      
    '''
    
    return 10*np.log10(sum(S**2)/len(S))



def findNearest(activationTimes, t):
    ''' Function to find closest times in activationTimes to a given t 
    
    Input: 
        - activationTimes: list of times from MedleyDB activations
        - t: time of onset
        
    Returns:
        - upper and lower bound on t from activationTimes
       
    '''

    idx = 0

    while idx < len(activationTimes):

        if activationTimes[idx] < t:
            idx += 1
        else:
            return idx-1, idx
        
            break
            
def sizePolyphony(stemActivations, t):
    ''' Function to find the size of polyphony.
    '''
    
    activationTimes = stemActivations[:,0] 

    left, right = findNearest(activationTimes, t)
    
    isActive = stemActivations[[left,right],1:] >= 0.5
    
    return sum(sum(isActive))//2


