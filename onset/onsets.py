"""
Onsets from multi-track data
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

from onset.util import *

def findPeaksOnStem(onsetEnvList_stem,pre_max = 3,post_max = 3,pre_avg = 3,post_avg = 3,wait = 1):
    ''' peak finding in onset strength envelope   

    Returns:
        stemPeakList: List of List of onset sample indices per stem
    '''
    
    stemPeakList = []

    for stemOnsetEnvelope in onsetEnvList_stem:

        onsetFrameIdx = librosa.util.peak_pick(stemOnsetEnvelope, pre_max, post_max, pre_avg, post_avg,  stemOnsetEnvelope.mean(), wait)
        onsetSampleIdx = [librosa.core.frames_to_samples(idx, n_fft=2048, hop_length=512)[0] for idx in onsetFrameIdx]

        stemPeakList.append(onsetSampleIdx)
        
    return stemPeakList

def appendOnsetSource(onsetList):
    ''' Function to append source to StemPeakList
    
    Input: 
        - [[1,2],[3,4]]
        
    Returns:
        - [[(1,0),(2,0)],[(3,1),(4,1)]]
       
    '''

    return [[(x,i) for x in onsetList[i]]for i in range(len(onsetList))]


def isMasked_loudness(X, onsetIdx, sourceIdx, threshold = -20):
    # Compare energy of onsets with the total mix
    # returns isMasked boolean for each stem, power of onsets and mix 
       
    X_section = getSurrounding(X,onsetIdx)
    combined_section = X_section.sum(1)

    powerMix = power_db(combined_section)

    sourceMask = [power_db(s) - powerMix < threshold for s in X_section.T[sourceIdx]] 

    idxNotMasked = [i for i,j in zip(sourceIdx, sourceMask) if not j]
    powerSource = power_db(X_section.T[idxNotMasked].sum(0))
    
    return sourceMask, powerSource, powerMix  

def createAnnotation(onsetTime, peakInstrument, polyphony, annotationRules,powerStem, powerMix):

    jam = jams.JAMS()

    onset_a = jams.Annotation(namespace='onset')
    onset_a.annotation_metadata.annotation_rules = annotationRules

    for t,i,p,s,m in zip(onsetTime,peakInstrument, polyphony,powerStem, powerMix):

        dataDict = {'onsetSource': i, 'polyphony': p, 'powerStem' : s, 'powerMix': m}

        onset_a.append(time=t, duration=0.0, value = dataDict)

    jam.annotations.append(onset_a)    
    
    return jam

def mergeOnset_greedy(gainWeightedStem, onsetList, temporalThreshold = 44100*0.05, loudnessThreshold = -20):
    
    ''' A function to merge sets of onset from multiple sources
    Accounts for temporal and loudness masking
 
    '''
    
    #append source id to onsets
    onsetSourceList = appendOnsetSource(onsetList)

    flat_onset = [item for sublist in onsetSourceList for item in sublist]
    combinedOnset = sorted(flat_onset, key = lambda x: x[0])

    #list of unique onsets
    keys = sorted(list(set([s[0] for s in combinedOnset])))

    # dictionary of onsets
    d = {key:[] for key in keys}

    for (onsetIdx, sourceID) in combinedOnset:
        d[onsetIdx].append(sourceID)

    mergedOnset = []
    sourceList = []
    powerStem = []
    powerMix = []
    
    # this should be rewritten to be faster
    for onset in keys:

        if len(mergedOnset) == 0:

            #check loudness
            sourceMask, pSource, pMix = isMasked_loudness(gainWeightedStem, onset, d[onset], threshold = loudnessThreshold)

            if any([x == False for x in sourceMask]):

                idxNotMasked = [i for i,x in enumerate(sourceMask) if not x]
                sourceIdx = [d[onset][i] for i in idxNotMasked]

                mergedOnset.append(onset)
                sourceList.append(sourceIdx)
                powerStem.append(pSource)
                powerMix.append(pMix)          

        else:

            # check temporal
            if onset - mergedOnset[-1] > temporalThreshold:

                #check loudness
                sourceMask, pSource, pMix = isMasked_loudness(gainWeightedStem, onset, d[onset], threshold = loudnessThreshold)

                if any([x == False for x in sourceMask]):

                    idxNotMasked = [i for i,x in enumerate(sourceMask) if not x]
                    sourceIdx = [d[onset][i] for i in idxNotMasked]
                    
                    mergedOnset.append(onset)
                    sourceList.append(sourceIdx)
                    powerStem.append(pSource)
                    powerMix.append(pMix)        

    return mergedOnset, sourceList, powerStem, powerMix
