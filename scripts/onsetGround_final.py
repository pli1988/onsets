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
from onset.onsets import *

def main():

    # get list of files on medleyDB path and load them
    trackList = os.listdir(mdb.AUDIO_PATH)
    trackList = [t for t in trackList if t[0]!='.']
    mtrack_generator = mdb.load_multitracks(trackList)

    # onset parameters
    sr = 44100
    gainWindow = int(sr*0.25)
    temporalThreshold = sr*0.05
    loudnessThreshold = -20
    baseOutPath = './OnsetAnnotations_truth'
    
    # annotation metadata
    g ='Gain Window: ' + str(gainWindow/sr) + 'ms'
    t = 'Temporal Threshold: ' + str(temporalThreshold/sr) + 'ms'
    l = 'Loudness Threshold: ' + str(loudnessThreshold) + 'dB'

    annotationRules = "\n".join([g,t,l])

    # iterate through tracks
    for track in mtrack_generator:

        outPath = os.path.join(baseOutPath,track.track_id+'.jams')

        if not os.path.exists(outPath):
        
            # only compute annotations for tracks without bleed
            if track.has_bleed == False:
                
                ### Load Thngs ###
                
                # data paths
                mixedAudioPath = track.mix_path
                stemsPathList = track.stem_filepaths()
                
                # audio
                mixAudio, stemsAudio = loadAudio(mixedAudioPath,stemsPathList, sr = sr)
                
                # track-level annotations 
                instList = track.stem_instruments
                stemActivations = np.array(track.stem_activations)
                
                ### Gain Estimation ###
                
                # estimate gain per stem
                gain = estimateGain(mixAudio, stemsAudio, gainWindow, int(gainWindow/2))
                
                # weight stem audio by gain
                gainWeightedStem = np.array(gain)*stemsAudio

                ### Onsets ###
                
                # compute onset strength envelopes
                onsetEnvList_stem = [librosa.onset.onset_strength(y=s, sr=sr) for s in gainWeightedStem.T]
                onsetEnv_mix = librosa.onset.onset_strength(y=mixAudio, sr=sr)
                
                # find peaks of individual stem onset envelopes
                stemPeakList = findPeaksOnStem(onsetEnvList_stem)

                # merge stem onset 
                mergedOnset, sourceList, powerStem, powerMix = mergeOnset_greedy(gainWeightedStem, stemPeakList, 
                                                            temporalThreshold = temporalThreshold,
                                                            loudnessThreshold = loudnessThreshold)
                
                # computations for annotation
                onsetTime = [s/sr for s in mergedOnset]
                peakInstrument = [[instList[i] for i in s] for s in sourceList]
                polyphony = [sizePolyphony(stemActivations, t) for t in onsetTime]
                
                j = createAnnotation(onsetTime, peakInstrument, polyphony, annotationRules, powerStem, powerMix)
                
                j.file_metadata.artist = track.artist
                j.file_metadata.title = track.title
                j.file_metadata.duration = len(mixAudio)/sr

                metaData = {}
                metaData['genre'] = track.genre
                metaData['is_instrumental'] = track.is_instrumental
                j.sandbox = metaData

                
                #j.save(outPath)
                print track.track_id

    
if __name__ == '__main__':

    # TODO: take arguments from command line

    main()
                




