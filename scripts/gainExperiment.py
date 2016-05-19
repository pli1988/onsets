# compute SNR for two different methods to estimate gain
# regression using signals is much better

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


if __name__ == '__main__':

    # get list of files on medleyDB path and load them
    trackList = os.listdir(mdb.AUDIO_PATH)
    trackList = [t for t in trackList if t[0]!='.']

    # multitrack generator
    mtrack_generator = mdb.load_multitracks(trackList)

    sr = 44100
    gainWindow = int(sr*0.25)
    temporalThreshold = sr*0.05
    loudnessThreshold = -20

    SNR = []

    for track in mtrack_generator:

    
        # only compute annotations for tracks without bleed
        if track.has_bleed == False:
            print track.track_id
    
            # data paths
            mixedAudioPath = track.mix_path
            stemsPathList = track.stem_filepaths()
            
            # audio
            mixAudio, stemsAudio = loadAudio(mixedAudioPath,stemsPathList, sr = sr)
            
            # compute envelope
            mixAudioEnvelope = np.array(computeEnvelope(mixAudio))
            stemsAudioEnvelope = np.array([computeEnvelope(s) for s in stemsAudio.T]).T 
            
            ### Gain Estimation ###
            
            # estimate gain per stem
            gain_signal = estimateGain(mixAudio, stemsAudio, gainWindow, int(gainWindow/2))
            gain_abs = estimateGain(np.abs(mixAudio), np.abs(stemsAudio), gainWindow, int(gainWindow/2))
            gain_env = estimateGain(mixAudioEnvelope, stemsAudioEnvelope, gainWindow, int(gainWindow/2))
            
            # weight stem audio by gain
            
            snr_signal = computeSNR(gain_signal, stemsAudio, mixAudio)
            snr_abs = computeSNR(gain_abs, stemsAudio, mixAudio)
            snr_env = computeSNR(gain_env, stemsAudio, mixAudio)
            
            SNR.append([snr_signal, snr_abs, snr_env]) 
    
    np.save('gainExp', np.array(SNR))




