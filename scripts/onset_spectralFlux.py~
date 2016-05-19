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

    sr = 44100

    # get list of files on medleyDB path and load them
    trackList = os.listdir(mdb.AUDIO_PATH)

    # multitrack generator
    mtrack_generator = mdb.load_multitracks(trackList)
    baseOutPath = './OnsetEstimation_spectralFlux'

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
                mixAudio = loadAudio(mixedAudioPath,stemsPathList, sr = sr)[0]
                
                ### Onsets ###
                onset_frames = librosa.onset.onset_detect(y=mixAudio, sr=sr)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                

                j = jams.JAMS()

                onset_a = jams.Annotation(namespace='onset')

                for t in onset_times:

                    onset_a.append(time=t, duration=0.0)

                j.annotations.append(onset_a)    

                j.file_metadata.artist = track.artist
                j.file_metadata.title = track.title
                j.file_metadata.duration = len(mixAudio)/sr

                
                j.save(outPath)
                
                print track.track_id
                #print j.dumps(indent=4)





