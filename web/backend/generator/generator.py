import sys,time, os

import midi2audio
import transformers
from transformers import AutoModelForCausalLM

from IPython.display import Audio

from anticipation import ops
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *

import tensorflow as tf
from basic_pitch.inference import predict, Model
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH


class MusAcGen:
    def __init__(self):
        # load an anticipatory music transformer
        self.AMT_model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').cuda()
        
        self.AMT_SMALL_MODEL = 'stanford-crfm/music-small-800k'     # faster inference, worse sample quality
        self.AMT_MEDIUM_MODEL = 'stanford-crfm/music-medium-800k'   # slower inference, better sample quality
        self.AMT_LARGE_MODEL = 'stanford-crfm/music-large-800k'     # slowest inference, best sample quality

        

        # a MIDI synthesizer
        fs = midi2audio.FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2')


    def audioToMidi(self,file_location):
        absolute_fl = os.path.abspath(file_location)
        predict_and_save(
            [absolute_fl],
            "./web/backend/generator/tmp",
            True,
            False,
            False,
            False,
            ICASSP_2022_MODEL_PATH)
    
    # the MIDI synthesis script
    def synthesize(self,fs, tokens):
        mid = events_to_midi(tokens)
        mid.save('tmp.mid')
        fs.midi_to_audio('tmp.mid', 'tmp.wav')
        return 'tmp.wav'
    
    def generateUsingAMT(self, file_location):
        
        relative_fl = "../"+file_location
        length = 10 # time in seconds
        events = generate(self.AMT_model, start_time=0, end_time=length, top_p=.98)
        mid = events_to_midi(events)
        mid.save('generated.mid')
        
        
