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
        # self.fs = midi2audio.FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2')


    def audioToMidi_basicPitch(self,absolute_fn):
        output_dir = "./web/backend/generator/tmp"
        
        
        # filename without path
        base_name = os.path.basename(absolute_fn) 
        # remove extension
        name_without_ext = os.path.splitext(base_name)[0] 
        
        # added basic pitch suffix
        expected_midi_name = f"{name_without_ext}_basic_pitch.mid"
        
        # join with output direction
        final_midi_path = os.path.join(output_dir, expected_midi_name)
        
        # absolute path
        final_midi_path = os.path.abspath(final_midi_path)
        if os.path.exists(final_midi_path):
            os.remove(final_midi_path)
        
        predict_and_save(
            [absolute_fn],
            output_dir,
            True,
            False,
            False,
            False,
            ICASSP_2022_MODEL_PATH)
        return final_midi_path
    
    # the MIDI synthesis script
    def synthesize(self,accompaniment_events,name_without_ext):
        acc_name = name_without_ext+"_accompaniment"
        output_dir = "./web/backend/generator/tmp"
        mid = events_to_midi(accompaniment_events)
        name_without_ext_with_path = os.path.join(output_dir,acc_name)
        if os.path.exists(name_without_ext_with_path+".mid"):
            os.remove(name_without_ext_with_path+".mid")
            
        mid.save(name_without_ext_with_path+".mid")
        return name_without_ext_with_path
    
    def generateUsingAMT(self, file_location):
        
        absolute_fn = os.path.abspath(file_location)
        midi_absolute_fn = self.audioToMidi_basicPitch(absolute_fn)
        events = midi_to_events(midi_absolute_fn)
        length = 10 # time in seconds

        accompaniment_events = generate(
            self.AMT_model, 
            start_time=0, 
            end_time=length, 
            controls=events,
            top_p=0.98
        )
        
        name_without_ext_with_path = self.synthesize(accompaniment_events,os.path.splitext(os.path.basename(file_location))[0])
        
    
        
        
