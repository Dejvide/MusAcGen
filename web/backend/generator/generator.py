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

import librosa
import pretty_midi
import numpy as np
import scipy.signal


class MusAcGen:
    def __init__(self):
        # load an anticipatory music transformer
        self.AMT_model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').cuda()
        
        self.AMT_SMALL_MODEL = 'stanford-crfm/music-small-800k'     # faster inference, worse sample quality
        self.AMT_MEDIUM_MODEL = 'stanford-crfm/music-medium-800k'   # slower inference, better sample quality
        self.AMT_LARGE_MODEL = 'stanford-crfm/music-large-800k'     # slowest inference, best sample quality

        self.OUTPUT_DIR = "./web/backend/generator/tmp"

        # a MIDI synthesizer
        # self.fs = midi2audio.FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2')

    def getOutputPath(self,input_fn,suffix):
        # filename without path
                base_name = os.path.basename(input_fn) 
                # remove extension
                name_without_ext = os.path.splitext(base_name)[0] 
                
                # added basic pitch suffix
                expected_midi_name = f"{name_without_ext}_{suffix}"
                
                # join with output direction
                final_midi_path = os.path.join(self.OUTPUT_DIR, expected_midi_name)
                
                # absolute path
                return os.path.abspath(final_midi_path)
        
    # for converting polyphonic audio
    def audioToMidi_basicPitch(self,absolute_fn):
        
        final_midi_path = self.getOutputPath(absolute_fn,"basic_pitch.mid")
        if os.path.exists(final_midi_path):
            os.remove(final_midi_path)
        
        predict_and_save(
            [absolute_fn],
            self.OUTPUT_DIR,
            True,
            False,
            False,
            False,
            ICASSP_2022_MODEL_PATH)
        return final_midi_path
    
    # for converting monophonic audio
    def audioToMidi_librosa(self,absolute_fn):
        final_midi_path = self.getOutputPath(absolute_fn,"librosa.mid")
        
        # loading audio, lowering sample rate to achieve better stability and speed
        # also, we dont need the highest frequencies for pitch prediction
        # since around 10 000 herz there is only noise
        y, sr = librosa.load(absolute_fn, sr=22050)
        # normalizing the volume
        y = librosa.util.normalize(y)
        # setting the the step length , by which we move
        hop_length = 512
        
        # using pyIN algorithm for pitch tracking (probabilistic YIN)
        # f0 is core frequency from the frame and voiced_flag is true, 
        # when there was a pitch recognised in the frame  
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C1'), # specifying range
            fmax=librosa.note_to_hz('C8'),
            sr=sr,
            frame_length=2048,
            hop_length=hop_length
        )
        # it assigns times to frequencies
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # calculation of root mean square of energy
        # basically volume
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0] 
        rms_norm = rms / np.max(rms)
        
        # basically autotuning the notes so that the natural changes in frequency
        # during the one note are not recognized as a note change
        midi_array = np.zeros_like(f0, dtype=int)                                              # <--- NOVÉ
        for i, freq in enumerate(f0):                                               # <--- NOVÉ
            if not np.isnan(freq):                                                  # <--- NOVÉ
                midi_array[i] = int(round(librosa.hz_to_midi(freq)))
        
        # deletes the miniature jumps (jitter) 
        # it looks at neighboring kernel_size-1 frames and takes the median
        midi_array = scipy.signal.medfilt(midi_array, kernel_size=5)
        
        # onset detection, which detects the natural start of the notes (like pluck or strum...)
        # it returns a list of numbers of frames, in which an onset was detected
        onset_frames = librosa.onset.onset_detect(y=y,
                                                  sr=sr, 
                                                  hop_length=hop_length,
                                                  wait=2 # number of frames it will wait, under it may detect another onset
                                                  )
        # we make it a set, for faster lookup (since it is implemented as a hashset)
        onset_frames_set = set(onset_frames)  
        
        # creating the midi
        midi = pretty_midi.PrettyMIDI()
        # setting the instrument to piano ( program=0 )
        # since the instrument is usually not important for accompaniment generators 
        instrument = pretty_midi.Instrument(program=0)
        
        current_pitch = None
        start_time = 0.0
        min_duration = 0.02  # ignoring notes shorter than min_duration
        
        # cycling through the detected frequencies frame by frame
        for i, midi_note in enumerate(midi_array):
            time = times[i]
            
            # checking if the current frame had an offset detected
            is_onset = i in onset_frames_set
            
            # flag for tone detection
            # note is flagged as active, when pyIN set the voiced_flag, 
            # or there was a loud sound recognized by rms 
            is_active = (voiced_flag[i] or rms_norm[i] > 0.05) and (midi_note > 0)
            
            # if it is a tone and not NaN (NaN means, that no clear frequency was found)
            if is_active:
                
                # the start of the first note
                if current_pitch is None:
                    current_pitch = midi_note
                    start_time = time
                    
                # if the pitch has changed or there was an onset
                # we append the previous note
                elif current_pitch != midi_note or is_onset:
                    if time - start_time >= min_duration:
                        note = pretty_midi.Note(
                            velocity=100, pitch=current_pitch, start=start_time, end=time
                        )
                        instrument.notes.append(note)
                    current_pitch = midi_note
                    start_time = time
                    
            
            else:
                # if the audio became silent, or there was no clear core frequency
                if current_pitch is not None:
                    if time - start_time >= min_duration:
                        note = pretty_midi.Note(
                            velocity=100, pitch=current_pitch, start=start_time, end=time
                        )
                        instrument.notes.append(note)
                    current_pitch = None

        # saving the midi
        midi.instruments.append(instrument)
        midi.write(final_midi_path)
        
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
        midi_absolute_fn = self.audioToMidi_librosa(absolute_fn)
        
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
        
    
        
        
