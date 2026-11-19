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

import music21

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
        
    # for converting polyphonic audio by basic pitch
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
    
    # for converting monophonic audio by librosa
    def audioToMidi_pyin_v2(self,absolute_fn):
        final_midi_path = self.getOutputPath(absolute_fn,"librosa.mid")
        
        # loading audio, lowering sample rate to achieve better stability and speed
        # also, we dont need the highest frequencies for pitch prediction
        # since around 10 000 herz there is only noise
        y, sr = librosa.load(absolute_fn, sr=22050)
        # normalizing the volume
        y = librosa.util.normalize(y)
        frame_len = 2048
        # setting the the step length , by which we move
        hop_length = 512
        
        frame_dur = hop_length / sr
        
        # using pyIN algorithm for pitch tracking (probabilistic YIN)
        # f0 is core frequency from the frame and voiced_flag is true, 
        # when there was a pitch recognised in the frame  
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('E1'), # specifying range
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=frame_len,
            hop_length=hop_length,
            max_transition_rate=200
        )
        # it assigns times to frequencies
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        
        # calculation of root mean square of energy
        # basically volume
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_length)[0] 
        rms_norm = rms / np.max(rms)
        
        # basically autotuning the notes so that the natural changes in frequency
        # during the one note are not recognized as a note change
        midi_float_array = librosa.hz_to_midi(f0)
        
        
        # onset detection, which detects the natural start of the notes (like pluck or strum...)
        # it returns a list of numbers of frames, in which an onset was detected
        onset_frames = librosa.onset.onset_detect(y=y,
                                                    sr=sr, 
                                                    hop_length=hop_length,
                                                    wait=2, # number of frames algorithm will wait, under it may detect another onset
                                                    delta=0.15 # sensitivity for onset strength
                                                    )
        # we make it a set, for faster lookup (since it is implemented as a hashset)
        onset_frames_set = set(onset_frames)  
        
        # creating the midi
        midi = pretty_midi.PrettyMIDI()
        # setting the instrument to piano
        instrument = pretty_midi.Instrument(program=0)
        
        
        # stores all the midi float values of the current note 
        current_note_buffer = None
        start_time = 0.0
        min_duration_from_onset = 0.05  # ignoring note shorter than min_duration, if it started from onset
        min_duration_from_pitch_change = 0.07  # ignoring note shorter than min_duration, if it started from pitch_change
        
    
        # cycling through the detected frequencies frame by frame
        for i, midi_note in enumerate(midi_float_array):
            time = times[i]
            
            # checking if the current frame had an onset detected
            is_onset = i in onset_frames_set
            
            # flag for tone detection
            # note is flagged as active, when pyIN set the voiced_flag, 
            # or there was a loud sound recognized by rms 
            is_active = voiced_flag[i] and rms_norm[i] > 0.05 and not np.isnan(midi_float_array[i])
            
            # if it is a tone and not NaN (NaN means, that no clear frequency was found)
            if is_active:
                
                # the start of the first note
                if current_note_buffer is None:
                    current_note_buffer = [midi_note]
                    start_time = time
                    
                # if the pitch has changed or there was an onset
                # we append the previous note
                elif abs(np.median(current_note_buffer) - midi_note) > 0.5 or is_onset:
                    duration = time - start_time
                    # if the note is changed by an onset then it may be shorter
                    min_duration = min_duration_from_onset if is_onset else min_duration_from_pitch_change
                    if duration >= min_duration:
                        # we calculate the pitch as a median of all pitches from current note buffer,
                        # to smooth out the note fluctuation
                        note = pretty_midi.Note(
                            velocity=100, pitch=int(np.round(np.median(current_note_buffer))), start=start_time, end=time
                        )
                        instrument.notes.append(note)
                    current_note_buffer = [midi_note]
                    start_time = time
                else:
                    current_note_buffer.append(midi_note)
                    
            
            else:
                # if the audio became silent, or there was no clear core frequency
                if current_note_buffer is not None:
                    duration = time - start_time
                    # if the note is changed by an onset then it may be shorter
                    min_duration = min_duration_from_onset if is_onset else min_duration_from_pitch_change
                    if duration >= min_duration:
                        note = pretty_midi.Note(
                            velocity=100, pitch=int(np.round(np.median(current_note_buffer))), start=start_time, end=time
                        )
                        instrument.notes.append(note)
                    current_note_buffer = None
                    
        # if the recording ended during the final note
        if current_note_buffer is not None:
            # takes the final recorded frequency note and adds the duration of a frame
            end_time = times[-1] + frame_dur
            duration = end_time - start_time
            # if the note is changed by an onset then it may be shorter
            min_duration = min_duration_from_onset if is_onset else min_duration_from_pitch_change
            if duration >= min_duration:
                note = pretty_midi.Note(
                    velocity=100, pitch=int(np.round(np.median(current_note_buffer))), start=start_time, end=end_time
                )
                instrument.notes.append(note)
        # saving the midi
        midi.instruments.append(instrument)
        midi.write(final_midi_path)
        
        return final_midi_path
    
    def midiToMusicXML(self, absolute_fn):
        xml_path = os.path.splitext(absolute_fn)[0]+".musicxml"
        # load the midi 
        score = music21.converter.parse(absolute_fn, quantizePost=False)
        # quantize the midi, allow 4 and 3 divisions of quarter notes 
        # score.quantize(quarterLengthDivisors=(4, 3), inPlace=True)
        
        # in every instrument (part) of the midi and create measures,
        # if they do not already exist
        for part in score.parts:
            try:
                part.makeMeasures(inPlace=True)
            except Exception:
                pass
            
        score.write("musicxml", fp=xml_path)
        return xml_path
            
    # the MIDI synthesis script
    def synthesize(self,accompaniment_events,name_without_ext):
        acc_name = name_without_ext+"_accompaniment"
        output_dir = "./web/backend/generator/tmp"
        mid = events_to_midi(accompaniment_events)
        name_without_ext_with_path = os.path.join(output_dir,acc_name)
        if os.path.exists(name_without_ext_with_path+".mid"):
            os.remove(name_without_ext_with_path+".mid")
            
        midi_file_name = name_without_ext_with_path+".mid"
        mid.save(midi_file_name)
        return midi_file_name
    
    def generateUsingAMT(self, file_location):
        
        absolute_fn = os.path.abspath(file_location)
        midi_absolute_fn = self.audioToMidi_pyin_v2(absolute_fn)
        
        input_events = midi_to_events(midi_absolute_fn)
        length = 10 # time in seconds

        accompaniment_events = generate(
            self.AMT_model, 
            start_time=0, 
            end_time=length, 
            controls=input_events,
            top_p=0.98
        )
        
        midi_file_name = self.synthesize(accompaniment_events,os.path.splitext(os.path.basename(file_location))[0])
        xml_path = self.midiToMusicXML(midi_file_name)
        return midi_file_name, xml_path


gen = MusAcGen()
gen.generateUsingAMT("web/uploads/basa spaky test.mp3")    
    
        
        
