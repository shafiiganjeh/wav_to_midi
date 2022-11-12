

# Main libraries
import tensorflow as tf
from tensorflow.keras import Model ,layers ,Sequential
from tensorflow.keras import backend as K
import numpy as np


# For data processing
from midiutil.MidiFile import MIDIFile
import math
import librosa
import pretty_midi
from sklearn import preprocessing
from midi2audio import FluidSynth


# Models
import seg_model
from model import NoteDET
import post_proc

