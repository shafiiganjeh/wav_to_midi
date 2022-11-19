

FluidSynth(sample_rate=44100)
min_max_scaler = preprocessing.MinMaxScaler()
pretty_midi.pretty_midi.MAX_TICK = 1e10


def create_sample(input_wav,input_midi):
    
    #sample slice length 
    input_width = 1000
    
    #shift length of each slice
    shift = 1000
    
    
    size = 0
    x1c = np.array([], dtype=np.float32).reshape(0,230,1)
    x2c = np.array([], dtype=np.float32).reshape(0,88)
    xv = np.array([], dtype=np.float32).reshape(0,88)
    yc = np.array([], dtype=np.float32).reshape(0,88)
    
    
    
    
    #convert the data audio into a mel-spectrogram
    scale, sr = librosa.load(input_wav, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr,  n_mels=230, fmin=0, hop_length = 300,win_length=2024)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    t = librosa.frames_to_time(np.arange(len(mel_spectrogram[1])), sr=sr,hop_length = 300,n_fft =2024 )
   
    log_mel_spectrogram = log_mel_spectrogram.T
    
    X1 = log_mel_spectrogram
    X1 = min_max_scaler.fit_transform(X1.reshape(-1, X1.shape[-1])).reshape(X1.shape)
    X1 = np.expand_dims(X1, axis=2)
    X1 = np.asarray(X1, dtype=np.float32)
    
    
    
    
    #extract onset notes and velocities from the midi
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    midi_list = []
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity])   
    
    midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))


    X2 = np.zeros((X1.shape[0],88), dtype=np.float32)
    XV = np.zeros((X1.shape[0],88), dtype=np.float32)
    Y = np.zeros((X1.shape[0],88), dtype=np.float32)
    
    for row in midi_list:
        difference_start = np.absolute(t-row[0])
        index_start = difference_start.argmin()
        difference_End = np.absolute(t-row[1])
        index_End = difference_End.argmin()
        X2[index_start:index_start+2,int(row[2])] = 1
        Y[index_start:index_End,int(row[2])] = 1
        XV[index_start:index_start+2,int(row[2])] = row[3]
        
    try:
        x1c = np.vstack([x1c, X1])
        x2c = np.vstack([x2c, X2])
        xv = np.vstack([xv, XV])
        yc = np.vstack([yc, Y])
    except Exception:
        print("error")
        
        
        
        
        
    #create slices
    x1c = np.vstack([x1c, np.zeros((input_width,230), dtype=np.float32).reshape(input_width,230,1)])
    x2c = np.vstack([x2c, np.zeros((input_width,88), dtype=np.float32)])
    xv = np.vstack([xv, np.zeros((input_width,88), dtype=np.float32)])
    yc = np.vstack([yc, np.zeros((input_width,88), dtype=np.float32)])
    shif = np.arange(0, math.floor(len(x1c)/input_width)-1, 1, dtype=int)

    X1 = np.stack([x1c[slice(shift*i, input_width+shift*i)] for i in shif], axis=0)
    X2 = np.stack([x2c[slice(shift*i, input_width+shift*i)] for i in shif], axis=0)
    X3 = np.stack([xv[slice(shift*i, input_width+shift*i)] for i in shif], axis=0)
    Y = np.stack([yc[slice(shift*i, input_width+shift*i)] for i in shif], axis=0)
    
    

    return tf.data.Dataset.from_tensor_slices(((X1,X2,X2), Y))
    

