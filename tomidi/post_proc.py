

def post_proc(onset, length , vel):

    o_confidence = .4
    l_confidence = .05
   
    length_pp = np.array([], dtype=np.float32).reshape(0,88) 
    vel_pp = np.array([], dtype=np.float32).reshape(0,88) 
    onset_pp = np.array([], dtype=np.float32).reshape(0,88) 
    
    for i in range(onset.shape[0]):
        onset_pp = np.vstack([onset_pp,onset[i] ])
        length_pp = np.vstack([length_pp,length[i] ])  
        vel_pp = np.vstack([vel_pp,vel[i] ])  
        
   
    vel_pp = np.round(100*vel_pp +10)
    
    
    sw = True
    post = np.zeros(onset_pp.shape[0]*onset_pp.shape[1])
    
    for i in range(88):
        notes = []
        for j in range(len(length_pp)):
            if sw and (onset_pp[j, i] > o_confidence):
                notes.append(j)
                sw = False
            elif not sw and (length_pp[j, i] < 1-l_confidence) and (onset_pp[j, i] < 1-o_confidence):
                notes.append(j)
                sw = True
            elif not sw and (onset_pp[j, i] > o_confidence):
                notes.append(j-1)
                notes.append(j)

        if len(notes) % 2 > 0:
            notes.append(float(notes[-1]+.1))
        for k in range(int(len(notes)/2)):
            post[int((len(length_pp)*(i))+notes[int(k*2)]):int((len(length_pp)*(i))+notes[int(k*2+1)])] = 1
            
        post = post.reshape(onset_pp.shape,order='F')
        
    return post,vel



def create_midi(post,vel,name):

    frq = 22050
    t = librosa.frames_to_time(np.arange(len(post)), sr=frq,hop_length = 300,n_fft =2024)
    mf = MIDIFile(1)
    track = 0
    channel = 0

    sw = True

    for i in range(88):
        notes = []
        veloc = []
        for j in range(len(post)):
            if sw and (post[j,i] > .2):
                notes.append(t[j])
                veloc.append(vel[j,i])
                sw = False
            elif not sw and (post[j,i] < .2):
                notes.append(t[j])
                sw = True
        if len(notes)%2 > 0:
            notes.append(float(notes[-1]+.1))
        for k in range(int(len(notes)/2)):
            mf.addNote(track, channel, i+21, notes[int(k*2)],notes[int(k*2+1)] - notes[int(k*2)], int(veloc[k]))
            
    with open(name+".mid", 'wb') as outf:
        mf.writeFile(outf)
    
    return 0

