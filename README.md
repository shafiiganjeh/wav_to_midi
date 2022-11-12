## <p style="text-align:center"> WavToMIDI </p>

### Description:
This project aims for automatic music transcription by using time series segmentation.

<br/> <br/>
### Current State:
Currently the model is able to transcribe piano music and output a midi. The quality of the transcription depends highly on the audio quality. At the moment the model is not designed to handle multiple instruments (other than the piano) or noisy audio.

Here are some examples:

<video width="320" height="240" controls>
  <source src="/home/borz/Desktop/git/wav_to_midi/resources/mozart.mp4" type="video/mp4">
</video>

[Original video](https://www.youtube.com/watch?v=Jo7ox5ptjGY)

<video width="320" height="240" controls>
  <source src="/home/borz/Desktop/git/wav_to_midi/resources/mario.mp4" type="video/mp4">
</video>

[Original video](https://www.youtube.com/watch?v=SdqKEHqt94g)

You can also test the model yourself with the following 
[google collab](https://colab.research.google.com/drive/1mvlBuz1tMfBfPryfPbepGqmLCfhL-Uke#scrollTo=2S8a6_o1P5tE).

<br/> <br/>
### How it works:
The audio track is transformed into a mel-spectrogram. The spectrogram is then fed into four different models. A segmentation model that splits the track between two note-onsets respectively, with output 1 at the border of the segments and 0 otherwise. 
An onset model and a length model based on the fully convolutional model from this [paper](https://archives.ismir.net/ismir2016/paper/000179.pdf) (with some modifications). 
And a standalone dense model, from the same paper, for Volume estimation.

The output of the segmentation model is concatenated together with the output of the onset model and given into a RNN (for the RNN a Bidirectional LSTM is used) that predicts the onset notes with respect to the time. This prediction is then again concatenated with the output of the length model and given to another RNN that predicts how long an onset note is being played. 

Together with the volume estimation a midi of the audio can be created.

A graphical scetch of the model can be found in the resource folder.

<br/> <br/>
### Why concentrate on segmentation:
Detecting what note is being played at a certain frame is not a very difficult task. By using segmentation we can transform the problem of music transcription (which is a much harder problem) into just a frame by frame note detection problem. Even multi instrument transcription can be reduced to frame wise transcription this way by segmenting the audio data not just by onset but also by instrument and feed different frames to different models.

<br/> <br/>
### How to use:
Main files are in the tomidi folder. Training samples can be generated with create_sample. The input should be a wav-file of the Piano recording and a corresponding midi file, create_sample will then returns a tensorflow dataset.

The model can be trained with the NoteDET class:

<br/> <br/>
NoteDET(sample_length,keys)

<pre>
    sample_length: int
    
       length of the training sample (should be at least 200)
       
    keys: 
    
       number of piano keys (usually 88)
</pre>
<br/> <br/>
The training routines are following:

NoteDET.takt_train(data_set,validation_set,epoch,optimizer)
NoteDET.length_train(data_set,validation_set,epoch,optimizer)
NoteDET.onset_train(data_set,validation_set,epoch,optimizer)
NoteDET.vel_train(data_set,validation_set,epoch,optimizer)

<pre>
    data_set: batched dataset of the form (tf.TensorSpec(shape=(sample_length, 230, 1), dtype=tf.float32), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32)), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32)
    
       training data
       
    validation_set: batched dataset of the form (tf.TensorSpec(shape=(sample_length, 230, 1), dtype=tf.float32), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32)), tf.TensorSpec(shape=(sample_length, keys), dtype=tf.float32)
    
       validation data 
       
    epoch: int
    
       number of epochs
       
    optimizer
    
       tensorflow optimizer
</pre>

<br/> <br/>
You can save and load the model with NoteDET.model_save(*path) and NoteDET.model_load(*path). 
To predict a onset, length and velocities use NoteDET.model_reconstruct(data_set).
You can further use the functions provided post_proc to creat a midi.

A pretrained model can be found in the fullmodel folder.









