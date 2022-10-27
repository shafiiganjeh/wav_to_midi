

class SegLayer(tf.keras.layers.Layer):
    def __init__(self,output_features):
      super(SegLayer, self).__init__()
      
      self.outp = None
      self.Kernels = [32,64,128,256]
      self.L1 = [None]*len(self.Kernels)
      self.L2 = [None]*len(self.Kernels)
      
      self.batch = [None]*len(self.Kernels)
      self.batch_end = None
      self.convend = None
      self.output_layer = None
      
      self.Lstm_Neurons = [16,32,64,128]
      self.Lb1 = [None]*len(self.Kernels)
      self.Lb2 = [None]*len(self.Kernels)
      self.LSTM1 = [None]*len(self.Lstm_Neurons)
      self.LSTM2 = [None]*len(self.Lstm_Neurons)
      
      self.output_features = output_features

    def build(self, inputs):
        j = 0
        for i in self.Kernels:
            self.L1[j] = layers.Conv1D(i, (3), activation='relu',padding = "causal")
            self.L2[j] = layers.Conv1D(i, (3), activation='relu',padding = "causal")
            self.Lb1[j] = layers.Conv1D(int(2*i), (3), activation='relu',padding = "causal")
            self.Lb2[j] = layers.Conv1D(int(2*i), (3), activation='relu',padding = "causal")
            self.batch[j] = layers.BatchNormalization()
            forward_layer = layers.LSTM(self.Lstm_Neurons[j], return_sequences=True,recurrent_dropout=0.15)
            backward_layer = layers.LSTM(self.Lstm_Neurons[j], return_sequences=True, go_backwards=True,recurrent_dropout=0.15)
            self.LSTM1[j] = layers.Bidirectional(forward_layer, backward_layer=backward_layer,dtype=tf.float32)
            j = j+1
        
        self.batch_end = layers.BatchNormalization()
        
        j = 0  
        for i in self.Lstm_Neurons:
            forward_layer = layers.LSTM(i, return_sequences=True,recurrent_dropout=0.15)
            backward_layer = layers.LSTM(i, return_sequences=True, go_backwards=True,recurrent_dropout=0.15) 
            self.LSTM2[j] = layers.Bidirectional(forward_layer, backward_layer=backward_layer,dtype=tf.float32)
            j = j+1
            
        self.convend = layers.Conv1D(512, (2), activation='relu',padding = "causal")
        self.output_layer = layers.Conv1D(self.output_features, (1),padding = "causal")


    def call(self, inputs):
        x = inputs
        y = [None]*len(self.Kernels)
        
        for i in range(len(self.Kernels)):
            x = self.L1[i](x)
            x = self.L2[i](x)
            # x = self.batch[i](x)
            y[i] = self.LSTM1[i](x)
            
        x = self.convend(x)
        x = self.LSTM2[3](x)
        x = layers.Concatenate(axis=2)([x,y[3]])
        x = self.Lb1[3](x)
        x = self.Lb2[3](x)
        
        
        for i in reversed(range(len(self.Kernels)-1)):
            x = self.LSTM2[i](x)
            x = layers.Concatenate(axis=2)([x,y[i]])
            x = self.Lb1[i](x)
            x = self.Lb2[i](x)
            
        # x = self.batch_end(x)
        x = self.output_layer(x)
        x = layers.Activation('sigmoid')(x)
        
        self.outp = K.int_shape(x)
        return x
  
    def compute_output_shape(self, input_shape):
        return self.outp