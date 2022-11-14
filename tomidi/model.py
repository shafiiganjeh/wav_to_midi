
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
      
class NoteDET:
    def __init__(self,
                 input_len,
                 input_onset):
        
        #Model parameters
        self.input_mel = 230
        self.LSTMneurons = 128
        self.LSTMneuronsTakt = 128
        
        self.input_len = input_len
        self.input_onset = input_onset
        
        
        self.convoutput_shape = None
        self.model_input = None
        
        self.Vel_model = None
        
        self.convstack_model = None
        self.lstm_model = None
        self.lstm_model_lng = None
        self.model = None
        self.optimizer = None
        self.taktModel = None
        self.convstack_model_vel = None
        
        self.takt_train = None
        self.onset_train = None
        self.length_train = None
        self.vel_train = None
        
        
        self.cce = tf.keras.losses.BinaryCrossentropy()
        
        
        self.build()
        
        
    #builds the model
    def build(self):
        #fully convolutional model
        self.build_convstack()
        
        #segmentation model
        self.TaktModel2()
        
        #onset lstm layer
        self.build_lstm()
        
        #vellocity lstm layer
        self.build_vel_lstm()
        
        #length lstm layer
        self.build_lstm_length()
        
        #the entire model
        self.build_model()
        
    
    def summary(self):
        # self.convstack_model.summary()
        # self.convstack_model_len.summary()
        # self.TaktModel2.summary()
        # self.lstm_model.summary()
        # self.lstm_model_lng.summary()
        self.model.summary()
        # plot_model(self.TaktModel2, to_file='model.png')
            
            
    def TaktModel2(self):
        M = [None]*int(self.input_len/40+4)
        for i in range(int(self.input_len/40+4)):
            M[i] =  tf.constant(np.arange(200)+int(i*40))
        inp = self._Inp()
        x = layers.ZeroPadding2D(padding=(160,0))(inp)
        x = tf.gather(x, M, axis=1, batch_dims=0)
        x = tf.squeeze(x,[4])
        # x = layers.GaussianNoise(.15)(x)
        x = layers.TimeDistributed(SegLayer(1))(x)
        x = tf.squeeze(x,[3])
        self.TaktModel2 = Model(inp,x,name = "Takt_Model")
        
        
    def build_convstack(self):
        inp = self._Inp()
        self.model_input = inp
        x = layers.ZeroPadding2D(padding=(7,0))(inp)
        
        xc = self.convstack(self.isnotTrain)(x)
        xcl = self.convstack(self.isnotTrain)(x)
        xc = tf.squeeze(xc,axis = 2)
        xcl = tf.squeeze(xcl,axis = 2)
        yc = tf.squeeze(inp,axis = -1)
        yc = self.convstackDense()(yc)
        
        self.convstack_model = Model(inp,xc,name = "convstack")
        self.convstack_model_len = Model(inp,xcl,name = "convstack_len")
        self.convstack_model_vel = Model(inp,yc,name = "convstackVel")
        
        self.convoutput_shape = K.int_shape(xc)
        
        
    def build_lstm(self):
        inp = self._Inp2(self.convoutput_shape)
        forward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, recurrent_dropout=0.15)
        backward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, go_backwards=True, recurrent_dropout=0.15)
        
        x = layers.Bidirectional(forward_layer, backward_layer=backward_layer)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(88,1)(x)
        x = layers.Activation('sigmoid')(x)
        self.lstm_model = Model(inp,x,name = "lstm")
        
        
    def build_vel_lstm(self):
        inp = self._Inp3()
        # inp = self._Inp()
        # x = tf.squeeze(inp,axis = -1)
        forward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, recurrent_dropout=0.15)
        backward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, go_backwards=True, recurrent_dropout=0.15)
        
        x = layers.Bidirectional(forward_layer, backward_layer=backward_layer)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(88)(x)
        x = layers.Activation('sigmoid')(x)
        self.Vel_model = Model(inp,x,name = "lstmVel")
        
        
    def build_lstm_length(self):
        inp = self._Inp4(self.convoutput_shape,(self.input_len,self.input_onset))
        forward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, recurrent_dropout=0.15)
        backward_layer = layers.LSTM(self.LSTMneurons, return_sequences=True, go_backwards=True, recurrent_dropout=0.15)
        
        x = layers.Bidirectional(forward_layer, backward_layer=backward_layer,dtype=tf.float32)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(88,1)(x)
        x = layers.Activation('sigmoid')(x)
        self.lstm_model_lng = Model(inp,x,name = "lstm_length")
        
        
    def build_model(self):     
        inp = self._Inp()
        
        xc = self.convstack_model(inp)
        xcl = self.convstack_model_len(inp)
        xcv = self.convstack_model_vel(inp)
        
        xt = self.TaktModel2(inp)
        xt = self.transformTakt(xt)
        
        xo = layers.Concatenate(axis = 2)([xc,xt])
        xo = self.lstm_model(xo)
        
        xl = layers.Concatenate(axis = 2)([xo,xcl])
        xl = self.lstm_model_lng(xl)
        
        xv = self.Vel_model(xcv)
        
        self.model = Model([inp], [xo,xl,xv] , name = "model")



    def _Inp(self):
        return layers.Input(shape = (self.input_len,self.input_mel,1), name = "input_layer_mel")
    
    
    def _Inp2(self,x):
        return layers.Input(shape = (x[1],x[2]+1), name = "input_layer_lstm")
    
    
    def _Inp3(self):
        return layers.Input(shape = (self.input_len,self.input_onset), name = "input_layer_onset")
    
    
    def _Inp4(self,x,y):
        return layers.Input(shape = (x[1],x[2]+y[1]), name = "input_layer_lstm_len")
    
    
    def _Inp5(self):
        return layers.Input(shape = (self.input_len,self.input_onset), name = "input_layer_vel")
    
    
    def convstack(self):
        weigth = tf.keras.initializers.VarianceScaling(scale=2.0,mode='fan_avg',distribution='uniform')
        model = Sequential()
        model.add(layers.Conv2D(48, (5,3), activation='relu',kernel_initializer = weigth))
        model.add(layers.Conv2D(48, (3,3), activation='relu',kernel_initializer = weigth))
        model.add(layers.Conv2D(48, (3,3), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(1, 2),strides=(1, 2), padding = "valid"))
        model.add(layers.Dropout(.25))
        model.add(layers.Conv2D(58, (3,3), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(58, (3,3), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(1, 2),strides=(1, 2), padding = "valid"))
        model.add(layers.Dropout(.25))
        model.add(layers.Conv2D(64, (3,25), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (1,25), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(.4))
        model.add(layers.Conv2D(88, (1,1), activation='relu',kernel_initializer = weigth))
        model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D(pool_size=(1, 6), padding='valid'))
        model.add(layers.Activation('sigmoid'))
        # model.add(layers.Flatten())
        return model
    
    
    def convstackDense(self):
        model = Sequential()
        model.add(layers.Dropout(.1))
        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(.25))
        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(.25))
        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(.25))
        model.add(layers.Dense(88))
        # model.add(layers.Activation('sigmoid'))
        return model
    
    
    def transformTakt(self,par):
        M = [None]*5
        for i in range(5):
            M[i] = tf.stack([par[:,j+4-i,int(40*i):int(40+40*i)] for j in range(25)], axis=-2)
            
        x = tf.stack([M[i] for i in range(5)])
        x = K.mean(x, axis = 0)
        x = layers.Flatten()(x)
        x = .5 + .5*K.tanh(30*(x-.38))
        x = tf.expand_dims(x, axis=-1)
        return x
    
    
    #losses-----------------------------------------------------------
    
    
    def loss_onset(self,x ,xpred):
        xpred = layers.Flatten()(xpred)
        f = layers.Flatten()(self.onset_train.layers[0].input)
        l1 = tf.compat.v1.losses.log_loss(f,xpred,epsilon=1e-09)
        return tf.clip_by_value(l1, clip_value_min=0, clip_value_max=10000)
    
    
    def loss_lng(self,x ,xpred):
        C = self.length_train.layers[2].output
        l1 = tf.compat.v1.losses.log_loss(C,xpred,weights=C*5,epsilon=1e-09)
        l2 = tf.compat.v1.losses.log_loss(x,xpred,weights=1.0,epsilon=1e-09)
        return l1+l2
    
    
    def lossTakt(self,x ,xpred):
        M = [None]*int(self.input_len/40+4)
        for i in range(int(self.input_len/40+4)):
            M[i] =  tf.constant(np.arange(200)+int(i*40))
            
        f = self.takt_train.layers[1].input
        f = tf.keras.layers.ZeroPadding1D(padding=160)(f)
        f = tf.gather(f, M, axis=1, batch_dims=0)
        f = tf.clip_by_value(K.sum(f,axis = -1), clip_value_min=0, clip_value_max=1)
        l1 = tf.compat.v1.losses.log_loss(f,xpred,epsilon=1e-09)
        return l1
    
    
    def loss_vel(self,x ,xpred):
        I = tf.clip_by_value(self.vel_train.layers[3].input, clip_value_min=0, clip_value_max=1)
        f = self.vel_train.layers[4].input
        l1 = tf.compat.v1.losses.mean_squared_error(f,xpred,weights = I)
        return l1


    # compile and train methods-------------------------------------------
    
    def model_reconstruct(self,data):
        rec = self.model.predict(data)
        return rec
        
        
    def takt_train(self, x,val,epoch,optm):  
        inp = self._Inp()
        inp2 = self._Inp3()
        inp3 = self._Inp5()
        xm = self.TaktModel2(inp)
        self.takt_train = Model([inp,inp2,inp3], xm , name = "Takt_model")
        
        M = [None]*int(self.input_len/40+4)
        for i in range(int(self.input_len/40+4)):
            M[i] =  tf.constant(np.arange(200)+int(i*40))
        f = inp2
        f = tf.keras.layers.ZeroPadding1D(padding=160)(f)
        f = tf.gather(f, M, axis=1, batch_dims=0)
        f = tf.clip_by_value(K.sum(f,axis = -1), clip_value_min=0, clip_value_max=1)
        
        self.takt_train.add_metric(tf.keras.metrics.binary_accuracy( f,xm ,threshold=0.3), name="bin.acc", aggregation="mean")
        self.takt_train.summary()
        self.takt_train.compile(optimizer = optm, loss=self.lossTakt)
        self.takt_train.fit(x,validation_data=val, epochs=epoch, verbose=1,shuffle = True)
        
        
    def length_train(self, x,val,epoch,optm):  
        inp = self._Inp()
        inp2 = self._Inp3()
        inp3 = self._Inp5()
        
        xc = self.convstack_model_len(inp)
        xc = layers.Concatenate(axis = 2)([xc,inp2])
        xc = self.lstm_model_lng(xc)
        
        self.length_train = Model([inp,inp2,inp3], xc , name = "length_train")
        self.length_train.summary()
        self.length_train.compile(optimizer = optm, loss=self.loss_lng,metrics=[tf.keras.metrics.BinaryAccuracy()])
        self.length_train.fit(x,validation_data=val, epochs=epoch, verbose=1,shuffle = True)
        
        
    def onset_train(self, x,val,epoch,optm):  
        inp = self._Inp()
        inp2 = self._Inp3()
        inp3 = self._Inp5()
        xi = self.convstack_model(inp)
        
        xt = K.sum(inp2,axis = -1)
        xt = tf.clip_by_value(xt, clip_value_min=0, clip_value_max=1)
        xt = tf.expand_dims(xt, axis = 2)

        xo = layers.Concatenate(axis = 2)([xi,xt])
        xo = self.lstm_model(xo)
        
        self.onset_train = Model([inp,inp2,inp3], xo , name = "onset_train")
        self.onset_train.summary()
        self.onset_train.add_metric(tf.keras.metrics.binary_accuracy( self.onset_train.layers[0].input,xo ,threshold=0.5), name="bin.acc", aggregation="mean")
        self.onset_train.compile(optimizer = optm, loss=self.loss_onset)
        self.onset_train.fit(x,validation_data=val, epochs=epoch, verbose=1,shuffle = True)
        
        
    def vel_train(self, x,val,epoch,optm):  
        inp = self._Inp()
        inp2 = self._Inp3()
        inp3 = self._Inp5()
        
        xc = tf.squeeze(inp,axis = -1)
        xc = self.convstack_model_vel(xc)
        xc = self.Vel_model(xc)
        
        self.vel_train = Model([inp,inp2,inp3], xc , name = "vel_train")
        self.vel_train.compile(optimizer = optm, loss=self.loss_vel)
        self.vel_train.summary()
        
        self.vel_train.fit(x,validation_data=val, epochs=epoch, verbose=1,shuffle = True)
        
        
    def model_save(self, folder = "."):
        self._create_folder(folder)
        parameters = [self.input_len,
                      self.input_onset]
        
        save_path = os.path.join(folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
        save_path = os.path.join(folder, "Modelweights.h5")
        self.model.save_weights(save_path)
        tf.saved_model.save(self.model, folder)
        
    
    def load_weights(self, folder):
        W_list =[os.path.join(folder, "Modelweights.h5")]
        self.model.load_weights(W_list[0])


    @classmethod
    def model_load(cls, folder="."):
        parameters_path = os.path.join(folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        var = NoteDET(*parameters)
        var.load_weights(folder)
        return var
        
        
    def _create_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
