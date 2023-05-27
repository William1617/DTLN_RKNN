
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D

import tensorflow as tf

import tf2onnx

                
class DTLN_model():
    '''
    Class to create and train the DTLN model
    '''
    
    def __init__(self):
        '''
        Constructor
        '''

        # empty property for the model
        self.model = []
        
        self.activation = 'sigmoid'
        self.numUnits = 128

        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25

        self.encoder_size = 256
        
    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]
    
    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer. The layer
        calculates the rFFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # expanding dimensions
        frame = tf.expand_dims(x, axis=1)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frame)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

 
        
    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        
        # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)
    
        

    def seperation_kernel(self, mask_size, x, stateful=False):
        '''
        Method to create a separation kernel. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''

        # creating num_layer number of LSTM layers
        for idx in range(2):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            # using dropout between the LSTM layer for regularization 
            if idx<(1):
                x = Dropout(self.dropout)(x)

        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        # returning the mask
        return mask
    
    def seperation_kernel_with_states(self, mask_size, x, in_h1,in_c1,in_h2,in_c2):
        '''
        Method to create a separation kernel, which returns the LSTM states. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''
        
    
        # creating num_layer number of LSTM layers
       
        in_state = [in_h1, in_c1]
        x, out_h1, out_c1 = LSTM(self.numUnits, return_sequences=True, unroll=False, return_state=True)(x, initial_state=in_state)
        # using dropout between the LSTM layer for regularization 
        x = Dropout(self.dropout)(x)
        in_state = [in_h2, in_c2]
        x, out_h2, out_c2 = LSTM(self.numUnits, return_sequences=True, unroll=False, return_state=True)(x, initial_state=in_state)
    
       
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        # returning the mask and states
        return mask, out_h1,out_c1,out_h2,out_c2

    def build_DTLN_model(self, norm_stft=False):
        '''
        Method to build and compile the DTLN model. The model takes time domain 
        batches of size (batchsize, len_in_samples) and returns enhanced clips 
        in the same dimensions. As optimizer for the Training process the Adam
        optimizer with a gradient norm clipping of 3 is used. 
        The model contains two separation cores. The first has an STFT signal 
        transformation and the second a learned transformation based on 1D-Conv 
        layer. 
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(None, None))
        # calculate STFT
        mag,angle = Lambda(self.stftLayer)(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1 = self.seperation_kernel ((self.blockLen//2+1), mag_norm)
        # multiply mask with magnitude
        estimated_mag = Multiply()([mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frames_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(self.encoder_size, encoded_frames_norm)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        # create waveform with overlap and add procedure
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)

        
        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # show the model summary
        print(self.model.summary())
        
    def build_DTLN_model_stateful(self, norm_stft=False):
        '''
        Method to build stateful DTLN model for real time processing. The model 
        takes one time domain frame of size (1, blockLen) and one enhanced frame. 
         
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(1, self.blockLen))
        # calculate STFT
        mag,angle = Lambda(self.fftLayer)(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1 = self.seperation_kernel((self.blockLen//2+1), mag_norm, stateful=True)
        # multiply mask with magnitude
        estimated_mag = Multiply()([mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frames_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(self.encoder_size, encoded_frames_norm, stateful=True)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frame = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        # create the model
        self.model = Model(inputs=time_dat, outputs=decoded_frame)
        # show the model summary
        print(self.model.summary())
        
    
    def create_onnx_model(self, weights_file, target_name):
        
        # check for type
        if weights_file.find('_norm_') != -1:
            norm_stft = True
            num_elements_first_core = 2 +2 * 3 + 2
        else:
            norm_stft = False
            num_elements_first_core = 2 * 3 + 2
        # build model    
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        # load weights
        self.model.load_weights(weights_file)
        
        #### Model 1 ##########################
        mag = Input(batch_shape=(1, 1, (self.blockLen//2+1)))
        inh1_1 = Input(batch_shape=(1, self.numUnits))
        inc1_1 = Input(batch_shape=(1, self.numUnits))
        inh2_1 = Input(batch_shape=(1, self.numUnits))
        inc2_1 = Input(batch_shape=(1, self.numUnits))
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1, outh1_1,outc1_1,outh2_1,outc2_1 = self.seperation_kernel_with_states( (self.blockLen//2+1), 
                                                    mag_norm, inh1_1,inc1_1,inh2_1,inc2_1)
        nnout1=Multiply()([mag, mask_1]) 
        
        model_1 = Model(inputs=[mag, inh1_1,inc1_1,inh2_1,inc2_1], outputs=[nnout1, outh1_1,outc1_1,outh2_1,outc2_1])
        
        #### Model 2 ###########################
        
        estimated_frame_1 = Input(batch_shape=(1, 1, self.blockLen))
        inh1_2 = Input(batch_shape=(1, self.numUnits))
        inc1_2 = Input(batch_shape=(1, self.numUnits))
        inh2_2 = Input(batch_shape=(1, self.numUnits))
        inc2_2 = Input(batch_shape=(1, self.numUnits))
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frame_1)
        encoded_frames=tf.expand_dims(encoded_frames,1)

        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
     
      #  estimated_frame_2 = Input(batch_shape=(1, 1, self.blockLen))
        encoded_frames=keras.layers.Reshape((1,-1))(encoded_frames)
        encoded_frames_norm=keras.layers.Reshape((1,-1))(encoded_frames_norm)
    
        mask_2, outh1_2,outc1_2,outh2_2,outc2_2 = self.seperation_kernel_with_states(self.encoder_size, encoded_frames_norm, inh1_2,inc1_2,inh2_2,inc2_2)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frame = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        
        model_2 = Model(inputs=[estimated_frame_1, inh1_2,inc1_2,inh2_2,inc2_2], 
                        outputs=[decoded_frame, outh1_2,outc1_2,outh2_2,outc2_2])
        
        # set weights to submodels
        weights = self.model.get_weights()
        model_1.set_weights(weights[:num_elements_first_core])
        model_2.set_weights(weights[num_elements_first_core:])

        tf2onnx.convert.from_keras(model_1,output_path=target_name+'_1.onnx',opset=12)
        tf2onnx.convert.from_keras(model_2,output_path=target_name+'_2.onnx',opset=12)
              
        print('Onnx conversion complete!')
        


class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')
 

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
       # ex_inputs=tf.expand_dims(inputs,axis=1)
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        diff=inputs-mean
        tot_var=diff*diff
        variance = tf.math.reduce_mean(tot_var, axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently 
        outputs = (inputs - mean) / std
        # scale with gamma
       
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
      #  outputs=keras.layers.Reshape((1,-1))(outputs)
        # return output
        return outputs
    
if __name__ =='__main__':
    dtlnmodel=DTLN_model()
    weightsname='model.h5'
    tartgetname='DTLNM_'
 
    dtlnmodel.create_onnx_model(weightsname,tartgetname)

    

    
