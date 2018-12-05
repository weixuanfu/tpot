import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_array
pd.options.mode.chained_assignment = None

class SimpleAutoencoder(object):
    
    def __init__(self, x_train, y_train, x_val, y_val, encoding_dim, activation, optimizer, loss, epochs, batch_size):
        self.x_train = x_train
        self.X_width = self.x_train.shape[1]
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.input_placeholder = Input(shape = (self.X_width, ))    #input placeholder
        self.encoded = Dense(self.encoding_dim, activation = self.activation)(self.input_placeholder)   #the encoded representation of input
        self.decoded = Dense(self.X_width, activation = 'sigmoid')(self.encoded)
        
        #define autoencoder model object
        self.autoencoder = Model(self.input_placeholder, self.decoded)
        
        #define separate encoder model object
        self.encoder = Model(self.input_placeholder, self.encoded)      
        
        
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        X = check_array(X, accept_sparse='csr')
        return self
    
    
    def transform(self):
        #compile autoencoder model
        self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
        
        self.autoencoder.fit(self.x_train, self.x_train, 
                             epochs = self.epochs, 
                             batch_size = self.batch_size, 
                             shuffle = True, 
                             validation_data = (self.x_val, self.x_val))
        encoded_preds = self.encoder.predict(self.x_val)
        return encoded_preds
    
  
    