# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 20:49:20 2018

@author: monicai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from keras import regularizers
pd.options.mode.chained_assignment = None

class SparseAutoencoder(BaseEstimator, TransformerMixin):

    def __init__(self,encoding_dim, regularizer, activation, optimizer, loss, epochs, batch_size, random_state = 42):
        self.regularizer = regularizer
        self.encoding_dim = encoding_dim
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state



    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """

        X, y = check_X_y(X, y, accept_sparse=True, dtype=None)
        x_train, x_val, _, _ = train_test_split(
            X, y, test_size=0.25, train_size=0.75, random_state=self.random_state)

        X = check_array(X, accept_sparse='csr')

        X_width = x_train.shape[1]
        if self.regularizer == 'l1__10e-5':
            activity_regularizer = regularizers.l1(10e-5)

        self.input_placeholder = Input(shape = (X_width, ))    #input placeholder
        self.encoded = Dense(self.encoding_dim, activation = self.activation,
                            activity_regularizer = activity_regularizer)(self.input_placeholder)   #the encoded representation of input
        self.decoded = Dense(X_width, activation = 'sigmoid')(self.encoded)

        #define autoencoder model object
        self.autoencoder = Model(self.input_placeholder, self.decoded)
        #compile autoencoder model
        self.autoencoder.compile(optimizer = self.optimizer, loss = self.loss)
        self.autoencoder.fit(x_train, x_train,
                             epochs = self.epochs,
                             verbose=0,
                             batch_size = self.batch_size,
                             shuffle = True,
                             validation_data = (x_val, x_val))

        #define separate encoder model object
        self.encoder = Model(self.input_placeholder, self.encoded)
        return self


    def transform(self, X):

        X = check_array(X, accept_sparse='csr')
        encoded_preds = self.encoder.predict(X)
        return encoded_preds
