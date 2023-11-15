"""
   @author: Rouzbeh Yazdi

   Goal is to do a time-series forecast of the day-ahead price of 
   electricity for the next 24 hours.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras



class Baseline(keras.Model):
    def __init__(self, label=None, label_index=None):
        """
            Baseline model: 
                the prediction for the next time step is the same 
                as the value in the current time step. 
        """
        super().__init__()
    
    def call(self, inputs):
        if self.label_index is None or self.label is None:
            return inputs
        result = inputs[:,:,self.label_index]
        return result[:,:,tf.newaxis]
    
class LinearModel(keras.Model):
    def __init__(self, units=24, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(units)
    
    def call(self, inputs):
        return self.hidden(inputs)

# class DeepLinearModel(keras.Model):
#     def __init__(self, )