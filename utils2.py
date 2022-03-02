import tensorflow as tf
import numpy as np

def get_dsc(labels,predictions):#,scope=None):
  shape = tf.shape(labels)
  axes_1 = tf.shape(shape)[0]-1
  axes = tf.range(tf.shape(shape)[0]-1)
  pred = tf.argmax(predictions, axis=axes_1)
  pred = tf.one_hot(indices=pred, 
    depth=shape[-1],
    on_value=1,
    off_value=0,
    axis=-1)
  pred = tf.cast(pred, tf.float32)
  label = tf.cast(labels, tf.float32)
  numer = 2*tf.reduce_sum(label*pred, axis=axes)
  denom = tf.reduce_sum(pred,axis=axes)+tf.reduce_sum(label,axis=axes)
  equal = tf.cast(tf.equal(numer,denom), tf.float32)
  dsc = (numer+equal) / (denom+equal)
  return dsc

