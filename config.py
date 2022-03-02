#import tensorflow as tf
import tensorflow.compat.v1 as tf




#### 
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
flags = tf.app.flags
############
tf.app.flags.DEFINE_string('f', '', 'kernel')
#####################
  
tf.app.flags.DEFINE_string('data_dir', '../data/data_maske/IBSR_MrBrains13_18_train_test_FSL/',
  """ Directory where to find the train dataset.""")

tf.app.flags.DEFINE_string('data_test', '../',
  """ Directory where to find the test dataset""")
  
tf.app.flags.DEFINE_string('save_model', 'save_model/',
  """save model path""")  
tf.app.flags.DEFINE_string('resuts_path', 'results/Output/',
  """save test MRI segmentation""")   

tf.app.flags.DEFINE_string('patch_size', "176,192,160", 
  """ Height and width of the image """)

tf.app.flags.DEFINE_string('cuda_device', '0',
  """ Select CUDA device to run the model.""")

tf.app.flags.DEFINE_integer('batch_size', 2,
  """ Number of images to be run at the same time.""")
  
tf.app.flags.DEFINE_integer('Nfilter_start', 32,
  """ First filters.""")   

tf.app.flags.DEFINE_integer('depth', 3,
  """ Model depth.""") 
  
tf.app.flags.DEFINE_integer('LAMBDA', 5, """.""")   
  
tf.app.flags.DEFINE_integer('max_steps', 25001,
  """ Number of repetitions during training.""")

tf.app.flags.DEFINE_float('learning_rate', 1e-4,
  """ Starting learning rate.""")

tf.app.flags.DEFINE_integer('steps_to_save_checkpoint', 5,
  """ Number of steps to save a checkpoint.""")


