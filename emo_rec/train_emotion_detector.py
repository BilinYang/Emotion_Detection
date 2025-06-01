import matplotlib
matplotlib.use("Agg")
from config import emotion_config as config
from model_loader import model_loader as l
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint, TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.neural_net_models.models import EmotionVGGNet
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model, Model
import keras.backend as K
import argparse
import os
import sys

# construct an argument parse and parse the arguments
ap = argparse.ArgumentParser() 
ap.add_argument("-m", "--model", type=str, help="name of model to train")
ap.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())


# initialize image preprocessor and the training and validation image generators for data augmentation
iap = ImageToArrayPreprocessor()
train_aug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale = 1/255.0)
val_aug = ImageDataGenerator(rescale = 1/255.0)
 

# initialize the training and validation dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug = train_aug, 
                                 preprocessors= [iap], classes = config.NUM_CLASSES)
val_gen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug = val_aug, 
                               preprocessors= [iap], classes = config.NUM_CLASSES)


loader = l.ModelLoader()
model, model_name, start_epoch = None, None, None

if args["start_epoch"] is None: 
    start_epoch = 0 
else: 
    start_epoch = args["start_epoch"]


# if there is no specific model checkpoint supplied, use emotionvggnet as the default 
if args["model"] is None: 
    model_name = "emotionvggnet"
    model = EmotionVGGNet.build(48, 48, 1, config.NUM_CLASSES)

# otherwise, use the model passed in the by the user
else: 
    model_name = (args["model"].lower()).strip()
    model_name = model_name.replace("_", "")
        
try: 
    print("[INFO] trying to load the neural net {}".format(model_name))

    model = loader.build_model(model_name, **l.get_model_config())
    print("[INFO] successfully built the model {}".format(model_name))

except Exception as e: 
    print(f"[INFO] error building model: {str(e)}")
    sys.exit(1)


opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# construct callbacks
fig_path = os.path.join(config.TRAINING_LOG_PATH, model_name +  "_training.png")
json_path = os.path.join(config.TRAINING_LOG_PATH, model_name + "_training.json")
checkpoint_path = "emo_rec/checkpoints"

callbacks = [
    EpochCheckpoint(checkpoint_path, every=5, startAt = start_epoch), 
    TrainingMonitor(fig_path, jsonPath = json_path, startAt = start_epoch)
]

model.fit(
    train_gen.generator(), 
    steps_per_epoch = train_gen.numImages // config.BATCH_SIZE, 
    validation_data = val_gen.generator(), 
    validation_steps = val_gen.numImages // config.BATCH_SIZE, 
    epochs = l.get_epoch_for_model(model_name), 
    callbacks = callbacks, 
    verbose=1
)

train_gen.close()
val_gen.close()

model.save(f"emo_rec/built_models/{model_name}.keras")
print(f'[INFO] model saved to "emo_rec/built_models"')

