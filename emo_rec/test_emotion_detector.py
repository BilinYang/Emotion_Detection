from config import emotion_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, 
                type=str, help="name of model (needs to be already trained and built) to test")
args = vars(ap.parse_args())

iap = ImageToArrayPreprocessor()
test_aug = ImageDataGenerator(rescale = 1/255.0)
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE, aug = test_aug,
                                preprocessors = [iap], classes = config.NUM_CLASSES)

model_name = args["model"].strip().lower().replace("_", "")
model_path = os.path.join("emo_rec/built_models", model_name + ".keras")
print("[INFO] loading model", model_name, "from the path" , model_path, "for testing")
model = load_model(model_path)
(loss, acc) = model.evaluate(
    test_gen.generator(), 
    steps = test_gen.numImages // config.BATCH_SIZE
)
print("[INFO] accuracy: {:.2f}%".format(acc*100))

test_gen.close()

