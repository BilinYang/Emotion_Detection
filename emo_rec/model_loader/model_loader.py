import importlib
import inspect
from typing import Dict, Any, Type
from keras.models import Model
import argparse
from config import emotion_config as config
import sys


class ModelLoader:
    
    VALID_MODELS  = {'alexnet', 'deepergooglenet', 'emotionvggnet', 'lenet', 'minigooglenet', 'minivggnet', 'shallownet'}
    CLASS_NAMES = {
          'alexnet': 'AlexNet',
          'deepergooglenet': 'DeeperGoogLeNet', 
          'emotionvggnet': 'EmotionVGGNet', 
          'lenet': 'LeNet', 
          'minigooglenet': 'MiniGoogLeNet', 
          'minivggnet': 'MiniVGGNet', 
          'shallownet': 'ShallowNet'
     }
    
    epochs_dict = {
          'alexnet': 10,
          'deepergooglenet': 9, 
          'emotionvggnet': 16, 
          'lenet': 5, 
          'minigooglenet': 10, 
          'minivggnet': 17, 
          'resnet': 10, 
          'shallownet': 13
     }

    def __init__(self):
          self.models_path = "pyimagesearch.neural_net_models.models"
          self._model_cache: Dict[str, Type] = {}

    def load_model_class(self, name): 
        model_name = (name.lower()).strip()
        model_name = model_name.replace("_", "")

        if model_name in self._model_cache: 
             print("[INFO] successfully loaded model {}".format(model_name))
             return self._model_cache[model_name]
        
        try: 
            module = importlib.import_module(f"{self.models_path}.{model_name}")
            if model_name in self.CLASS_NAMES.keys(): 
                 class_name = self.CLASS_NAMES[model_name]
                 model_class = getattr(module, class_name)
                 self._model_cache[model_name] = model_class
                 return model_class
            raise AttributeError(f"No valid model class found in {model_name}.py")

        
        except ImportError as e: 
            raise ValueError(
               f"Model '{model_name}' is not available.\n"
               f"Available models: {self.VALID_MODELS}") from e
        
    
    def build_model(self, name, **build_kwargs): 
         model_class = self.load_model_class(name)
         print("CLASS", model_class)
         
         sig = inspect.signature(model_class.build)
         filtered_kwargs = {}
         for param in sig.parameters.values():
            if param.name in build_kwargs:
                filtered_kwargs[param.name] = build_kwargs[param.name]
         return model_class.build(**filtered_kwargs)
    

         
def get_model_config(): 
     return {
          'width': 48,
          'height': 48,
          'depth': 1,
          'classes': config.NUM_CLASSES
     }


def get_epoch_for_model(model_name): 
    return ModelLoader.epochs_dict[model_name]
