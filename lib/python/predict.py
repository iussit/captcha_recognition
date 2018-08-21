import numpy as np
from PIL import Image
import preprocessing as prep
import keras
import base64
from io import BytesIO
from keras.models import load_model, model_from_json
from keras.preprocessing.image import img_to_array
# from erlport.erlang import set_message_handler


model = 0
def load_neural():
    global model
    print('load_model')
    json_file = open('lib/python/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("lib/python/w.h5")
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

def predict(img):
    image = np.array(Image.open(BytesIO(base64.b64decode(img))).convert('L'))
    image_list = prep.full_process_segmentation(image)
    image_list = [img_to_array(img) / 255 for img in image_list]
    return ''.join(map(str, model.predict_classes(np.array(image_list))))
    


