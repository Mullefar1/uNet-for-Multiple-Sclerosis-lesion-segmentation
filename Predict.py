import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from tensorflow.keras.models import Model, load_model, save_model

# %%

# Set Parameters
im_width = 256
im_height = 256

model = load_model('PATH FOR MODEL HERE', custom_objects={'dice_coef_loss': "dice_coef_loss", 'iou': "iou", 'dice_coef': "dice_coef"})

OriMaskPath = 'PATH FOR GROUND TRUTH HERE'
save_path = 'PATH FOR SAVED PREDICTION HERE'
TEST_PATH = 'PATH FOR TEST DATA HERE'


for i in range(512):
    img = io.imread(os.path.join(TEST_PATH, "PT_" + "%d.tif" % i))
    img = cv2.resize(img, (im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred = model.predict(img)

    prediction = np.squeeze(pred) > .5
    plt.imsave(os.path.join(save_path, str(i) + '_predicted.png'), prediction)