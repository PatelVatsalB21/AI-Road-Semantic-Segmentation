import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import load_model
import sys
sys.path.insert(0, '../../model/pipeline')

from pipeline import process_data


model = load_model("../model/semantic_model.h5")
index = random.randint(0, 10)

input_images, mask_images, h, w, l = process_data()

test = input_images[index]
truth = mask_images[index]

plt.figure()
plt.title("Input")
plt.imshow(test)
plt.show()

plt.figure()
plt.title("Expected")
plt.imshow(truth.reshape(truth.shape[0], truth.shape[1]))
plt.show()

prediction = model.predict(test.reshape(1, test.shape[0], test.shape[1], test.shape[2]))

print(prediction.shape)
result = np.zeros((256, 256))
for i in range(13):
    for j in range(256):
        for k in range(256):
            if prediction[0, j, k, i] > 0.5:
                result[j, k] = i

# plt.imshow(test, result)
# plt.imshow(test, truth)
def plotter(img,mask):
    fig,axes=plt.subplots(1,2)
    axes[0].imshow(img)
    plt.imshow(mask)
    plt.show()


# plotter(test,result)
plt.imshow(result, cmap="Paired")
plotter(test,mask_images[5])

