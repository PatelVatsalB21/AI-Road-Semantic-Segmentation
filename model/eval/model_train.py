import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sys

sys.path.insert(0, '../../model/pipeline')
sys.path.insert(0, '../../model/model')

from pipeline import process_data
from model import get_model

input_images, mask_images, h, y, l = process_data()

keras.backend.clear_session()
model = get_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.1, verbose=1, min_lr=1e-12),
]

history = model.fit(input_images, mask_images, batch_size=32, epochs=50, callbacks=callbacks, validation_split=0.1)

plt.figure()
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('loss')
plt.legend()

model.save_weights("/kaggle/working/weights.h5")
model.save("../../model/semantic_model.h5")
