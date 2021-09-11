import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import cv2

model = load_model("../model/semantic_model(92.4).h5")
cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()
    frame = cv2.resize(frame, (256, 256))
    pred = model.predict(frame.reshape(1, frame.shape[0], frame.shape[1], frame.shape[2]))

    # print(pred.shape)
    newImg = np.zeros((256, 256))
    # print(pred)
    for i in range(13):
        for j in range(256):
            for k in range(256):
                if pred[0, j, k, i] > 0.5:
                    newImg[j, k] = i

    cv2.imshow('Segmented Result', newImg)
    plt.imshow(newImg, cmap="Paired")

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

