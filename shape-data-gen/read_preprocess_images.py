import cv2
import numpy as np
import os

train = []
rootdir = r'DISENTANGLEMENT METRIC data\scale20'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        frame = cv2.imread(os.path.join(subdir, file)) 
        resize_down = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resize_down, cv2.COLOR_BGR2GRAY)
        binarized = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\binzarized_scale20\_' + file + '.jpg', binarized)
        train.append(binarized / 255)
cv2.imshow('dark', binarized)
train = np.array(train)

# save to npz file
savedir = r'DISENTANGLEMENT METRIC data\binzarized_scale20.npz'
np.savez_compressed(savedir, train)
