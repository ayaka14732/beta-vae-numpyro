# Python 3 program to draw rectangle shape on solid image
import cv2
import numpy as np
import random

scaling = 20
for i in range(1000):

    shape_label = random.randint(0, 4)
    img = np.zeros((400, 400, 3), dtype='uint8')
    if shape_label == 0:  # rectangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g

        c = int(300*np.random.uniform())
        d = int(300*np.random.uniform())
        cv2.rectangle(img, (c, int(np.random.uniform() * d)), (c + scaling, d + scaling), (r, g, b), -1)

    elif shape_label == 1:  # square
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(300*np.random.uniform())
        b = int(300*np.random.uniform())
        cv2.rectangle(img, (a, b), (a + scaling, b + scaling), (r, g, blue), -1)

    elif shape_label == 2:  # ellipse
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = int(50+300*np.random.uniform()), int(50+300*np.random.uniform())
        axes = scaling, int(scaling*2*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)

    elif shape_label == 3:  # circle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (50+int(300*np.random.uniform()), 50 + int(300*np.random.uniform())), scaling, (r, g, b), -1)

    elif shape_label == 4:  # triangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g

        c = int(300*np.random.uniform())
        d = int(300*np.random.uniform())
        pts = [(c, d), (c + int(scaling/np.sqrt(2)), d + int(scaling/np.sqrt(2))), (c + int(scaling/np.sqrt(2)), d - int(scaling/np.sqrt(2)))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(r'DISENTANGLEMENT METRIC data\scale20\scale20_' +
                str(i) + '.jpg', img)
