

# Python3 program to draw rectangle shape on solid image
import numpy as np
import random
import cv2

# invaiant shape
for i in range(1000, 3000):

    shape_label = random.randint(0, 4)
    if shape_label == 0:  # rectangle
        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (int(400*np.random.uniform()), int(400*np.random.uniform())), (int(400*np.random.uniform()), int(400*np.random.uniform())), (r, g, b), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_1.jpg', img)

        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (int(400*np.random.uniform()), int(400*np.random.uniform())), (int(400*np.random.uniform()), int(400*np.random.uniform())), (r, g, b), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_2.jpg', img)

    elif shape_label == 1:  # square
        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(400*np.random.uniform())
        b = int(400*np.random.uniform())
        c = int(400*np.random.uniform())
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_1.jpg', img)

        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(400*np.random.uniform())
        b = int(400*np.random.uniform())
        c = int(400*np.random.uniform())
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_2.jpg', img)

    elif shape_label == 2:  # ellipse
        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = int(50+300*np.random.uniform()), int(50+300*np.random.uniform())
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' +  str(i) + '_1.jpg', img)

        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = int(50+300*np.random.uniform()), int(50+300*np.random.uniform())
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_2.jpg', img)

    elif shape_label == 3:  # circle
        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (50+int(300*np.random.uniform()), 50+int(300*np.random.uniform())), int(50*np.random.uniform()), (r, g, b), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_1.jpg', img)

        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (50+int(300*np.random.uniform()), 50+int(300*np.random.uniform())), int(50*np.random.uniform()), (r, g, b), -1)
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_2.jpg', img)

    elif shape_label == 4:  # triangle
        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(350*np.random.uniform()), int(350*np.random.uniform())), (int(350*np.random.uniform()), int(350*np.random.uniform())), (int(350*np.random.uniform()), int(350*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_1.jpg', img)

        img = np.zeros((400, 400, 3), dtype='uint8')
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(350*np.random.uniform()), int(350*np.random.uniform())), (int(350*np.random.uniform()), int(350*np.random.uniform())), (int(350*np.random.uniform()), int(350*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))
        cv2.imwrite(r'DISENTANGLEMENT METRIC data\shape\shape_' + str(i) + '_2.jpg', img)


# invariant x position
for i in range(1000, 3000):
    x_position = int(30 + 340*np.random.uniform())
    shape_label = random.randint(0, 4)
    img = np.zeros((400, 400, 3), dtype='uint8')
    if shape_label == 0:  # rectangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (x_position-int(100*np.random.uniform()/2), int(400*np.random.uniform())), (x_position+int(100*np.random.uniform()/2), int(400*np.random.uniform())), (r, g, b), -1)

    elif shape_label == 1:  # square
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(x_position-100*np.random.uniform()/2)
        b = int(400*np.random.uniform())
        c = int(x_position+100*np.random.uniform()/2)
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)

    elif shape_label == 2:  # ellipse
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = x_position, int(50+300*np.random.uniform())
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)

    elif shape_label == 3:  # circle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (x_position, 50+int(300*np.random.uniform())), int(30*np.random.uniform()), (r, g, b), -1)

    elif shape_label == 4:  # triangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(50*np.random.uniform()/2+x_position), int(300*np.random.uniform())), (int(x_position-50 * np.random.uniform()/2), int(300*np.random.uniform())), (int(300*np.random.uniform()), int(300*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(
        r'DISENTANGLEMENT METRIC data\position_x\position_x_'+str(i)+'_1.jpg', img)

    shape_label = random.randint(0, 4)
    img = np.zeros((400, 400, 3), dtype='uint8')
    if shape_label == 0:  # rectangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (x_position-int(100*np.random.uniform()/2), int(400*np.random.uniform())), (x_position+int(100*np.random.uniform()/2), int(400*np.random.uniform())), (r, g, b), -1)

    elif shape_label == 1:  # square
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(x_position-100*np.random.uniform()/2)
        b = int(400*np.random.uniform())
        c = int(x_position+100*np.random.uniform()/2)
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)

    elif shape_label == 2:  # ellipse
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = x_position, int(50+300*np.random.uniform())
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)

    elif shape_label == 3:  # circle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (x_position, 50+int(300*np.random.uniform())), int(30*np.random.uniform()), (r, g, b), -1)

    elif shape_label == 4:  # triangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(50*np.random.uniform()/2+x_position), int(300*np.random.uniform())), (int(x_position-50 * np.random.uniform()/2), int(300*np.random.uniform())), (int(300*np.random.uniform()), int(300*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(r'DISENTANGLEMENT METRIC data\position_x\position_x_'+str(i)+'_2.jpg', img)


# invariant y position
for i in range(1000, 3000):
    y_position = int(30 + 340*np.random.uniform())
    shape_label = random.randint(0, 4)
    img = np.zeros((400, 400, 3), dtype='uint8')
    if shape_label == 0:  # rectangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (int(400*np.random.uniform()), y_position-int(100*np.random.uniform()/2),), (int(400*np.random.uniform()), y_position+int(100*np.random.uniform()/2)), (r, g, b), -1)

    elif shape_label == 1:  # square
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(400*np.random.uniform())
        b = int(y_position+100*np.random.uniform()/2)
        c = int(400*np.random.uniform())
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)

    elif shape_label == 2:  # ellipse
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = int(50+300*np.random.uniform()), y_position
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)

    elif shape_label == 3:  # circle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (50+int(300*np.random.uniform()), y_position), int(30*np.random.uniform()), (r, g, b), -1)

    elif shape_label == 4:  # triangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(300*np.random.uniform()), int(50*np.random.uniform()/2+y_position)), (int(300*np.random.uniform()), int(y_position-50*np.random.uniform()/2)), (int(300*np.random.uniform()), int(300*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(
        r'DISENTANGLEMENT METRIC data\position_y\position_y_'+str(i)+'_1.jpg', img)

    shape_label = random.randint(0, 4)
    img = np.zeros((400, 400, 3), dtype='uint8')
    if shape_label == 0:  # rectangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.rectangle(img, (int(400*np.random.uniform()), y_position-int(100*np.random.uniform()/2),), (int(400*np.random.uniform()), y_position+int(100*np.random.uniform()/2)), (r, g, b), -1)

    elif shape_label == 1:  # square
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        blue = 255 - r - g
        a = int(400*np.random.uniform())
        b = int(y_position+100*np.random.uniform()/2)
        c = int(400*np.random.uniform())
        d = b + c - a
        cv2.rectangle(img, (a, b), (c, d), (r, g, blue), -1)

    elif shape_label == 2:  # ellipse
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        colour = (r, g, b)
        center = int(50+300*np.random.uniform()), y_position
        axes = int(10+50*np.random.uniform()), int(15+50*np.random.uniform())
        angle = int(360*np.random.uniform())
        cv2.ellipse(img, center, axes, angle, 0, 360, colour, -1)

    elif shape_label == 3:  # circle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        cv2.circle(img, (50+int(300*np.random.uniform()), y_position), int(30*np.random.uniform()), (r, g, b), -1)

    elif shape_label == 4:  # triangle
        r = int(255*np.random.uniform())
        g = int((255-r)*np.random.uniform())
        b = 255 - r - g
        pts = [(int(300*np.random.uniform()), int(50*np.random.uniform()/2+y_position)), (int(300*np.random.uniform()), int(y_position-50*np.random.uniform()/2)), (int(300*np.random.uniform()), int(300*np.random.uniform()))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(r'DISENTANGLEMENT METRIC data\position_y\position_y_'+str(i)+'_2.jpg', img)


# invariant scale
for i in range(1000, 3000):
    scaling = int(10 + 50*np.random.uniform())
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

    cv2.imwrite(r'DISENTANGLEMENT METRIC data\scale\scale_' + str(i) + '_1.jpg', img)

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
        pts = [(c, d), (c + int(scaling/np.sqrt(2)), d + int(scaling/np.sqrt(2))), (int(scaling/np.sqrt(2)), d - int(scaling/np.sqrt(2)))]
        cv2.fillPoly(img, np.array([pts]), (r, g, b))

    cv2.imwrite(r'DISENTANGLEMENT METRIC data\scale\scale_' + str(i) + '_2.jpg', img)
