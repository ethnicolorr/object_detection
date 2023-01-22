import numpy as np
import pandas as pd
import match_template
from matplotlib import pyplot as plt
import cv2
import orb

# for i in range(1, 10):
#    match_template.findObject(cv2.imread("data/img_" + str(i) + ".jpg", 0), cv2.imread("data/template.jpg", 0))
#    plt.show()

for i in range(1, 10):
    orb.findObject(cv2.imread("data/img.jpg"), cv2.imread("data/img_" + str(i) + ".jpg"))
    plt.show()
