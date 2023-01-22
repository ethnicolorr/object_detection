import cv2
from matplotlib import pyplot as plt


def findObject(img, template):
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(img, template, eval('cv2.TM_CCORR_NORMED'))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 0, 10)
    titles = ['Эталон', 'Карта вероятности', 'Обнаруженный объект']
    images = [template, result, img]
    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    return plt
