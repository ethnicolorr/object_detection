import cv2
from matplotlib import pyplot as plt


def findObject(template, img):
    orb = cv2.ORB_create()
    kp_template, des_template = orb.detectAndCompute(template, None)
    kp_img, des_img = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des_img)
    matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(template, kp_template, img, kp_img, matches[:25], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    x_min = x_max = y_min = y_max = 0
    for m in matches[:25]:
        # template_idx = m.queryIdx  # индексы kp_template
        img_idx = m.trainIdx  # индексы kp_img
        (x1, y1) = kp_img[img_idx].pt
        # (x2, y2) = kp_template[template_idx].pt
        if x_min == 0 and y_min == 0:
            x_min = x1
            y_min = y1
        x_max = max(x_max, x1)
        x_min = min(x_min, x1)
        y_max = max(y_max, y1)
        y_min = min(y_min, y1)
    img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 10)
    titles = ['Сопоставление', 'Обнаруженный объект']
    images = [result, img]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    return plt
