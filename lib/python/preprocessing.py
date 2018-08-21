import cv2
import numpy as np
from skimage import feature
from skimage import measure
from scipy.spatial import distance
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


corners = 0
y_top_left = 20
y_down_right = 90
mean_distance_character = 9
lenght_character_and_epsilon = mean_distance_character + 2
quantiti_character = 6

def image_processing(img):
    kernel = np.ones((2, 1), np.uint8)
    opening = cv2.erode(img, kernel, iterations=1)
    kernel = np.ones((1, 2), np.uint8)
    opening = cv2.bilateralFilter(cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel), d=9, sigmaColor=5, sigmaSpace=5)
    kernel = np.ones((2, 1), np.uint8)
    opening = cv2.adaptiveThreshold(cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel), 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize=5, C=7)
    opening = cv2.bilateralFilter(opening, d=9, sigmaColor=5, sigmaSpace=5)
    invert = cv2.bitwise_not(opening)
    return invert


def segmentation(invert, sigma):
    global y_top_left
    global y_down_right

    edges = feature.canny(invert, sigma=sigma, low_threshold=5, high_threshold=10)
    label_image = measure.label(edges)

    rectangles_segmentation = []
    for region in measure.regionprops(label_image):
        _, x_top_left, _, x_down_right = region.bbox
        rectangles_segmentation.append([x_top_left, y_top_left, x_down_right, y_down_right])

    return rectangles_segmentation


def post_filtration_segmentation(rectangles_segmentation):
    lines = []
    for rect in rectangles_segmentation:
        for rect_temp in rectangles_segmentation:
            if ((rect[0] - rect_temp[2] <= mean_distance_character) and (
                    rect[0] - rect_temp[2] >= -mean_distance_character)) and (rect[1] != rect_temp[3]):
                lines.append([rect[0], rect_temp[2]])

    lines = sorted(lines, key=lambda line: line[0])

    region_lines = []
    flag = -1
    for index, line in enumerate(lines):
        region = []
        if index <= flag:
            continue
        for index_temp, line_temp in enumerate(lines):
            if (line_temp[0] - line[0] <= lenght_character_and_epsilon):
                region.append(line_temp)
                continue
            flag = index_temp
            break

        region_lines.append(region)

    true_lines = []
    q_number = []
    for region in region_lines:
        flag = False
        for q in q_number:
            if max([reg[0] for reg in region]) == q:
                flag = True
        if flag:
            continue
        true_lines.append([max([reg[0] for reg in region]), max([reg[1] for reg in region])])
        q_number.append(max([reg[0] for reg in region]))

    true_lines_return = []
    for index, line in enumerate(true_lines):
        if (index < len(true_lines) - 1) and (true_lines[index + 1][0] - line[0] >= lenght_character_and_epsilon) and (
                index != 0):
            continue
        true_lines_return.append([line[0], line[1]])

    return true_lines_return


def get_min_max():
    global corners
    corners_sort = sorted([[c[0][0], c[0][1]] for c in corners], key=lambda doth: doth[0])
    min_x = min([d[0] for d in corners_sort])
    max_x = max([d[0] for d in corners_sort])
    max_y = max(d[1] for d in corners_sort)
    min_y = min(d[1] for d in corners_sort)
    return min_x, max_x, max_y, min_y


def full_segmentation(img, sigma=1.75):
    invert = image_processing(img)

    global corners
    corners = cv2.goodFeaturesToTrack(invert, maxCorners=100, qualityLevel=0.2, minDistance=10, blockSize=5,
                                      useHarrisDetector=True, k=0.05)
    corners = np.int0(corners)

    rectangles_segmentation = segmentation(invert, sigma)

    lines = post_filtration_segmentation(rectangles_segmentation)

    min_X, max_X, _, _ = get_min_max()

    lines.append([min_X, min_X])
    lines.append([max_X, max_X])

    lines = sorted(lines, key=lambda line: line[0])

    return lines, invert


def post_post_filtration(lines):
    delete_lines = []
    for index, line in enumerate(lines):
        if (index != len(lines) - 1) and (lines[index + 1][0] - line[0] <= 8):
            delete_lines.append(line[0])

        if (line[0] - lines[index - 1][0] <= lenght_character_and_epsilon) and (index != 0):
            delete_lines.append(line[0])

    for index, l in enumerate(lines):
        if l[0] in delete_lines:
            lines.remove(l)

    return lines


def segmentation_on_mean():
    global corners
    doths_corner = sorted([[c[0][0], c[0][1]] for c in corners], key=lambda doth: doth[0])
    all_X = [d[0] for d in doths_corner]

    len_line = (np.max(all_X) - np.min(all_X)) // quantiti_character
    lines = [np.min(all_X) + r * len_line for r in range(1, quantiti_character)]
    lines.append(np.max(all_X))
    lines.append(np.min(all_X))
    lines.sort()

    true_lines = []
    for index, line in enumerate(lines):
        for doth in doths_corner:
            if (index == 0) or (index == len(lines) - 1):
                true_lines.append([line, line])
                break
            if (doth[0] < line) and (doth[0] - mean_distance_character >= lines[index - 1]):
                if doth[0] + mean_distance_character > lines[index + 1]:
                    continue
                line = doth[0]
                true_lines.append([line, line])
                break

    return true_lines


def stupid_segmentation_on_mean():
    global corners
    doths_corner = sorted([[c[0][0], c[0][1]] for c in corners], key=lambda doth: doth[0])
    all_X = [d[0] for d in doths_corner]

    len_line = (np.max(all_X) - np.min(all_X)) // quantiti_character
    lines = [np.min(all_X) + r * len_line for r in range(1, quantiti_character)]
    lines.append(np.max(all_X))
    lines.append(np.min(all_X))
    lines.sort()

    reshape_lines = []
    for l in lines:
        reshape_lines.append([l, l])

    return reshape_lines


def bisect(lines):
    distance = []
    for index, l in enumerate(lines):
        if index != 0:
            distance.append(l[0] - lines[index - 1][0])

    max_distance = max(distance)
    index_distance = distance.index(max_distance)
    av_distance = max_distance // 2

    lines.append([lines[index_distance][0] + av_distance, lines[index_distance][0] + av_distance])
    return lines


def threesect(lines):
    distance = []
    for index, l in enumerate(lines):
        if index != 0:
            distance.append(l[0] - lines[index - 1][0])

    max_distance = max(distance)
    index_distance = distance.index(max_distance)

    #45 - this in length three character + epsilon
    if max_distance >= 45:
        av_distance = max_distance // 2
        lines.append([lines[index_distance][0] + av_distance, lines[index_distance][0] + av_distance])
        lines.append([lines[index_distance][0] + 2 * av_distance, lines[index_distance][0] + 2 * av_distance])

    else:
        lines = bisect(lines)
        lines = bisect(lines)

    return lines


def crop_image(image, rect):
    points_sum = {}
    points_diff = {}
    for x, y in rect:
        points_sum[x + y] = x, y
        points_diff[y - x] = x, y
    points_sum = sorted(points_sum.items())
    points_diff = sorted(points_diff.items())

    tl = points_sum[0][1]
    tr = points_diff[0][1]
    bl = points_diff[-1][1]
    br = points_sum[-1][1]

    w = int(max(distance.euclidean(tl, tr), distance.euclidean(bl, br)))
    h = int(max(distance.euclidean(tl, bl), distance.euclidean(tr, br)))

    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matix, (w, h))


def get_image_list(invert, lines):
    min_x, max_x, max_y, min_y = get_min_max()
    image_list = []

    lines = sorted(lines, key=lambda line: line[0])

    for index, line in enumerate(lines):
        if index != len(lines) - 1:
            image = crop_image(invert, [[line[0], max_y], [line[1], min_y],
                                        [lines[index + 1][0], max_y], [lines[index + 1][1], min_y]])

            image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
            image = cv2.resize(image, (28, 28))
            image_list.append(image)

    return image_list


def full_process_segmentation(img):
    lines, invert = full_segmentation(img)
    lines = post_post_filtration(lines)

    if len(lines) < 7:
        for sigma in [2, 2.25, 2.5]:
            lines, invert = full_segmentation(img, sigma=sigma)
            lines = post_post_filtration(lines)
            if len(lines) == 7:
                break

    if len(lines) > 7:
        for sigma in [1.5, 1.25, 1]:
            lines, invert = full_segmentation(img, sigma=sigma)
            lines = post_post_filtration(lines)
            if len(lines) == 7:
                break

    if len(lines) == 6:
        lines = bisect(lines)

    if len(lines) != 7:
        lines = segmentation_on_mean()

    if len(lines) == 6:
        lines = bisect(lines)

    if len(lines) == 5:
        lines = threesect(lines)

    if len(lines) != 7:
        lines = stupid_segmentation_on_mean()

    image_list = get_image_list(invert, lines)

    return image_list


def model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

