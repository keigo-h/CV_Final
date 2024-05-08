import cv2
import numpy as np
from scipy.stats import mode

def process_input_img(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.GaussianBlur(gray_img, (11,11), 5)
    # gray_img = cv2.medianBlur(gray_img, 5)
    gray_img = cv2.medianBlur(gray_img, 25)
    _, thresh1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV | cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)
    horizontal_contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return horizontal_contours

def get_char_cors(horizontal_contours):
    cors = []
    for cnt in horizontal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 3)
        if w * h > 1000 and w*h < 1000000:
            # cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cors.append((x,y,w,h))
    cors = sorted(cors)
    return cors

def calculate_threshold(cords):
    dists = []
    for i in range(len(cords) - 1):
        x,y,w,h = cords[i]
        x_p, y_p, w_p, h_p = cords[i+1]
        dists.append(abs(x_p - (x + w)))

    threshold = 150
    return threshold

def generate_text(img, word_locs):
    for left, right, top, bottom in word_locs:
       pass 
    pass

def extract_text(word_img, model):
    pass

def translate_phrase(word, lang):
    pass


def get_word_cords(img, cors, threshold):
    i = 0
    start = cors[0]
    end = cors[0]
    max_height = cors[0][1] + cors[0][3]
    min_height = cors[0][1]
    word_locs = []
    while i < len(cors) - 1:
        x,y,w,h = cors[i]
        x_p, y_p, w_p, h_p = cors[i+1]
        if abs(x_p - (x + w)) < threshold:
            end = cors[i+1]
            max_height = max(max_height, y_p + h_p)
            min_height = min(min_height, y_p)
            i += 1
        else:
            if start == end:
                cv2.rectangle(img, (start[0], start[1]), (start[0] + start[2], start[1]+ start[3]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (start[0], min_height), (end[0] + end[2], max_height), (255, 0, 0), 2)
            word_locs.append((start, end, max_height, min_height))
            start = cors[i + 1]
            end = cors[i + 1]
            max_height = start[1] + start[3]
            min_height = start[1]
            i += 1
    if start == end:
        cv2.rectangle(img, (start[0], start[1]), (start[0] + start[2], start[1]+ start[3]), (255, 0, 0), 2)
    else:
        cv2.rectangle(img, (start[0], min_height), (end[0] + end[2], max_height), (255, 0, 0), 2)
    word_locs.append((start, end, max_height, min_height))
    return word_locs

def cover_text(img, word_locs):
    bottom, top, left, right = 0
    for start, end, maxx, minn in word_locs:
        bottom = min(bottom, minn)
        top = max(top, maxx)
        left = min(left, start[0])
        right = max(right, end[0] + end[2])


    avg_color = np.mean(img[bottom:top, left:right])
    img[bottom:top, left:right] = avg_color
    cv2.imshow('Text Covered', img)
    cv2.waitKey()
    return img

def write_text(img, text, x,y, font_size):
    texted_image =cv2.putText(img=img, text=text, org=(x, y),fontFace=3, fontScale=3, color=(0,0,0), thickness=5)

def preProcessing(myImage):
    grayImg = cv2.cvtColor(myImage, cv2.COLOR_BGR2GRAY)
    

    im2 = myImage.copy()
    

    


    # threshold = 150
    



    cv2.imshow('After threshold', im2)
    cv2.waitKey()

    return im2

preProcessing(cv2.imread("/Users/keigoh/Desktop/CS1430_Attempt_3/finalproject-keigo-h/code_3/IMG_6588.png"))
preProcessing(cv2.imread("/Users/keigoh/Desktop/CS1430_Attempt_3/finalproject-keigo-h/code_3/IMG_6587.png"))
preProcessing(cv2.imread("/Users/keigoh/Desktop/CS1430_Attempt_3/finalproject-keigo-h/code_3/IMG_6586.png"))