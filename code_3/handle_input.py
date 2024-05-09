import cv2
import numpy as np
from scipy.stats import mode

def process_input_img(img_path):
    img = cv2.imread(img_path)
    orig_img = img.copy()
    cv2.imshow('orig', orig_img)
    cv2.waitKey()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area < 50:
    #         cv2.drawContours(opening, [c], -1, 0, -1)

    # # Invert and apply slight Gaussian blur
    # result = 255 - opening
    # result = cv2.GaussianBlur(result, (3,3), 0)


    # cv2.imshow('thresh', thresh)
    # cv2.waitKey()
    # cv2.imshow('opening', opening)
    # cv2.waitKey()
    # cv2.imshow('result', result)
    # cv2.waitKey()     
    # gray_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # img = cv2.medianBlur(img, 25)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # gray_img = cv2.GaussianBlur(gray_img, (11,11), 5)
    # # gray_img = cv2.medianBlur(gray_img, 5)
    # gray_img = cv2.medianBlur(gray_img, 25)
   
    # _, thresh1 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV | cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # cv2.imshow('blur', thresh1)
    # cv2.waitKey()
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    # dilation = cv2.dilate(thresh1, horizontal_kernel, iterations=1)

    # horizontal_contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return img, orig_img, cnts

def get_char_cors(img, horizontal_contours):
    cors = []
    for cnt in horizontal_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 255, 255), 3)
        if w * h > 10000 and w*h < 1000000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cors.append((x,y,w,h))
    cv2.imshow('chars', img)
    cv2.waitKey()
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
    phrase = ""
    for left, right, top, bottom in word_locs:
       text = extract_text(img[bottom:top, left:right], model=None)
       phrase += f'{text} '
    return phrase[:-1]

def extract_text(word_img, model):
    return "Hello World"

def translate_phrase(phrase, lang='de'):
    translated_text = "Hallo mein name ist Waj"
    translated_text = "Hello"
    return translated_text

def get_word_loc(img, start, end, min_height, max_height):
    left, right, top, bottom = 0, 0, 0, 0
    if start == end:
        left = start[0]
        right = end[0] + end[2]
        bottom = start[1]
        top = start[1] + start[3]
        cv2.rectangle(img, (start[0], start[1]), (start[0] + start[2], start[1]+ start[3]), (255, 0, 0), 2)
        cv2.imshow('Words', img)
        cv2.waitKey()
    else:
        left = start[0]
        right = end[0] + end[2]
        bottom = min_height
        top = max_height
        cv2.rectangle(img, (start[0], min_height), (end[0] + end[2], max_height), (255, 0, 0), 2)
        cv2.imshow('Words', img)
        cv2.waitKey()
    return left, right, top, bottom

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
            word_locs.append(get_word_loc(img, start, end, min_height, max_height))
            start = cors[i + 1]
            end = cors[i + 1]
            max_height = start[1] + start[3]
            min_height = start[1]
            i += 1
    word_locs.append(get_word_loc(img, start, end, min_height, max_height))
    return word_locs

def cover_text(img, word_locs):
    bottom, top, left, right = np.inf, 0, np.inf, 0
    for start, end, maxx, minn in word_locs:
        bottom = min(bottom, minn)
        top = max(top, maxx)
        left = min(left, start)
        right = max(right, end)


    font_size = ((top-bottom) / 100) * 1.5
    avg_color = np.mean(img[bottom:top, left:right])
    img[bottom:top, left:right] = avg_color
    cv2.imshow('Text Covered', img)
    cv2.waitKey()
    return img, ((top+bottom) // 2), left, font_size

def write_text(img, text, x,y, font_size):
    return cv2.putText(img=img, text=text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=(0,0,0), thickness=10)
    

def translate_text(img_path):
    img, orig_img, horz_cont = process_input_img(img_path)
    char_cors = get_char_cors(orig_img, horz_cont)
    word_locs = get_word_cords(orig_img, char_cors, calculate_threshold(char_cors))
    pharse = generate_text(orig_img, word_locs)
    translated_text = translate_phrase(pharse, "german")
    cover_img, bottom, left, font_size = cover_text(orig_img, word_locs)
    final_img = write_text(cover_img, translated_text, left, bottom, font_size)
    cv2.imshow('Translated Image', final_img)
    cv2.waitKey()

    
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6586.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6587.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6588.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6589.png")
translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6590.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6591.png")