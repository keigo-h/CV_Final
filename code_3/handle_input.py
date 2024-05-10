import cv2
import numpy as np
from scipy.stats import mode
from model import create_model
import keras.backend as K
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

chars = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

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
        # if w * h > 10000 and w*h < 100000:
        if w * h > 500:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
            cors.append((x,y,w,h))
    # cv2.imshow('chars', img)
    # cv2.waitKey()
    cors = sorted(cors)
    return cors

def calculate_threshold(cords):
    dists = []
    for i in range(len(cords) - 1):
        x,y,w,h = cords[i]
        x_p, y_p, w_p, h_p = cords[i+1]
        dists.append(abs(x_p - (x + w)))
    threshold = 200
    return threshold

def generate_text(img, word_locs):
    phrase = ""
    for left, right, top, bottom in word_locs:
       text = extract_text(img[bottom-10:top+10, left-10:right+10])
       phrase += f'{text} '
    return phrase[:-1]

def extract_text(word_img, trained_model_path="val_loss_cur_best.hdf5"):
    model,_,_ = create_model()
    model.load_weights(trained_model_path)
    word_img = process_img(word_img)

    prediction = model.predict(word_img)
    
    # use CTC decoder
    decoded = K.ctc_decode(prediction, 
                        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                        greedy=True)[0][0]
    out = K.get_value(decoded)
    res = ""
    for l in out[0]:
        if l != -1: 
            res += chars[l]
    # print(res)

    return res

def process_img(word_img):
    img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    
    w,h = img.shape
    aspect_width = 32
    aspect_height = int(h * (aspect_width / w))
    img = cv2.resize(img, (aspect_height, aspect_width))

    w, h = img.shape
    img = img.astype('float32')

    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)

    img = np.expand_dims(img, -1)

    print(img.shape)
    # img = cv2.transpose(img)
    img = img / 255 - 0.5
    # cv2.imshow('after process', img)
    # cv2.waitKey()
    img = np.reshape(img, (1,32,128,1))

    return img

def translate_phrase(phrase, lang='de'):
    translator = Translator()   
    translated_phrase = translator.translate(phrase, src='en', dest=lang)
    return translated_phrase.text

def get_word_loc(img, start, end, min_height, max_height):
    left, right, top, bottom = 0,0,0,0
    if start == end:
        left = start[0]
        right = end[0] + end[2]
        bottom = start[1]
        top = start[1] + start[3]
        # cv2.rectangle(img, (start[0], start[1]), (start[0] + start[2], start[1]+ start[3]), (255, 0, 0), 2)
        # cv2.imshow('Words', img)
        # cv2.waitKey()
    else:
        left = start[0]
        right = end[0] + end[2]
        bottom = min_height
        top = max_height
        # cv2.rectangle(img, (start[0], min_height), (end[0] + end[2], max_height), (255, 0, 0), 2)
        # cv2.imshow('Words', img)
        # cv2.waitKey()
    return left, right, top, bottom

def get_word_cords(img, cors, threshold):
    i = 0
    start = cors[0]
    end = cors[0]
    max_height = cors[0][1] + cors[0][3]
    min_height = cors[0][1]
    word_locs = []
    print(len(cors))
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

def print_utf8(cover_img, translated_text, left, bottom, font_size):  
    # font = ImageFont.load_default()
    font_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/Roboto-Bold.ttf"
    font = ImageFont.truetype(font=font_path, size=100)
    img_pil = Image.fromarray(cover_img)  
    draw = ImageDraw.Draw(img_pil)  
    draw.text((left, bottom), translated_text, font=font,
           fill=(0, 0, 0, 0)) 
    image = np.array(img_pil) 
    return image
    

def translate_text(img_path):
    img, orig_img, horz_cont = process_input_img(img_path)
    char_cors = get_char_cors(orig_img, horz_cont)
    word_locs = get_word_cords(orig_img, char_cors, calculate_threshold(char_cors))
    pharse = generate_text(orig_img, word_locs)
    translated_text = translate_phrase(pharse, "german")
    cover_img, bottom, left, font_size = cover_text(orig_img, word_locs)
    final_img = print_utf8(cover_img, translated_text, left, bottom, font_size)
    # final_img = write_text(cover_img, translated_text, left, bottom, font_size)
    cv2.imshow('Translated Image', final_img)
    cv2.waitKey()

def quick_translate(path):
   img = cv2.imread(path)
   extract_text(img) 
    
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6586.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6587.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6588.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6589.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6590.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/IMG_6591.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/a02/a02-000/a02-000-01-03.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/d03/d03-112/d03-112-01-03.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/hi2.png")
# translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/hi.png")
   
# dataset my
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/a01/a01-096u/a01-096u-07-01.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/a01/a01-102/a01-102-01-04.png")

# data look alike my
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/data_like_my_2.png")

# data name
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/c06/c06-031/c06-031-01-02.png")

# data like name
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/data_like_name_2.png")

# data sentence word
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/test_sent_1.png")
   
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/words/a04/a04-092/a04-092-07-00.png")
   
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/bill.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/hi.png")

# The
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/the.png")
   
# door
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/door.png")

# is
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/is.png")
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/ls.png")

# open
# quick_translate("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/open.png")

# the door is open
translate_text("/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/the_door_is_open.png")
   


# print(translate_phrase("hi"))