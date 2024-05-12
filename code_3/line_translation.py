import cv2
from scipy.stats import mode
import numpy as np
from model import create_model
import keras.backend
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
from spellchecker import SpellChecker
import params as p
from tensorflow.python.framework.errors_impl import NotFoundError

def get_lines(img_path, klw, klh):
    img = cv2.imread(img_path)
    orig_img = img.copy()
    cv2.imshow('original', orig_img)
    cv2.waitKey()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    print(np.unique(thresh))

    orig_h = orig_img.shape[0]
    orig_w = orig_img.shape[1]

    h = int(orig_img.shape[0]*klh)
    w = int(orig_img.shape[1]*klw)

    struct_shape = (w,h)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, struct_shape)
    dilation = cv2.dilate(thresh, kernel, iterations = 1)

    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    line_locs = []

    threshold = int((orig_h * orig_w) * 0.035)

    print(threshold)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if (w*h) > threshold:
            line_locs.append(cv2.boundingRect(cnt))
    return line_locs, orig_img

def get_line_img(img, img_cors):
    imgs = []
    for x,y,w,h in img_cors:
        copy_img = img.copy()
        imgs.append((copy_img[y:y+h, x:x+w], (x,y,w,h)))
    return imgs

def get_word_locs(line_img, kww, kwh):
    word_locs = []
    orig_img = line_img.copy()

    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    orig_h = orig_img.shape[0]
    orig_w = orig_img.shape[0]


    kh = int(orig_img.shape[0] * kwh)
    kw = int(orig_img.shape[1] * kww)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw,kh))
    dilation = cv2.dilate(thresh, kernel, iterations = 1)

    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    threshold = int((orig_h * orig_w) * 0.2)
    print(threshold)
    for cnt in contours:
        x,y,w,h =cv2.boundingRect(cnt)
        if w*h > threshold:
            word_locs.append((x,y,w,h))
    return line_img, orig_img, word_locs

def get_word_img(line_img, word_cors):
    word_imgs = []
    word_cors = sorted(word_cors)
    for x,y,w,h in word_cors:
        word_imgs.append(line_img[y:y+h, x:x+w])
    return word_imgs 

def cover_text(orig_img, line_img, line_img_locs):
    x,y,w,h = line_img_locs
    avg_color = mode(line_img.flatten(), keepdims=False).mode
    orig_img[y:y+h, x:x+w] = avg_color
    return orig_img

def generate_text(word_imgs, model):
    phrase = ""
    for im in word_imgs:
       text = extract_text(im, model)
       phrase += f'{text} '
    return phrase[:-1]

def get_model(trained_model_path="trained_models/model.h5"):
    model,_,_ = create_model()
    try:
        model.load_weights(trained_model_path)
    except NotFoundError as e:
        model.load_weights("trained_models/model.h5")
    return model

def process_img(word_img):
    img = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)

    h,w = img.shape
    aspect_width = 32
    aspect_height = int(w * (aspect_width / h))
    img = cv2.resize(img, (aspect_height, aspect_width))

    h, w = img.shape
    img = img.astype('float32')

    if h < 32:
        add_zeros = np.full((32-h, w), 255)
        img = np.concatenate((img, add_zeros))
        h, w = img.shape

    if w < 128:
        add_zeros = np.full((h, 128-w), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        h, w = img.shape
        
    if w > 128 or h > 32:
        shape = p.cv2_img_size
        img = cv2.resize(img, shape)

    img = np.expand_dims(img, -1)
    img = img / 255 - 0.5
    img = np.reshape(img, (1,32,128,1))

    return img

def extract_text(word_img, model):
    word_img = process_img(word_img)

    prediction = model.predict(word_img)

    decoded = keras.backend.ctc_decode(prediction, 
                        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                        greedy=True)[0][0]
    out = keras.backend.get_value(decoded)
    res = ""
    for l in out[0]:
        if l != -1: 
            res += p.char_lst[l]
    print(res)
    spell = SpellChecker()
    cor_res = spell.correction(res)
    print(cor_res)
    if cor_res is None:
        cor_res = res
    return cor_res

def translate_phrase(phrase, lang='de'):
    translator = Translator()   
    translated_phrase = translator.translate(phrase, src='en', dest=lang)
    print(translated_phrase)
    return translated_phrase.text


def write_text(cover_img, translated_text, left, bottom, font_size= 100):  
    font_path = "fonts/Roboto-Regular.ttf"
    font = ImageFont.truetype(font=font_path, size=100)
    img_pil = Image.fromarray(cover_img)  
    draw = ImageDraw.Draw(img_pil)  
    draw.text((left, bottom), translated_text, font=font, spacing=10, fill=(0, 0, 0, 0), align='center') 
    image = np.array(img_pil) 
    return image

def translate_just_word(img, model):
    extract_text(img, model)

def run_translation(file_path=None, file_type=1, model_file_path="trained_models/model.h5"):
    model = get_model(model_file_path)
    if file_type == 0:
        klw, klh = p.line_l_w, p.line_l_h
        kww, kwh = p.line_w_w, p.line_w_h
    elif file_type == 1:
        klw, klh = p.sing_l_w, p.sing_l_h
        kww, kwh = p.sing_w_w, p.sing_w_h
    else:
        klw, klh = 1,1
        kww, kwh = 1,1
    
    line_locs, orig_img = get_lines(file_path, klw, klh)
    line_imgs = get_line_img(orig_img, line_locs)
    for img, cors in line_imgs:
        line_img, original_img, word_locs = get_word_locs(img, kww, kwh)
        word_imgs = get_word_img(original_img, word_locs)
        phrase = generate_text(word_imgs, model)
        translated_phrase = translate_phrase(phrase)
        orig_img = cover_text(orig_img, line_img, cors)
        orig_img = write_text(orig_img, translated_phrase, cors[0], cors[1])
        cv2.imshow('translated', orig_img)
        cv2.waitKey() 

# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/the_door_is_open.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/multi_line.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/bill_multi.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/pineapple3.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/is.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/data_like_my_2.png"
# file_path = "/Users/keigoh/Desktop/CS1430_Attempt_3/CV_Final/code_3/test_imgs/data_like_name_2.png"
# run_translation()