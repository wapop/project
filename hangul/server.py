
from tensorflow.keras.models import load_model 
from flask import Flask,render_template,url_for,request
import tensorflow as tf
import flask
import base64
import numpy as np
import cv2
import io

init_Base64 = 21
app = Flask(__name__)

model_kor = load_model('hand_written_korean_classification.hdf5')

# load korean label
labels_file = io.open("label.txt", 'r', encoding='utf-8').read().splitlines()
label = [str for str in labels_file]

@app.route('/')
def home():
    return render_template("mnist.html")

@app.route('/upload', methods=['POST'])
def upload():        
    global model_digit
    
    draw = request.form['url']  
    draw = draw[init_Base64:]
    draw_decoded = base64.b64decode(draw)
    image = np.asarray(bytearray(draw_decoded), dtype="uint8")

    mode = request.form.get("mode", "korean")
    
    if mode == "digit":
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)    
        image = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_AREA)    
        image = image.reshape(1,28,28,1)
        p = model_digit.predict(image)
        p = np.argmax(p)
        
    elif mode == "korean":
        f= open("data1", 'r')
        line= f.readline()
        print(line)
        f.close
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)    
        image = cv2.resize(image, dsize=(32, 32), interpolation=cv2.INTER_AREA)    
        image = (255 - image) / 255.0
        image = image.reshape(1,32,32,3)
        p = model_kor.predict(image)
        p = label[np.argmax(p)]
        image = image.reshape(32, 32, 3)
        print(image)
        cv2.imwrite('gak_result/result.jpg', image)
        print(p)
        
    return f"<div result : {p} <a href='javascript:history.back()'>뒤로</a>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8000)
