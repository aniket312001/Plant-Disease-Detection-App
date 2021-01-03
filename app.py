from flask import Flask,render_template,request,send_from_directory,url_for
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2


app = Flask(__name__)
COUNT = 0
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


IMAGE_SIZE = [224, 224, 3]
vgg16 = VGG16(input_shape=IMAGE_SIZE , weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg16.layers:
    layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(vgg16.output)

prediction = Dense(38, activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)

model.load_weights('static/leaf_model.h5')


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predictions():
    global COUNT
    img = request.files['img']  # loading img

    img.save(f'static/{COUNT}.jpg')    # saving img
    img_arr = cv2.imread(f'static/{COUNT}.jpg')    # converting into array

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    ans = model.predict(img_arr)  # it will give class
    classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy']

    res = ans[0]==max(ans[0])
    j=0;
    for i in res:
        if i:
            predict = classes[j]
        j = j+1

    COUNT += 1

    return render_template('prediction.html',data=predict)


@app.route('/load_img')
def display_image():
    global COUNT
    return send_from_directory('static', f"{COUNT-1}.jpg")


if __name__ == "__main__":
    app.run(debug=True)
