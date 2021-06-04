import mysql.connector
from flask import Flask, render_template, request, session,url_for,redirect,current_app
from datetime import timedelta
import re
import os
import secrets
import numpy as np
import keras.models
import sys
import base64
import pandas as pd
import numpy as np 
import itertools
import keras
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 
import datetime
import time
sys.path.append(os.path.abspath("./model_saved"))

app = Flask(__name__)

app.secret_key = 'your secret key'
app.permanent_session_lifetime = timedelta(minutes=5)

def save_images(photo):
    hash_photo=secrets.token_urlsafe(10)
    _, file_extension=os.path.splitext(photo.filename)
    photo_name=hash_photo+file_extension
    file_path=os.path.join(current_app.root_path,'static/images',photo_name)
    photo.save(file_path)
    return photo_name

@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    session.permanent = True
    mydb=mysql.connector.connect(
        host="localhost",
        user="root",
        #password="",
        database="kaarigarkiduniya"
    )
    mycursor = mydb.cursor(dictionary=True)
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        typo=request.form['typo']
        #print(typo)
        if typo=='seller':
            mycursor.execute("SELECT * FROM loginreg WHERE username = '"+username+"' AND password = '"+password+"'")
            account = mycursor.fetchone()
            if account:
                session['loggedin'] = True
                session['id'] = account['seller_id']
                session['username'] = account['username']
                msg = 'Logged in successfully !'
                return redirect('/sellerdash')
            else:
                msg = 'Incorrect username / password !'
            return render_template('login.html', msg = msg)
        else:
            mycursor.execute("SELECT * FROM logincust WHERE username = '"+username+"' AND password = '"+password+"'")
            account = mycursor.fetchone()
            if account:
                session['loggedin'] = True
                session['id'] = account['cust_id']
                session['username'] = account['username']
                msg = 'Logged in successfully !'
                return redirect('/custdash')
            else:
                msg = 'Incorrect username / password !'
                return render_template('login.html', msg = msg)
    return render_template('login.html', msg = msg)


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))



@app.route('/register', methods =['GET', 'POST'])
def register():
    mydb=mysql.connector.connect(
        host="localhost",
        user="root",
        #password="",
        database="kaarigarkiduniya"
    )
    msg = ''
    mycursor = mydb.cursor(dictionary=True)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        dob = request.form['dob']
        l_name = request.form['l_name']
        f_name = request.form['f_name']
        phone = request.form['phone']
        city = request.form['city']
        state = request.form['state']
        typo=request.form['typo']

        if typo=='seller':
            mycursor.execute("SELECT * FROM loginreg WHERE username = '"+username+"'")
            account = mycursor.fetchone()
            if account:
                msg = 'Account already exists !'
                return render_template('register.html', msg = msg)
            elif not re.match(r'[A-Za-z0-9]+', username):
                msg = 'Username must contain only characters and numbers !'
                return render_template('register.html', msg = msg)
            elif not username or not password:
                msg = 'Please fill out the form !'
                return render_template('register.html', msg = msg)
            else:
                mycursor.execute("INSERT INTO loginreg (f_name, l_name, username, password, phone, city,state,dob) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(f_name,l_name,username,password,phone,city,state,dob))
                mydb.commit()
                mycursor.close()
                msg = 'You have successfully registered !'
                return redirect('/login')
        else:
            mycursor.execute("SELECT * FROM logincust WHERE username = '"+username+"'")
            account = mycursor.fetchone()
            if account:
                msg = 'Account already exists !'
                return render_template('register.html', msg = msg)
            elif not re.match(r'[A-Za-z0-9]+', username):
                msg = 'Username must contain only characters and numbers !'
                return render_template('register.html', msg = msg)
            elif not username or not password:
                msg = 'Please fill out the form !'
                return render_template('register.html', msg = msg)
            else:
                mycursor.execute("INSERT INTO logincust (f_name, l_name, username, password, phone, city,state,dob) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(f_name,l_name,username,password,phone,city,state,dob))
                mydb.commit()
                mycursor.close()
                msg = 'You have successfully registered !'
                return redirect('/login')
        
    else:
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg)




@app.route("/home")
def home():
    if 'loggedin' in session:
        print('heloooo')
        name='rohan'
        return render_template('index.html',name1=name)
    return redirect('/login')


@app.route("/contact")
def contact():
    return render_template('contact.html')


@app.route("/sellerdash")
def sellerdash():
    if 'loggedin' in session:
        return render_template('sellerdash.html')
    return redirect('/login')



@app.route("/prevup", methods=['GET'])
def prevup():
    if 'loggedin' in session:
        mydb=mysql.connector.connect(
            host="localhost",
            user="root",
            #password="",
            database="kaarigarkiduniya"
        )
        msg = ''
        mycursor = mydb.cursor(dictionary=True)
        mycursor.execute("SELECT * FROM prev_sold WHERE seller_id = %s",(session['id'],))
        account = mycursor.fetchall()
        #print(account)
        if account:
            msg='got values!'
            data=[]
            img1=[]
            for i in account:
                #print("i= ",i)
                mycursor.execute("SELECT * FROM pro_table_seller WHERE product_id = %s",(i['product_id'],))
                dat=mycursor.fetchone()
                path='static/images'
                full = os.path.join(path,dat['image'])
                img1.append(full)
                data.append(dat)
                #print(img1)
            return render_template('prevup.html',data=data,user_img=img1)
        else:
            msg='NO PRODUCTS UPLOADED YET!'
        return redirect('/sellerdash')
    return redirect('/login')



@app.route("/uploadpic")
def uploadpic():
    if 'loggedin' in session:
        return render_template('uploadpic.html')
    return redirect('/login')

@app.route("/upload", methods=['POST'])
def upload():
    if 'loggedin' in session:
        if request.method=='POST':
            name=request.form.get('name')
            description=request.form.get('description')
            cost_price=request.form.get('cost_price')
            shipping=request.form.get('shipping')
            brand=request.form.get('brand')
            category=request.form.get('category')
            size=request.form.get('size')
            colour=request.form.get('colour')
            material=request.form.get('material')
            product_type=request.form.get('product_type')
            '''
            with open("myntradata.csv","r") as csvfile:
                data_txt = csvfile.read().splitlines()
                last_row = data_txt[58530].split()
            print(last_row)
            selling_price = last_row[-5]
            print(selling_price)
            '''
            selling_price=0
            import csv
            f=open('spdata.csv','r')
            reader=csv.reader(f)
            l=[]
            for row in reader:
                l.append(row)
            print(l)
            for i in range(1,15):
                print(l[i][0])
                print(cost_price)
                if((l[i][1])==cost_price):
                    selling_price=(l[i][2])
                    break
            print(selling_price)
            profit=(float(selling_price)-float(cost_price))
            #print(profit)
            #photo=save_images(request.files.get('photo'))
            mydb=mysql.connector.connect(
                host="localhost",
                user="root",
                #password="",
                database="kaarigarkiduniya"
            )
            msg = ''
            total_sale='0'
            mycursor = mydb.cursor(dictionary=True)
            mycursor.execute("INSERT INTO pro_table_seller (name,description,cost_price,shipping,brand,category,size,colour,material,product_type,sell_price) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(name,description,cost_price,shipping,brand,category,size,colour,material,product_type,selling_price))            
            mydb.commit()
            #mycursor.execute("SELECT product_id FROM pro_table_seller WHERE name= %s",(name,))
            #dat=mycursor.fetchone()
            #mycursor.execute("INSERT INTO prev_sold (seller_id,product_id) VALUES (%s,%s)",(session['id'],dat['product_id']))
            #mydb.commit()
            msg = 'You have successfully inserted !'
            return render_template('pricepred.html',name=name,desc=description,cp=cost_price,sp=selling_price,profit=profit)
        msg='error occured'
        return render_template('uploadpic.html',msg=msg)
    return redirect('/login')


@app.route("/checkout")
def checkout():
    return render_template('checkout.html')


@app.route("/cart")
def cart():
    return render_template('cart.html')

@app.route("/blog")
def blog():
    return render_template('blog-single-sidebar.html')



@app.route("/custdash")
def custdash():
    if 'loggedin' in session:
        return render_template('custdash.html')
    
    return redirect('/login')

@app.route("/kurtidisp")
def kurtidisp():
    if 'loggedin' in session:
        return render_template('kurtidisp.html')
    return redirect('/login')

@app.route("/shawldisp")
def shawldisp():
    if 'loggedin' in session:
        return render_template('shawldisp.html')
    return redirect('/login')

@app.route("/sareedisp")
def sareedisp():
    if 'loggedin' in session:
        return render_template('sareedisp.html')
    return redirect('/login')

@app.route("/plazzodisp")
def plazzodisp():
    if 'loggedin' in session:
        return render_template('plazzodisp.html')
    return redirect('/login')

@app.route("/blousedisp")
def blousedisp():
    if 'loggedin' in session:
        return render_template('blousedisp.html')
    return redirect('/login')



@app.route("/predict",methods=['POST'])
def predict():
    if request.method =='POST':
        file =request.files['photo']
        filename=file.filename
        file.save(filename)
    image_path = filename
    orig = mpimg.imread(image_path)
    print("[INFO] Image Loaded")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    # important! otherwise the predictions will be '0'
    image = image / 255
    
    image = np.expand_dims(image, axis=0)
    # build the VGG16 network
    model =tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(rescale=1. / 255)
#Default dimensions we found online
    img_width, img_height = 224, 224 
    # batch size used by flow_from_directory and predict_generator 
    batch_size = 16
    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)
    train_data_dir = 'flask/v_data/train' 
    generator_top = datagen.flow_from_directory( train_data_dir,target_size=(img_width, img_height),batch_size=batch_size,class_mode='categorical',shuffle=False) 
    num_classes = len(generator_top.class_indices) 
    top_model_weights_path = 'bottlenecksss_fc_model.h5'
    # build top model  
    model = Sequential()  
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
    model.add(Dense(100, activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(num_classes, activation='softmax'))  
    model.load_weights(top_model_weights_path)  
    # use the bottleneck prediction on the top model to get the final classification  
    class_predicted = model.predict_classes(bottleneck_prediction)
    inID = class_predicted[0]  
    class_dictionary = generator_top.class_indices  
    inv_map = {v: k for k, v in class_dictionary.items()}  
    label = inv_map[inID]  
    # get the prediction label  
    print("Image ID: {}, Label: {}".format(inID, label))
    if label=='blouses':
        return redirect('/blousedisp')
    if label=='kurtas':
        return redirect('/kurtidisp')
    if label=='plazzo':
        return redirect('/plazzodisp')
    if label=='shawl':
        return redirect('/shawldisp')
    return redirect("/custdash")



if __name__=='__main__':
    app.run(debug=True)
