import model_creation
import preprocessing_1
from matplotlib import pyplot as plt
import numpy as np  
import tensorflow as tf
import cv2

model=model_creation.model()
p=preprocessing_1.IMVO ()

class_names=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,47]
    

output=[]
image_1=[]
    #loop through the list l and save the image 
for i in range(len(p)):
    o=p[i].shape
    print(o[0])
    if o[0] == 0:
        pass
    else:
        j=p[i]
        j = cv2.GaussianBlur(j, (3,3), 2)
        image = cv2.resize(j, (28, 28)) 
        
        #cv2.imshow("y",image)
        j=image.reshape(-1,28,28)
        print(j.shape)
        image_1.append(j)
        yhat = model.predict(j)
        pred_class = class_names[np.argmax(yhat)]
        output.append(pred_class)
                
        print(pred_class)
    
    
        
    
    
dic={0:"\u004f",1:"\u0031",2:"\u0032",3:"\u0033",4:"\u0034",5:"\u0035",6:"\u0036",7:"\u0037",8:"\u0038",9:"\u0039",10:"\u0041",11:"\u0042",12:"\u0043",13:"\u0044",14:"\u0045",15:"\u0046",16:"\u0047",
         17:"\u0048",18:"\u0049",19:"\u004a",20:"\u004b",21:"\u004c",22:"\u004d",23:"\u004e",24:"\u004f",25:"\u0050",26:"\u0051",27:"\u0052",28:"\u0053",29:"\u0054",30:"\u0055",
         31:"\u0056",32:"\u0057",33:"\u0058",34:"\u0059",35:"\u0059",36:"\u0061",37:"\u0062",38:"\u0064",39:"\u0065",40:"\u0066",41:"\u0067",42:"\u0068",43:"\u006e",
         44:"\u0071",45:"\u0072",46:"\u0074",47:"\u0020"}
z=[]
for i in output:
    for key,values in dic.items():
            if i == key:
                print(values)
                z.append(values)
  
w = 10
h = 10
fig = plt.figure(figsize=(20, 20))
columns =10
rows = 10
for i in range(len(image_1)):
    img = np.random.randint(10, size=(h,w))
    f=i+1
    fig.add_subplot(rows, columns, f)  
    d=image_1[i].reshape(28,28)
    r=z[i]
    plt.title(r)
    plt.imshow(d) 
plt.show()

file = open(r"C:\Users\user33\Desktop\neha\output\out.txt", "w+", encoding='utf-8') 
#  the values in the file
for i in z:
    file.write(i)  
file.close( )



import cv2
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding,Activation
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 


train=pd.read_csv(r"C:\Users\user33\Desktop\WORK\-cap\code\emnist-balanced-train.csv")
train=pd.DataFrame(train)


test=pd.read_csv(r"C:\Users\user33\Desktop\WORK\-cap\code\emnist-balanced-test.csv")
test=pd.DataFrame(test)


img_test=[]
label_test=[]
img_train=[]
label_train=[]


for index, row in train.iterrows():
    row=row["45"]
    label_train.append(row)
    
    
for index, row in test.iterrows():
    row=row["41"]
    label_test.append(row)


no_lable_train=train.drop(['45'], axis = 1)
no_lable_test=test.drop(['41'], axis = 1)


for index, row in no_lable_train.iterrows():
    row=row.values.reshape(28,28)
    row=row.T
    img_train.append(row)
    
    
for index, row in no_lable_test.iterrows():
    row=row.values.reshape(28,28)
    row=row.T
    img_test.append(row)
    

img_test=np.array(img_test)
label_test=np.array(label_test)
img_train=np.array(img_train)
label_train=np.array(label_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),padding='same',activation = "relu" , input_shape = (28,28,1)) ,
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation = "relu") ,  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(1000,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(752,activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.1,seed = 2019),
    tf.keras.layers.Dense(376,activation ="relu"),
    tf.keras.layers.Dropout(0.3,seed = 2019),
    tf.keras.layers.Dense(188,activation="relu"),
    tf.keras.layers.Dropout(0.4,seed = 2019),
    tf.keras.layers.Dense(94,activation ="relu"),
    tf.keras.layers.Dropout(0.2,seed = 2019),
    tf.keras.layers.Dense(48,activation = "softmax")   #Adding the Output Layer
])


opt = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],)
model.fit(img_train,
          label_train,
          epochs=15,batch_size=50,
          validation_data=(img_test, label_test))


class_names=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,47]




    
    
    

import cv2
from imutils import contours
import string
import numpy as np
blank=cv2.imread(r'C:\Users\user33\Desktop\CLIENT\neha client 3\code\blank.png')
blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
image = cv2.imread(r'C:\Users\user33\Desktop\WORK\-cap\code\IMG_20220108_155152.JPG')
image = cv2.resize(image, (700, 700))  



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur the image
blur = cv2.GaussianBlur(gray, (5,5), 1)


#set a treshold value
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,29,38)



#set a kernal value 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,15))

dilate = cv2.dilate(thresh, kernel, iterations=3)

cv2.waitKey()
# Find contours, highlight text areas, and extract ROIs
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
l=[]
ROI_number = 0
#("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
(cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")


for c in cntts:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            ROI = image[y:y+h, x:x+w]

            l.append(ROI)
#loop through the list l and save the image 
for i in range(len(l)):
    cv2.imwrite(f'C:/Users/user33/Desktop/CLIENT/neha client 3/code/out/line/image{i}.png',l[i])
    
d=[]
for i in l:
    i = cv2.resize(i, (500, 100)) 

    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)


    #blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 1)

    #set a treshold value
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,21)
 
    #set a kernal value 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,7))


    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 0
    #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
    (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
    (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
        
    for c in cnttts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        ROI = i[y:y+h, x:x+w]
        d.append(ROI)
#loop through the list l and save the image 
b=[]
for i in d:
    f=i.shape
    x=f[0]
    print(x)
    if x <11:
        pass
    else:
        b.append(i)
for i in b:
    print(i.shape)
    
for i in range(len(b)):
    cv2.imwrite(f'C:/Users/user33/Desktop/CLIENT/neha client 3/code/out/word/image{i}.png',b[i])
cv2.waitKey()

p=[]
for i in b:
    r = cv2.resize(i, (250, 150)) 
    r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    r = cv2.GaussianBlur(r, (1,3), 1)
    r = cv2.convertScaleAbs(r, alpha=1.5, beta=50)

    cv2.waitKey(0) 
    #blur the image
    blur = cv2.GaussianBlur(r, (3,3), 1)



    #set a treshold value
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,61,23)


    #set a kernal value 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
    cv2.waitKey(0) 
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 0
    #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):

    (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
    (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
        
    for c in cnttts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        h=h+14
        w=w+14
        x=x-6
        y=y-6
        a=cv2.rectangle(r, (x, y), (x + w, y + h), (255,255,255),0)
        ROI = dilate[y:y+h, x:x+w]
        p.append(ROI)
    p.append(blank)

out=[]
inr=[]
#loop through the list l and save the image 
for i in range(len(p)):
        o=p[i].shape
        print(o[0])
        if o[0] == 0:
            pass
        else:
            j=p[i]
            j = cv2.GaussianBlur(j, (3,3), 2)
            image = cv2.resize(j, (28, 28)) 
    
            #cv2.imshow("y",image)
            j=image.reshape(-1,28,28)
            print(j.shape)
            inr.append(j)
            yhat = model.predict(j)
            pred_class = class_names[np.argmax(yhat)]
            out.append(pred_class)
            
            print(pred_class)

cv2.waitKey()

    


dic={0:"\u0030",1:"\u0031",2:"\u0032",3:"\u0033",4:"\u0034",5:"\u0035",6:"\u0036",7:"\u0037",8:"\u0038",9:"\u0039",10:"\u0041",11:"\u0042",12:"\u0043",13:"\u0044",14:"\u0045",15:"\u0046",16:"\u0047",
     17:"\u0048",18:"\u0049",19:"\u004a",20:"\u004b",21:"\u004c",22:"\u004d",23:"\u004e",24:"\u004f",25:"\u0050",26:"\u0051",27:"\u0052",28:"\u0053",29:"\u0054",30:"\u0055",
     31:"\u0056",32:"\u0057",33:"\u0058",34:"\u0059",35:"\u0059",36:"\u0061",37:"\u0062",38:"\u0064",39:"\u0065",40:"\u0066",41:"\u0067",42:"\u0068",43:"\u006e",
     44:"\u0071",45:"\u0072",46:"\u0074",47:"\u0020"}
z=[]
for i in out:
    for key,values in dic.items():
        if i == key:
            print(values)
            z.append(values)
            
            
w = 10
h = 10
fig = plt.figure(figsize=(20, 20))
columns =10
rows = 10
for i in range(len(inr)):
    img = np.random.randint(10, size=(h,w))
    print(i)
    f=i+1
    fig.add_subplot(rows, columns, f)
    d=inr[i].reshape(28,28)
    r=z[i]
    plt.title(r)
    plt.imshow(d)
plt.show()


file = open(r"C:\Users\user33\Desktop\file.txt", "w+", encoding='utf-8') 
for i in z:
    print(i)
    file.write(i)
file.close()


import cv2
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sparsenet.core import sparse
from sklearn.metrics import precision_recall_fscore_support as score



def model():
    train=pd.read_csv(r"C:\Users\user33\Desktop\CLIENT\neha\code\dataset\emnist-balanced-train.csv")
    train=pd.DataFrame(train)
    
    
    test=pd.read_csv(r"C:\Users\user33\Desktop\CLIENT\neha\code\dataset\emnist-balanced-test.csv")
    test=pd.DataFrame(test)
     
    
    img_test=[]
    label_test=[]
    img_train=[]
    label_train=[]
    
    
    for index, row in train.iterrows():
        row=row["45"]
        label_train.append(row)
        
    
    for index, row in test.iterrows():
        row=row["41"]
        label_test.append(row)
    
    
    no_lable_train=train.drop(['45'], axis = 1)
    no_lable_test=test.drop(['41'], axis = 1)
    
    
    for index, row in no_lable_train.iterrows():
        row=row.values.reshape(28,28)
        row=row.T
        img_train.append(row)
        
        
    for index, row in no_lable_test.iterrows():
        row=row.values.reshape(28,28)
        row=row.T
        img_test.append(row)
        
    
    img_test=np.array(img_test)
    label_test=np.array(label_test)
    img_train=np.array(img_train)
    label_train=np.array(label_train)
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),padding='same',activation = "relu" , input_shape = (28,28,1)) ,
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),padding='same',activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),padding='same',activation = "relu"),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),  
        tf.keras.layers.Dense(1000,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(752,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(376,activation ="relu"),
        tf.keras.layers.Dropout(0.3,seed = 2019),
        tf.keras.layers.Dense(188,activation="relu"),
        tf.keras.layers.Dropout(0.4,seed = 2019),
        tf.keras.layers.Dense(94,activation ="relu"),
        tf.keras.layers.Dropout(0.2,seed = 2019),
        sparse(60, activation="relu"),#Adding the Output Layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(48,activation = "softmax") 
    
    ])
    
    
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],)
    model.fit(img_train,
              label_train,
              epochs=10,
              validation_data=(img_test, label_test))
    y_true=label_test
    class_names=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,47]
    li=[]
    import time
    start_time = time.time()
    for i in img_test:
        image = cv2.resize(i, (28, 28)) 
        i=image.reshape(-1,28,28)
        y_pred=model.predict(i)
        pred_class = class_names[np.argmax(y_pred)]
        li.append(pred_class)
        print(time.time() - start_time)
    y_pred=np.array(li)

    precision,recall,fscore,support=score(y_true,y_pred,average='macro')
    print ('Precision : {}'.format(precision))
    print ('Recall    : {}'.format(recall))
    print ('F-score   : {}'.format(fscore))
    return model
    

import cv2
import model_creation
from imutils import contours
import numpy as np
import keras
import tensorflow as tf

def IMVO():

    blank=cv2.imread(r'./input image/code/blank.png')
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    
    
    
    image = cv2.imread(r'.\input image\53db32076c4bb6b8aef3f2579d6dac6b.jpg')
    image = cv2.resize(image, (700, 700))  
    
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    
    
    #set a treshold value
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,29,38)
    
    
    
    #set a kernal value 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,15))
    
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    
    cv2.waitKey()
    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    l=[]

    #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
    (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
    
    
    for c in cntts:

                x,y,w,h = cv2.boundingRect(c)
                ROI = image[y:y+h, x:x+w]
    
                l.append(ROI)
    #loop through the list l and save the image 
    for i in range(len(l)):
        cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/line/image{i}.png',l[i])
        
    d=[]
    for i in l:
        i = cv2.resize(i, (500, 100)) 
    
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    
    
        #blur the image
        blur = cv2.GaussianBlur(gray, (5,5), 1)
    
        #set a treshold value
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,21)
     
        #set a kernal value 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,7))
    
    
        dilate = cv2.dilate(thresh, kernel, iterations=2)
    
        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
        (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
        (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
            
        for c in cnttts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = i[y:y+h, x:x+w]
            d.append(ROI)
    #loop through the list l and save the image 
    b=[]
    for i in d:
        f=i.shape
        x=f[0]
        print(x)
        if x <11:
            pass
        else:
            b.append(i)
    for i in b:
        print(i.shape)
        
    for i in range(len(b)):
        cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/word/image{i}.png',b[i])
    cv2.waitKey()
    
    p=[]
    for i in b:
        r = cv2.resize(i, (250, 150)) 
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        r = cv2.GaussianBlur(r, (1,3), 1)
        r = cv2.convertScaleAbs(r, alpha=1.5, beta=50)
    
        cv2.waitKey(0) 
        #blur the image
        blur = cv2.GaussianBlur(r, (3,3), 1)
    
    
    
        #set a treshold value
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,61,23)
    
    
        #set a kernal value 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
        cv2.waitKey(0) 
        dilate = cv2.dilate(thresh, kernel, iterations=1)
    
        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
    
        (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
        (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
            
        for c in cnttts:
            x,y,w,h = cv2.boundingRect(c)
            h=h+14
            w=w+14
            x=x-6
            y=y-6
            ROI = dilate[y:y+h, x:x+w]
            p.append(ROI)
        p.append(blank)
        
        for i in range(len(p)):
           cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/letter/image{i}.png',p[i]) 

    
    
    #return p
IMVO()

