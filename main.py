import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from keras import datasets, models, layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
import cv2



#LOAD OF THE DATA SET, USING CIFAR10 DATA
(X_train, y_train),(X_test, y_test) = datasets.cifar10.load_data()
print("TRAINING DATA SET HAS (IMAGES,X,Y,RGB) : " + str(X_train.shape))
print("TEST DATA SET HAS (IMAGES,Xpxl,Ypxl,RGB) : " + str(X_test.shape))

X_trainLog = np.array([cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in X_train])
X_trainLog = np.delete(X_trainLog,1,0)
y_trainLog = y_train
print("TRAINING DATA SET HAS (IMAGES,X,Y,RGB) : " + str(X_trainLog.shape))












#SHOW A IMAGE OF THE DATA SET (EXAMPLE)
print()
print("IN THE Y_TEST ARRAY WE HAVE THE LABLE OF THE IMAGE STORED IN THE X_TEST: 2 DIMENSION ARR")
print(y_train[:5])

print("CONVERTING THE 2 DIMENSION ARR INTO A 1 DIMENSION")
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
print(y_train[:5])


print("SHOW A IMAGE OF THE DATA SET WITH ITS LABEL")
label_name = ["airplain","automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship","truck"] #THIS LABLES ARE GIVEN BY THE DATA SET AT: https://www.cs.toronto.edu/~kriz/cifar.html
#DEFINING A FUNCTION THAT GIVEN A DATA SET OF IMAGES AND ITS LABELS IN ORDER, THE FUNCTION WILL SHOW (PLOT->MATPLOTLIB) THE IMAGE AND ITS LABEL
def plot_sample(X_train,y_train,index):
    plt.imshow(X_train[index]) #LOADS THE IMAGE
    plt.xlabel(label_name[y_train[index]]) #ADDS A LABLE TO THE X PART OF THE PLOT
    plt.show() #DISPLAYS THE PLOT

#USING THE FIRST IMG AS AN EXAMPLE
#plot_sample(X_train,y_train,0)






#NORMALIZING THE IMAGE MATRIX INTO A 0 TO 1 RANGE -> RGB GOES FROM 0 - 255 AND SO I DIVIDE BY 255
print()
print("BEFORE NORMALIZING: ")
print(X_train[0])
print()
print("AFTER NORMALIZING: ")
#DIVIDES EVERY ELEMENT IN THE ARRAY BY 255
X_train = X_train/255
X_test = X_test/255
print(X_train[0])

#ONLY FOR CHECKING I CREATED THIS SIMPLE NEURAL NETWORK
#CREATION OF A NORMAL NEURAL NETWORK - NO CNN - ACURRACY IS TO LOW -> BAD PERFORMANCE
# ann = models.Sequential([
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(3000, activation='relu'),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(10, activation='sigmoid')
# ])
#
# ann.compile(
#             optimizer= 'SGD',
#             loss= 'sparse_categorical_crossentropy',
#             metrics= ['accuracy'])
#
# ann.fit(X_train,y_train,epochs=5)


#CNN DOES THE SAME BUT USING CONVULATION
cnn = models.Sequential([
    #CNN NETWORK
    layers.Conv2D(filters=32, activation="relu" ,kernel_size=(3,3), input_shape=(32,32,3)), #MAKING THE CONVULATION -> IT WILL FIGURE OUT THE KIND OF FILTERS YOU PASS HOW MANY FILTERS YOU WANT, THE ACTIVATION METHOD (RELU), THE SIZE OF THE FILTER MATRIX, AND THE SHAPE OF THE IMAGE
    layers.MaxPool2D((2,2)), #DOWN-SAMPLING OPERATION THAT REDUCES THE DIMENSIONALITY OF THE FEATURE MAP

        #TWO CONV AND POOL LAYERS FOR IMPROVING ACURRACY
    layers.Conv2D(filters=32, activation="relu" ,kernel_size=(3,3), input_shape=(32,32,3)), #MAKING THE CONVULATION -> IT WILL FIGURE OUT THE KIND OF FILTERS YOU PASS HOW MANY FILTERS YOU WANT, THE ACTIVATION METHOD (RELU), THE SIZE OF THE FILTER MATRIX, AND THE SHAPE OF THE IMAGE
    layers.MaxPool2D((2,2)), #DOWN-SAMPLING OPERATION THAT REDUCES THE DIMENSIONALITY OF THE FEATURE MAP


    #DENSE
     layers.Flatten(), #THE NETWORK WILL MANAGE AUTOMATICALLY THE SHAPE
    #I DONT NEED TO MANY NEURONS BECAUSE CNN WILL DO MOST OF THE WORK
     layers.Dense(64, activation='relu'),
     layers.Dense(10, activation='softmax') #SOFTMAX WILL NORMALIZE THE PROBABILITY TO 1!!
 ])

cnn.compile(optimizer="adam", #OPTIMIZER ALGORITHM, MOST POPULAR USE IN CNN
            loss= 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

#TRAIN THE NETWORK
cnn.fit(X_train,y_train,epochs=20)

#TRY THE NETWORK WITH ALL IMAGES IN TEST
print()
print("EVALUATE THE NETWORK FOR ALL THE 'TEST' EXAMPLES")
cnn.evaluate(X_test,y_test)
print()

#SHOWING THE NETWORK RESULTS
prediction = cnn.predict(X_test)
print("SHOWING THE PREDICTION FOR THE FIRST FIVE IMAGES AS NUMBERS")
print(prediction[:5])
print("SHOWING THE PREDICTION FOR THE FIRST FIVE IMAGES AS TYPES")
y_classes = [np.argmax(element)for element in prediction]
for i in range(0,5):
    print(y_classes[i])
print("SHOWING THE ACTUAL REAL VALUE FOR THOSE FIVE IMAGES")
for i in range(0,5):
    print(y_test[i])





















#SMALL PROGRAM TO COMPARE ANY IMAGE GIVEN WITH THE PREDICTION GIVEN BY THE TRAINED NETWORK
program = str(input("WRITE 'EXIT' TO EXIT THE PROGRAM OR ANY NUMBER TO CONTINUE: "))
while(program.upper() != "EXIT"):
    usr = int(input("PUT THE NUMBER OF THE PICTURE YOU WANT TO COMPARE: "))
    plot_sample(X_test,y_test,usr)
    print(label_name[y_classes[usr]])
    program = str(input("WRITE 'EXIT' TO EXIT THE PROGRAM OR ANY LETTER TO CONTINUE: "))
