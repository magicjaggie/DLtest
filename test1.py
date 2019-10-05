# test of DL

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#load the dataset
dataset = loadtxt('prima-indians-diabetes.data.cvs', delimiter= ',')

#split the dataset
X = dataset(:,0:8) #toutes les lignes des 7 premieres colonnes
Y = dataset(:,8) #toutes les lignes de la 8eme colonne

# Define the Keras model
model = Sequential()
model.add(Dense(12,inputs_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid')) # sigmoid to ensure the output is between 0 and 1

# Compilation of the keras model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# the error of the current state of the model is estimated repeatedly -> error (or loss) function
# here, the cross entropy is chosen as the loss argument (cross-entropy for binary classification)

