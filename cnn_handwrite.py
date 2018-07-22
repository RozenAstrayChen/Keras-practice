#%%
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import random

#%%
(x_train, y_train),(x_test, y_test) = mnist.load_data('mnist.pkl.gz')
randomInteger = random.randint(0,len(x_test))
x = x_test[randomInteger]
#28*28 = 784
x_train = x_train.reshape(60000, 784)/125
# /125 data
x_test = x_test.reshape(x_test.shape[0], 784)/125

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
#model
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
#optimzers,lost func
rms = optimizers.RMSprop(lr=0.01, epsilon=1e-8, rho=0.9)
#%%
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
# training
model.fit(x_train, y_train, batch_size=20, epochs=2, shuffle=True)
# evalutate
score = model.evaluate(x_test, y_test, batch_size=500)
h = model.predict_classes(x.reshape(1, -1),batch_size=1)
#%%

print ('loss:\t', score[0], '\naccuracy:\t', score[1])
print ('\nclass:\t', h)
plt.imshow(x)
plt.show()
