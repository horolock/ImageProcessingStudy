import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from keras import Sequential
from keras.src.layers import Dense

# Example
# Predict `Slope` and `Intercept` value that using single perceptron.
# In here, using Y = 1.5X + 3

x = tf.random.uniform(shape=[100], minval=1, maxval=4)

slope = 1.5
intercept = 3
epsilon = tf.random.truncated_normal(shape=[100], mean=0, stddev=0.3)
y = slope * x + intercept + epsilon

print(y)

sns.set_style("darkgrid")
plt.scatter(x, y)
plt.xlabel('Feature (X)')
plt.ylabel('Label (Y)')
plt.title('synthetic dataset')

plt.show()

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(loss='mean_squared_error', optimizer='sgd')

x_train = tf.reshape(x, (-1, 1)) 
print(x_train.shape)

history = model.fit(x_train, y, epochs=500)

weights, bias = model.get_weights()

print("Weights (Slope) : ", weights)
print("Bias (Intercept) : ", bias)