import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
x = tf.transpose(tf.constant([x1, x2], dtype=tf.float32))

and_y = tf.constant([0, 0, 0, 1], dtype=tf.float32)
or_y = tf.constant([0, 1, 1, 1], dtype=tf.float32)
xor_y = tf.constant([0, 1, 1, 0], dtype=tf.float32)

print(f"x : \n{x}")
print(f"AND y:\t{and_y}", f"OR y:\t{or_y}", f"XOR y:\t{xor_y}")

sns.set_style("darkgrid")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# AND 
sns.scatterplot(x=x[:,0], y=x[:,1], hue=and_y, ax=axs[0])
axs[0].set_title("AND Problem")
axs[0].set_xlabel("X1")
axs[0].set_ylabel("X2", rotation=0)

# OR
sns.scatterplot(x=x[:,0], y=x[:,1], hue=or_y, ax=axs[1])
axs[1].set_title("OR Problem")
axs[1].set_xlabel("X1")
axs[1].set_ylabel("X2", rotation=0)

# XOR
sns.scatterplot(x=x[:,0], y=x[:,1], hue=xor_y, ax=axs[2])
axs[2].set_title("XOR Problem")
axs[2].set_xlabel("X1")
axs[2].set_ylabel("X2", rotation=0)

plt.tight_layout()
# plt.show()

# Declare MLP Model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid')
# ])


################
# AND Prediction
################
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss='binary_crossentropy', metrics=['accuracy'],)
# model.fit(x, and_y, epochs=100, batch_size=4)
# loss, accuracy = model.evaluate(x, and_y)
# print(f"Loss : {loss}, Accuracy : {accuracy}")
# predictions = model.predict(x)
# print(f"Predictions : \n{predictions}")

################
# OR Prediction
################
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss='binary_crossentropy', metrics=['accuracy'],)
# model.fit(x, or_y, epochs=100, batch_size=4)
# loss, accuracy = model.evaluate(x, or_y)
# print(f"Loss : {loss}, Accuracy : {accuracy}")
# predictions = model.predict(x)
# print(f"Predictions : \n{predictions}")

################
# XOR Prediction
################
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss='binary_crossentropy', metrics=['accuracy'],)
# model.fit(x, xor_y, epochs=100, batch_size=4)
# loss, accuracy = model.evaluate(x, xor_y)
# print(f"Loss : {loss}, Accuracy : {accuracy}")
# predictions = model.predict(x)
# print(f"Predictions = {predictions}")

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, xor_y, epochs=100, batch_size=4)

loss, accuracy = model.evaluate(x, xor_y)
print(f"Loss : {loss}, Accuracy : {accuracy}")
predictions = model.predict(x)
print(f"Predictions : \n{predictions}")
