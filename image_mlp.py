from keras.models import Sequential
from keras.layers import Dense, Flatten


# Define the model
model = Sequential()

# Add the Flatten layer (input layer)
model.add(Flatten(input_shape=(28, 28))) 

# Add two Dense layers with 512 nodes each
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

# Add output layer with 10 nodes
model.add(Dense(10, activation='softmax'))

# Print summary of the model
model.summary()
