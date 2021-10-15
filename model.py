from RL import Sequential, Dense, Dropout, SGD, np, os
from train_test import train_x, train_y

# Build the deep neural network model
model = Sequential([
                    Dense(128, input_shape=(len(train_x[0]), ), activation='relu'),
                    Dropout(0.5), # reduce the model rate to 0.5 to reduce the overfitting
                    Dense(64, activation='relu'),
                    Dropout(0.5), # reduce the model rate to 0.5 to reduce the overfitting
                    Dense(len(train_y[0]), activation='softmax'),
            ]) 

# Initialize an optimizer
## Stochastic gradient descent (SGD) with Nesterov Accelerated Gradient gives good result and speed to the model
sgd_nag = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Start to compile the model to prepare the training dataset for the fitting process
model.compile(optimizer=sgd_nag, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting or training the model
hist = model.fit(x=np.array(train_x), y=np.array(train_y), batch_size=5, epochs=200, verbose=2)

# Save the model
if os.path.isfile('chatPyMe.h5') is False:
        model.save(os.path.abspath('chatPyMe.h5'), hist)

print("Model Created!")