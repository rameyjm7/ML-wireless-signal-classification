import time
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from math import sqrt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

DEBUG = True
USE_GAN = True
LOAD_EXISTING_MODEL = False

MODEL_SAVE_PATH = "rnn_model.keras"
DIRECTORY = "../ML-wireless-signal-classification"
FILE_NAME = "RML2016.10a_dict.pkl"
SAVE_PLOTS_FLAG = 1
ACCGAN_PLOT = "accuracy_gan_plot.png"
ACCGAN_TEMP_PLOT = "accuracy_gan_plot.png"
ACCSNR_PLOT = "accuracy_snr_plot.png"
CONFMATRIX_PLOT = "confusion_matrix.png"
ACC_PLOT = "accuracy_plot.png"
LOSS_PLOT = "loss_plot.png"
RMLGAN_SAMPLE_TRIAN_IMAGES = 'RMLGAN2_real_samples_IQ_signals.png'

# -------------------------------
# Definitions
# -------------------------------
def define_model(myClasses, myInputShape, myLearningRate=0.001):
    if DEBUG:
        print("Classes:", myClasses)  # Classes: 11
        print("Input Shape:", myInputShape)  #
        print("Leaning Rate:", myLearningRate)  #

    initial_lr = myLearningRate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=15000,
        decay_rate=0.98,
        staircase=True  # optional: makes it decay in steps instead of smoothly
    )

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=myInputShape))
    model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(myClasses, activation="softmax"))

    optimizer = Adam(learning_rate=lr_schedule) #myLearningRate) #
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------------
# Generator and Discriminator Definitions
# -------------------------------
# define the standalone discriminator model
# def define_discriminator(in_shape=(28,28,1), n_classes=10):
#     # weight initialization
#     init = RandomNormal(stddev=0.02)
#     # image input
#     in_image = Input(shape=in_shape)
#     # downsample to 14x14
#     fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Dropout(0.5)(fe)
#     # normal
#     fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
#     fe = BatchNormalization()(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Dropout(0.5)(fe)
#     # downsample to 7x7
#     fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
#     fe = BatchNormalization()(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Dropout(0.5)(fe)
#     # normal
#     fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
#     fe = BatchNormalization()(fe)
#     fe = LeakyReLU(alpha=0.2)(fe)
#     fe = Dropout(0.5)(fe)
#     # flatten feature maps
#     fe = Flatten()(fe)
#     # real/fake output
#     out1 = Dense(1, activation='sigmoid')(fe)
#     # class label output
#     out2 = Dense(n_classes, activation='softmax')(fe)
#     # define model
#     model = Model(in_image, [out1, out2])
#     # compile model
#     opt = Adam(lr=0.0002, beta_1=0.5)
#     model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
#     return model

def define_discriminator(input_shape, n_classes):
    init = RandomNormal(stddev=0.02)
    in_signal = Input(shape=input_shape)  # shape: (128 timesteps, 2 channels: I and Q)

    x = Conv1D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(in_signal)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Conv1D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    # Output 1: real/fake
    out1 = Dense(1, activation='sigmoid', name='real_or_fake')(x)

    # Output 2: class prediction
    out2 = Dense(n_classes, activation='softmax', name='modulation_class')(x)

    model = Model(inputs=in_signal, outputs=[out1, out2])

    opt = Adam(learning_rate=0.00001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)

    return model

# define the standalone generator model
# def define_generator(latent_dim, n_classes=10):
#     # weight initialization
#     init = RandomNormal(stddev=0.02)
#     # label input
#     in_label = Input(shape=(1,))
#     # embedding for categorical input
#     li = Embedding(n_classes, 50)(in_label)
#     # linear multiplication
#     n_nodes = 7 * 7
#     li = Dense(n_nodes, kernel_initializer=init)(li)
#     # reshape to additional channel
#     li = Reshape((7, 7, 1))(li)
#     # image generator input
#     in_lat = Input(shape=(latent_dim,))
#     # foundation for 7x7 image
#     n_nodes = 384 * 7 * 7
#     gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
#     gen = Activation('relu')(gen)
#     gen = Reshape((7, 7, 384))(gen)
#     # merge image gen and label input
#     merge = Concatenate()([gen, li])
#     # upsample to 14x14
#     gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
#     gen = BatchNormalization()(gen)
#     gen = Activation('relu')(gen)
#     # upsample to 28x28
#     gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
#     out_layer = Activation('tanh')(gen)
#     # define model
#     model = Model([in_lat, in_label], out_layer)
#     return model
def define_generator(latent_dim, n_classes, output_shape=(128, 2)):
    init = RandomNormal(stddev=0.02)

    # Input: class label (integer)
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)        # Embed the class label
    li = Dense(latent_dim, kernel_initializer=init)(li)
    li = Reshape((latent_dim,))(li)

    # Input: latent noise vector
    in_lat = Input(shape=(latent_dim,))
    merge = Concatenate()([in_lat, li])            # Combine noise + label

    # Project and reshape
    n_nodes = 64 * 32  # 64 filters, 32 timesteps
    x = Dense(n_nodes, kernel_initializer=init)(merge)
    x = Activation('relu')(x)
    x = Reshape((32, 64))(x)

    # Upsample to 64 timesteps
    x = UpSampling1D()(x)
    x = Conv1D(64, kernel_size=5, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Upsample to 128 timesteps
    x = UpSampling1D()(x)
    x = Conv1D(32, kernel_size=5, padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final Conv layer to get 2 output channels (I and Q)
    x = Conv1D(output_shape[1], kernel_size=5, padding='same', kernel_initializer=init)(x)
    out_layer = Activation('tanh')(x)

    model = Model([in_lat, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(learning_rate=0.00001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model


# load images
#NOT USED
def load_real_samples():
    # load dataset
    (trainX, trainy), (_, _) = fashion_mnist.load_data()
    print('Train', trainX.shape, trainy.shape)
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    print('Train Scaled', X.shape, trainy.shape)
    return [X, trainy]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = np.random.randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = np.ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
# this just generates a 2D vector of random gaussian noise
# and a random class value per 2D vector (image)
# and returns n_samples of such vectors
# not sure I understand how 100 laten dims gets into a 28x28 image...
def generate_latent_points(latent_dim, n_samples, n_classes):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
# this generates n_nsamples of the noise vector w/ random class
# and generates the image with the generator per requested class
# it then AND adds a ground truth classifier (0 = fake) incase this was
# needed for training
def generate_fake_samples(generator, latent_dim, n_samples, n_classes):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, n_classes)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y


# generate samples and save as a plot and save the model
# this is just for routine epoch monitoring, not for traning.
# it generates 100 random noise vectors and then uses the generator.predict
# (at time of traning, once per epoch) to print out how well the generator is working
def summarize_performance(step, g_model, latent_dim, n_samples, n_classes):
    # prepare fake examples
    [X, y], _ = generate_fake_samples(g_model, latent_dim, n_samples, n_classes)
    # scale from [-1,1] to [0,1]
    #X = (X + 1) / 2.0
    # plot images
    grid_size = n_samples # must be square
    plt.figure(figsize=(15, 8))

    for idx in range(grid_size):
        real_sample = X[idx]
        label_idx = y[idx]
        mod_label = le.classes_[label_idx]
        # snr_value = snrtrain[idx]

        # define subplot
        side = int(sqrt(grid_size))
        plt.subplot(side, side, 1 + idx)

        # Plot I and Q components
        plt.plot(real_sample[:, 0], label="I")
        plt.plot(real_sample[:, 1], label="Q")
        plt.title(f"Mod: {mod_label}") #, SNR: {snr_value} dB")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    if SAVE_PLOTS_FLAG:
        # save plot to file
        filename1 = 'RMLGAN2_generated_plot_%04d.png' % (step + 1)
        plt.savefig(filename1)
        plt.close()
        print('Plot saved as', filename1)
    else:
        plt.show()

    # save the generator model
    filename2 = 'RMLGAN2_model_%04d.keras' % (step + 1)
    g_model.save(filename2)
    print('>Model Saved: %s' % filename2)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch, n_classes):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print('Batches Per Epoch:', bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, n_classes)
        # update discriminator model weights
        _, d_f1, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch, n_classes)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        # model return three loss values from the call to the train_on_batch() function.
        # The first value is the sum of the loss values and can be ignored,
        # whereas the second value (?_1) is the loss for the real/fake output layer
        # and the third value (?_2) is the loss for the clothing label classification.
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, d_r1, d_r2, d_f1, d_f2, g_1, g_2))
        # evaluate the model performance every 'epoch'
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, latent_dim, 16, n_classes)

def plot_sample_training_data(xtrain, ytrain, snrtrain, grid_size):

    plt.figure(figsize=(15, 8))
    for i in range(grid_size):
        # Randomly choose one sample
        idx = np.random.randint(len(xtrain))
        real_sample = xtrain[idx]
        label_idx = ytrain[idx]
        mod_label = le.classes_[label_idx]
        snr_value = snrtrain[idx]

        # define subplot
        side = int(sqrt(grid_size))
        plt.subplot(side, side, 1 + i)

        # Plot I and Q components
        plt.plot(real_sample[:, 0], label="I")
        plt.plot(real_sample[:, 1], label="Q")
        plt.title(f"Mod: {mod_label}, SNR: {snr_value} dB")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    if SAVE_PLOTS_FLAG:
        plt.savefig(RMLGAN_SAMPLE_TRIAN_IMAGES)
        print('Plot saved as', RMLGAN_SAMPLE_TRIAN_IMAGES)
    else:
        plt.show()

####################### START OF CODE  #####################
start_time = time.time()

file = os.path.join(DIRECTORY, FILE_NAME)
data = None

try:
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
except FileNotFoundError:
    print(f"Error: The file '{file}' was not found.")
except Exception as e:
    print(f"Error: {e}")

if data is None:
    raise ValueError("Dataset failed to load.")

if DEBUG:
    print(f"Dataset keys: {data.keys()}")

# Flatten data and extract features, labels
X = []
y = []
SNRs = []

for (mod_type, snr), signals in data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T
                #snr_signal = np.full((128, 1), snr)
                #combined_signal = np.hstack([iq_signal, snr_signal])
                #X.append(combined_signal)
                X.append(iq_signal)
                y.append(mod_type)
                SNRs.append(snr)

X = np.array(X)
y = np.array(y)
SNRs = np.array(SNRs)

if DEBUG:
    print("X shape:", X.shape) # X shape: (220000, 128, 2)
    print("X min:", X.min())
    print("X max:", X.max())
    print("Labels shape:", y.shape) # Labels shape: (220000,)

#Scale data to be between -1/1 (its already between -1/1 but is small, so we scale it up
max_val = np.max(np.abs(X))  # symmetric scaling
X = X / max_val

# Filter data set by SNR > threshold
HIGH_SNR_THRESHOLD = 16
mask = SNRs > HIGH_SNR_THRESHOLD
X = X[mask]
y = y[mask]
SNRs = SNRs[mask]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
le_snr = LabelEncoder()
snr_encoded = le_snr.fit_transform(SNRs)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded,
#                                                     test_size=0.4,
#                                                     random_state=42,
#                                                     stratify=y)

X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(X, y_encoded, SNRs,
                                                                         test_size=0.4,
                                                                         random_state=40,
                                                                         stratify=y
)

# One-hot encode
# use this with loss='categorical_crossentropy'
#y_train_cat = to_categorical(y_train)
#y_test_cat = to_categorical(y_test)
# else use sparse_categorical_crossentropy
y_train_cat = y_train
y_test_cat = y_test

# X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
# X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

# Filter training set by SNR > threshold
# HIGH_SNR_THRESHOLD = 0
# train_mask = snr_train > HIGH_SNR_THRESHOLD
#
# X_train_high = X_train[train_mask]
# y_train_cat_high = y_train_cat[train_mask]
# snr_train_high = snr_train[train_mask]

num_classes = len(le.classes_)
num_snrs = len(le_snr.classes_)
input_shape = (X_train.shape[1:]) # (128, 2)

if DEBUG:
    print("X_train shape:", X_train.shape)  # X_train shape: (132000, 128, 2)
    print("X_train min:", X_train.min())
    print("X_train max:", X_train.max())
    print("X_test min:", X_test.min())
    print("X_test max:", X_test.max())
    print("y_train_cat shape:", y_train_cat.shape) # y_train_cat shape: (132000, )
    # print("X_train_high shape:", X_train_high.shape)  # X_train_high shape: (59473, 128, 2), with 0dB+
    # print("y_train_cat_high shape:", y_train_cat_high.shape)  # y_train_cat_high shape: (59473,) with 0dB+
    print("Number of classes:", num_classes) #Number of classes: 11
    print("Number of SNRs:", num_snrs)  # Number of SNRs: 1
    print("Input Shape:", input_shape) # Input Shape: (128, 2)

#note gridsize must be square
#plot_sample_training_data(X_train_high, y_train_cat_high, snr_train_high , 16)
plot_sample_training_data(X_train, y_train_cat, snr_train , 16)




# Lowest performing classes:
# Class Index: 7, Label: QAM16, True Positives: 900
# Class Index: 10, Label: WBFM, True Positives: 1702
# Class Index: 9, Label: QPSK, True Positives: 3062
# Class Index: 8, Label: QAM64, True Positives: 3473
# Class Index: 0, Label: 8PSK, True Positives: 3679

#Generate more data with GAN
if USE_GAN:
    print("using GAN")
    # size of the latent space
    latent_dim = 100
    gan_epochs = 500
    gan_batch_size = 64

    # create the discriminator
    discriminator = define_discriminator(input_shape, num_classes)
    # summarize the model
    discriminator.summary()
    # plot the model
    plot_model(discriminator, to_file='RMLGAN2_discriminator_plot.png', show_shapes=True, show_layer_names=True)

    # create the generator
    generator = define_generator(latent_dim, num_classes)
    # summarize the model
    generator.summary()
    # plot the model
    plot_model(generator, to_file='RMLGAN2_generator_plot.png', show_shapes=True, show_layer_names=True)

    # create the gan
    gan_model = define_gan(generator, discriminator)
    # load image data
    # dataset = load_real_samples()
    dataset = [X_train, y_train_cat]
    # # train model
    train(generator, discriminator, gan_model, dataset, latent_dim, gan_epochs, gan_batch_size, num_classes)

########################


    # X_train_aug = np.concatenate([X_train, generated_samples], axis=0)
    # y_train_aug = np.concatenate([y_train, generated_labels], axis=0)
    # y_train_cat_aug = to_categorical(y_train_aug, num_classes=num_classes)

#END Generate more data with GAN

exit(0)


### THE CODE BELOW WAS FOR TESTING ONLY AND SHOULD NOT BE USED ###

learning_rate = 0.001

model = define_model(num_classes, input_shape, learning_rate)
model.summary()

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

trainStart_time = time.time()
print("Training Start", trainStart_time)

# check if we are an interactive session
if hasattr(sys, 'ps1'):
    verbose_setting = 0
else:
    verbose_setting = 1

if LOAD_EXISTING_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model from {MODEL_SAVE_PATH}")
    model = load_model(MODEL_SAVE_PATH)
    history = None
else:
    # Train
    if USE_GAN:
        if DEBUG:
            print("X_train_aug shape:", X_train_aug.shape)  #
            print("y_train_cat_aug shape:", y_train_cat_aug.shape)  #
        history = model.fit(
            X_train_aug,
            y_train_cat_aug,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            verbose=verbose_setting,
            callbacks=[early_stopping]
        )
    else:
        history = model.fit(
            X_train,
            y_train_cat,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            verbose=verbose_setting,
            callbacks=[early_stopping]
        )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

trainDone_time = time.time()
print("Training Done", trainDone_time)
print("Predicting")

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Group by SNR and compute accuracy
df_eval = pd.DataFrame({
    'true': y_test,
    'pred': y_pred,
    'snr': snr_test
})

snr_acc = df_eval.groupby('snr').apply(lambda x: accuracy_score(x['true'], x['pred']))

plt.figure(figsize=(8, 4))
plt.plot(snr_acc.index, snr_acc.values, marker='o')
plt.title("Accuracy vs. SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
if SAVE_PLOTS_FLAG:
    plt.savefig(ACCSNR_PLOT)
else:
    plt.show()


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
true_positives = np.diag(cm)
worst_classes = np.argsort(true_positives)[:5]  # indices of lowest TP
print("Lowest performing classes:")
for idx in worst_classes:
    print(f"Class Index: {idx}, Label: {le.classes_[idx]}, True Positives: {true_positives[idx]}")

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
if SAVE_PLOTS_FLAG:
    plt.savefig(CONFMATRIX_PLOT)
else:
    plt.show()

if history is not None:
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if SAVE_PLOTS_FLAG:
        plt.savefig(ACC_PLOT)
    else:
        plt.show()

    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if SAVE_PLOTS_FLAG:
        plt.savefig(LOSS_PLOT)
    else:
        plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


############################### END CODE #################################
end_time = time.time()
print("Done!", end_time)

print(f"Total Training Time: {(trainDone_time - trainStart_time):.2f} seconds")
print(f"Total Predicting Time: {(end_time - trainDone_time):.2f} seconds")
print(f"Total Time: {(end_time - start_time):.2f} seconds")
