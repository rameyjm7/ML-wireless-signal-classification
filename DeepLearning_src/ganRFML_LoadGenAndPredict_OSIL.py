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
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import itertools
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, KLDivergence



DEBUG = True
USE_GAN_GENERATED_SAMPLES = False
GAN_MODEL_TO_LOAD = 'RMLGAN2_model_9270.keras'
LOAD_EXISTING_MODEL = False

MODEL_SAVE_PATH = "RNNOSIL_rnn_model.keras"
DIRECTORY = "../ML-wireless-signal-classification"
FILE_NAME = "RML2016.10a_dict.pkl"
SAVE_PLOTS_FLAG = 1
ACCGAN_PLOT = "RNNOSIL_accuracy_gan_plot.png"
ACCGAN_TEMP_PLOT = "RNNOSIL_accuracy_gan_plot.png"
ACCSNR_PLOT = "RNNOSIL_accuracy_snr_plot.png"
CONFMATRIX_PREOSIL_PLOT = "RNNOSIL_preOSIL_confusion_matrix.png"
CONFMATRIX_PLOT = "RNNOSIL_confusion_matrix.png"
ACC_PLOT = "RNNOSIL_accuracy_plot.png"
LOSS_PLOT = "RNNOSIL_loss_plot.png"
RMLGAN_SAMPLE_TRIAN_IMAGES = 'RNNOSIL_real_samples_IQ_signals.png'
RMLGAN_SAMPLEGEN_TRIAN_IMAGES = 'RNNOSIL_generated_samples_IQ_signals.png'
MODEL_PLOT = 'RNNOSIL_model_plot.png'
OSIL_DISTANCE_PLOT = 'RNNOSIL_distance_plot.png'

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
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def define_model2(num_classes, input_shape, learning_rate=0.001):
    # LR schedule unchanged
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=15000, decay_rate=0.98, staircase=True
    )

    inputs = Input(shape=input_shape, name="iq_input")     # (128,2)

    # ————————— I branch —————————
    # OLD: Conv1D(50,5) → reshape → Conv2D((1,5))
    # NEW: single Conv2D with kernel=(1,8), filters=50
    i_chan = Lambda(lambda x: x[..., 0:1], name="slice_I")(inputs)  # (batch, 128, 1)
    i_chan = Reshape((1, input_shape[0], 1), name="reshape_I")(i_chan)  # (batch, 1, 128, 1)
    x0 = Conv2D(50, (1, 8), padding="same", activation="relu",
               name="conv2d_I")(i_chan)                              # ← UPDATED

    # ————————— Q branch —————————
    # OLD: Conv1D(50,5) → reshape → Conv2D((1,5))
    # NEW: single Conv2D with kernel=(1,8), filters=50
    q_chan = Lambda(lambda x: x[..., 1:2], name="slice_Q")(inputs)  # (batch, 128, 1)
    q_chan = Reshape((1, input_shape[0], 1), name="reshape_Q")(q_chan)  # (batch, 1, 128, 1)
    x1 = Conv2D(50, (1, 8), padding="same", activation="relu",
               name="conv2d_Q")(q_chan)                              # ← UPDATED

    # ————————— IQ branch —————————
    # (2,1) → (1,3) → (1,3) with 24→24→50 filters
    iq = Reshape((input_shape[1], input_shape[0], 1), name="reshape_full")(inputs)
    iq = Conv2D(24, (2, 1), activation="relu", name="conv2d_IQ_1")(iq)
    iq = Conv2D(24, (1, 3), padding="same", activation="relu",
                name="conv2d_IQ_2")(iq)
    iq = Conv2D(50, (1, 3), padding="same", activation="relu",
                name="conv2d_IQ_3")(iq)                              # ← UPDATED

    # ——— merge all three paths ———
    merged = Concatenate(axis=1, name="concat_IQ")([iq, x0, x1])      # (3,128,50)
    c = Conv2D(100, (3, 5), activation="relu", name="conv2d_combined")(merged)

    # ——— squeeze “height” dim to make (time, features) ———
    seq = Lambda(lambda x: K.squeeze(x, axis=1), name="squeeze")(c)   # (124,100)

    # ——— LSTM → BiLSTM → classifier ———
    l1 = LSTM(128, return_sequences=True, name="lstm1")(seq)
    l2 = Bidirectional(LSTM(128), name="bilstm")(l1)
    d1 = Dense(128, activation="relu", name="fc")(l2)
    d1 = Dropout(0.5, name="dropout")(d1)
    d1 = GaussianDropout(0.2, name="gauss_dropout")(d1)
    outputs = Dense(num_classes, activation="softmax", name="softmax")(d1)

    model = Model(inputs, outputs, name="IQGMCL_exact")
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

## Gram matrix branch
def gram_matrix_fn(x):
    # Compute Gram matrix: x shape (batch_size, time_steps, channels)
    x_t = tf.transpose(x, perm=[0, 2, 1])  # (batch_size, channels, time_steps)
    gram = tf.matmul(x_t, x_t, transpose_b=True)  # (batch_size, channels, channels)
    return gram

def define_model3(myClasses, myInputShape, myLearningRate=0.001):
    if DEBUG:
        print("Classes:", myClasses)
        print("Input Shape:", myInputShape)
        print("Learning Rate:", myLearningRate)

    initial_lr = myLearningRate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=15000,
        decay_rate=0.98,
        staircase=True
    )

    ## Main Input: IQ sequence
    main_input = Input(shape=myInputShape, name='iq_input')  # (128, 2)

    # LSTM branch
    x = LSTM(128, return_sequences=True)(main_input)
    x = Dropout(0.5)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)

    gram = Lambda(gram_matrix_fn, name='gram_matrix')(main_input)
    gram = Flatten()(gram)
    gram = Dense(1024, activation='relu')(gram)
    gram = Dropout(0.1)(gram)

    ## Merge the two branches
    merged = Concatenate()([x, gram])

    # Final output
    output = Dense(myClasses, activation="softmax")(merged)

    # Create the model
    model = Model(inputs=main_input, outputs=output)

    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

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
    if DEBUG:
        print('z_input shape', z_input.shape)
    # generate labels
    labels = np.random.permutation(np.tile(np.arange(n_classes), n_samples // n_classes))
    #labels = np.random.randint(0, n_classes, n_samples)

    # Print occurrences of each label
    label_counts = np.bincount(labels)
    for i, count in enumerate(label_counts):
        print(f"Label {i}: {count}")
    return [z_input, labels]

def plot_sample_training_data(xtrain, ytrain, snrtrain, grid_size, filename):

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
        plt.savefig(filename)
        print('Plot saved as', filename)
    else:
        plt.show()

def softmax_with_temperature(logits, temperature):
    """Applies temperature scaling to logits"""
    logits = tf.convert_to_tensor(logits)
    return tf.nn.softmax(logits / temperature)

def kd_loss(y_packed, y_pred_logits):
    num = num_classes
    y_true = y_packed[:, :num]
    teacher_soft = tf.nn.softmax(y_packed[:, num:] / temperature)
    student_soft = tf.nn.softmax(y_pred_logits / temperature)

    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred_logits, from_logits=False)
    kl = tf.keras.losses.KLDivergence()(teacher_soft, student_soft) * (temperature**2)

    return alpha * kl + (1 - alpha) * ce

def softmax_with_temperature(logits, temperature):
    logits = tf.convert_to_tensor(logits)
    return tf.nn.softmax(logits / temperature)

def distillation_loss(y_true, y_pred, teacher_pred,
                      temperature=2.0, alpha=0.5):
    """
    y_true:      one-hot hard labels
    y_pred:      student logits or probs
    teacher_pred:teacher logits or probs
    """
    y_true = tf.cast(y_true, tf.float32)
    teacher_soft = softmax_with_temperature(teacher_pred, temperature)
    student_soft = softmax_with_temperature(y_pred, temperature)

    ce = CategoricalCrossentropy(from_logits=False)
    kl = KLDivergence()

    hard_loss = ce(y_true, y_pred)
    kd_loss   = kl(teacher_soft, student_soft) * (temperature**2)

    return alpha * kd_loss + (1.0 - alpha) * hard_loss

# custom kd_loss that closes over T, alph
def kd_loss_tmp(y_packed, y_pred):
    y_true   = y_packed[:, :num_classes]
    teach_lp = y_packed[:, num_classes:]
    return distillation_loss(y_true, y_pred,
                             teach_lp,
                             temperature=T,
                             alpha=alph)

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
    print("SNRs shape:", SNRs.shape)  #

#Scale data to be between -1/1 (its already between -1/1 but is small, so we scale it up
max_val = np.max(np.abs(X))  # symmetric scaling
X = X / max_val

# Filter data set by SNR > threshold
# HIGH_SNR_THRESHOLD = 16
# mask = SNRs > HIGH_SNR_THRESHOLD
# X = X[mask]
# y = y[mask]
# SNRs = SNRs[mask]

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
if USE_GAN_GENERATED_SAMPLES:
    # else use sparse_categorical_crossentropy
    y_train_cat = y_train
    y_test_cat = y_test
else:
    # use this with loss='categorical_crossentropy'
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)


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
plot_sample_training_data(X_train, y_train, snr_train , 25, RMLGAN_SAMPLE_TRIAN_IMAGES)




# Lowest performing classes:
# Class Index: 7, Label: QAM16, True Positives: 900
# Class Index: 10, Label: WBFM, True Positives: 1702
# Class Index: 9, Label: QPSK, True Positives: 3062
# Class Index: 8, Label: QAM64, True Positives: 3473
# Class Index: 0, Label: 8PSK, True Positives: 3679
# Class Index: 2, Label: AM-SSB

###### OSIL
# Define your initial classes, start with maybe ~50% of the initial classes
#initial_classes = [0, 1, 3, 4, 5, 6, 9]  # these are the integer labels from LabelEncoder
initial_classes = [2,7,8,10]  # these are the integer labels from LabelEncoder


# Mask for initial classes
train_mask = np.isin(y_train, initial_classes)

# Select only initial class samples
X_train_initial = X_train[train_mask]
y_train_initial = y_train[train_mask]
snr_initial = snr_train[train_mask]


if DEBUG:
    print("OSIL STATS>>>>>>>>")
    print("X_train_initial shape:", X_train_initial.shape)  # X_train_initial shape: (84000, 128, 2)
    print("X_train_initial min:", X_train_initial.min()) # X_train_initial min: -0.9205978
    print("X_train_initial max:", X_train_initial.max()) # X_train_initial max: 0.695032
    print("X_test min:", X_test.min()) # X_test min: -0.9434717
    print("X_test max:", X_test.max()) # X_test max: 1.0
    print("y_train_initial shape:", y_train_initial.shape) # y_train_initial shape: (84000,)

###### OSIL END

#Generate more data with GAN
if USE_GAN_GENERATED_SAMPLES:
    print("using GAN generated samples")
    # load model
    model = load_model(GAN_MODEL_TO_LOAD)
    latent_dim = 100
    n_examples = 11 * 90  # must be multiple of 11
    n_class = 11
    #Since we don't specific which SNR to generate, we make a dummy array assuming they are all 18dB
    generated_snr = np.full((220000,), 18)
    # generate images
    latent_points, generated_labels = generate_latent_points(latent_dim, n_examples, n_class)
    # generate images
    generated_samples = model.predict([latent_points, generated_labels])
    # scale back to org dataset range
    generated_samples = generated_samples * max_val
    # # plot the result
    # save_plot(X, n_examples, n_class)

    plot_sample_training_data(generated_samples, generated_labels, generated_snr, 25, RMLGAN_SAMPLEGEN_TRIAN_IMAGES)

    ########################

    X_train_aug = np.concatenate([X_train, generated_samples], axis=0)
    y_train_aug = np.concatenate([y_train, generated_labels], axis=0)
    y_train_cat_aug = to_categorical(y_train_aug, num_classes=num_classes)

    if DEBUG:
        print("X_train_aug shape:", X_train_aug.shape)  #
        print("X_train_aug min:", X_train_aug.min())
        print("X_train_aug max:", X_train_aug.max())
        # print("X_test min:", X_test.min())
        # print("X_test max:", X_test.max())
        print("y_train_cat_aug shape:", y_train_cat_aug.shape)  #
        # # print("X_train_high shape:", X_train_high.shape)  # X_train_high shape: (59473, 128, 2), with 0dB+
        # # print("y_train_cat_high shape:", y_train_cat_high.shape)  # y_train_cat_high shape: (59473,) with 0dB+
        # print("Number of classes:", num_classes)  # Number of classes: 11
        # print("Number of SNRs:", num_snrs)  # Number of SNRs: 1
        # print("Input Shape:", input_shape)  # Input Shape: (128, 2)

#END Generate more data with GAN

learning_rate = 0.0001
input_shape = X_train_initial.shape[1:]  # (128,2)
print("Initial (OSIL) Input Shape:", input_shape)  # Initial (OSIL) Input Shape: (128, 2)
num_initial_classes = len(initial_classes)

if DEBUG:
    print("OSIL STATS>>>>>>>>")
    print("Number of initial classes:", num_initial_classes) # Number of initial classes: 7
    print("Initial Input Shape:", input_shape) # Initial Input Shape: (128, 2)

model = define_model(num_classes, input_shape, learning_rate)

#model = define_model(num_classes, input_shape, learning_rate)
model.summary()
plot_model(model, to_file=MODEL_PLOT, show_shapes=True, show_layer_names=True)

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
    if USE_GAN_GENERATED_SAMPLES:
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
            #X_train, ##### REMOVED for OSIL
            X_train_initial,
            y_train_initial,
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
#####OSIL

print("\nClassification Report PRE OSIL:")
# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
if SAVE_PLOTS_FLAG:
    plt.savefig(CONFMATRIX_PREOSIL_PLOT)
    print('\nConfusion Matrix PRE OSIL saved to', CONFMATRIX_PREOSIL_PLOT)
else:
    plt.show()


# Clone the trained model as teacher (freeze it)
teacher_model = tf.keras.models.clone_model(model)
teacher_model.set_weights(model.get_weights())
teacher_model.trainable = False

print("\nStarting OSIL with Knowledge Distillation...")

remaining_classes = sorted(set(range(num_classes)) - set(initial_classes))
print ('Remaining Classes:', remaining_classes ) # Remaining Classes: [7, 8, 9, 10]
temperature = 5.0
alpha = 0.7

#####TUNING
# 1) build a small tuning split from initial + first remaining class
first_cls = remaining_classes[0]
mask_first = np.isin(y_train, initial_classes + [first_cls])
X_base = np.concatenate([X_train_initial, X_train[y_train==first_cls]], axis=0)
y_base = np.concatenate([y_train_initial, y_train[y_train==first_cls]], axis=0)
y_base_cat = to_categorical(y_base, num_classes=num_classes)
teacher_base = teacher_model.predict(X_base)

# 2) hold out 20% for tuning
X_tr, X_tune, y_tr_cat, y_tune_cat, teach_tr, teach_tune = train_test_split(
    X_base, y_base_cat, teacher_base,
    test_size=0.2, random_state=42, stratify=y_base
)

# 3) pack once so we don’t repeat
y_tr_packed   = np.concatenate([y_tr_cat,  teach_tr],   axis=1)
y_tune_packed = np.concatenate([y_tune_cat, teach_tune], axis=1)

# 4) grid‐search ranges
temps  = [1.0, 2.0, 5.0, 7.0, 10.0]
alphas = [0.1, 0.3, 0.5, 0.7]
best = {"acc": 0.0, "T":None, "alph":None}

for T, alph in itertools.product(temps, alphas):
    print(f"Testing T={T}, alph={alph},")
    # a small fresh copy of your student
    student = define_model(num_classes, input_shape, learning_rate)
    student.set_weights(model.get_weights())  # start from the same initial‐only checkpoint

    student.compile(
        optimizer=Adam(learning_rate),
        loss=kd_loss_tmp,
        metrics=['accuracy']
    )
    # 5) train just 3 epochs on the small train split
    student.fit(X_tr, y_tr_packed,
                epochs=3, batch_size=64, verbose=0)
    # 6) eval on tuning split
    acc = student.evaluate(X_tune, y_tune_packed, verbose=0)[1]
    if acc > best["acc"]:
        best.update(acc=acc, T=T, alph=alph)

print(f"Best on tuning set → T={best['T']}, alph={best['alph']}, val_acc={best['acc']:.4f}")
temperature, alpha = best['T'], best['alph']

#####TUNING

osil_acc = []          # training accuracy over time
osil_val_acc = []      # validation accuracy
osil_loss = []         # training loss
osil_val_loss = []     # validation loss
osil_labels = []       # label for each OSIL step (e.g., class name)

for cls in remaining_classes:
    print(f"\n--- Incrementally adding class {cls}: {le.classes_[cls]} ---")

    # New class data
    new_class_mask = (y_train == cls)
    X_new = X_train[new_class_mask]
    y_new = y_train[new_class_mask]

    # Combine with previous data (optional: sample a portion to speed up)
    X_combined = np.concatenate([X_train_initial, X_new], axis=0)
    y_combined = np.concatenate([y_train_initial, y_new], axis=0)

    # One-hot for distillation
    y_combined_cat = to_categorical(y_combined, num_classes=num_classes)

    if DEBUG:
        print("OSIL STATS>>>>>>>>")
        print("X_combined Shape:", X_combined.shape)  #
        print("y_combined Shape:", y_combined.shape)  #
        print("y_combined_cat Shape:", y_combined_cat.shape)  #

    # Teacher predictions
    teacher_preds = teacher_model.predict(X_combined)

    y_packed = np.concatenate([y_combined_cat, teacher_preds], axis=1)

    X_train_ds, X_val_ds, y_train_p, y_val_p = train_test_split(
        X_combined, y_packed,
        test_size=0.1,
        random_state=42,
        stratify=y_combined
    )

    # 4) compile with your new kd_loss
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=kd_loss,
        metrics=['accuracy']
    )

    # 5) fit directly on numpy arrays
    history_osil = model.fit(
        X_train_ds, y_train_p,
        validation_data=(X_val_ds, y_val_p),
        epochs=100,
        batch_size=64,
        verbose=verbose_setting,
        callbacks=[early_stopping]
    )

    # Save last epoch's accuracy/loss from this phase
    osil_acc.append(history_osil.history['accuracy'][-1])
    osil_val_acc.append(history_osil.history['val_accuracy'][-1])
    osil_loss.append(history_osil.history['loss'][-1])
    osil_val_loss.append(history_osil.history['val_loss'][-1])
    osil_labels.append(le.classes_[cls])  # e.g., "QAM64"

    # Update teacher model to latest student model after each step
    teacher_model.set_weights(model.get_weights())

    # Expand training data tracker
    X_train_initial = X_combined
    y_train_initial = y_combined

    # Save model checkpoint
    model.save(MODEL_SAVE_PATH)
    print(f"Model updated and saved after class {cls}")

# recompile model to get rid of the custom loss function
# and use sparse_categorical_crossentropy for transportability
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    # loss='sparse_categorical_crossentropy',
    loss='categorical_crossentropy', # <-- so I can use it in my ganRFML_LoadGenerateAndPredict.py script as is
    metrics=['accuracy'])

model.save(MODEL_SAVE_PATH)
print(f"Final model (w/ built‐in loss) saved to {MODEL_SAVE_PATH}")

plt.figure(figsize=(10, 5))
plt.plot(osil_labels, osil_acc, label="Train Accuracy", marker='o')
plt.plot(osil_labels, osil_val_acc, label="Val Accuracy", marker='o')
plt.title("OSIL Accuracy Over Time")
plt.xlabel("New Class Added")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
if SAVE_PLOTS_FLAG:
    plt.savefig("RNNOSIL_osil_accuracy_plot.png")
else:
    plt.show()

plt.figure(figsize=(10, 5))
plt.plot(osil_labels, osil_loss, label="Train Loss", marker='o')
plt.plot(osil_labels, osil_val_loss, label="Val Loss", marker='o')
plt.title("OSIL Loss Over Time")
plt.xlabel("New Class Added")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
if SAVE_PLOTS_FLAG:
    plt.savefig("RNNOSIL_osil_loss_plot.png")
else:
    plt.show()
###### OSIL END


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
print("Accuracy by SNR:")
for snr, acc in snr_acc.items():
    print(f"  {snr} dB : {acc:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(snr_acc.index, snr_acc.values, marker='o')
plt.title("Accuracy vs. SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
# Set y-axis limits
plt.ylim(0, 1)

# Major grid lines every 0.1
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.grid(which='major', linestyle='-', linewidth=0.75)

# Minor grid lines every 0.01 (dotted)
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
plt.grid(which='minor', linestyle=':', linewidth=0.5)
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
