'''Tekne Consulting blogpost --- teknecons.com'''


from einops import reduce
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import mixed_precision


''' GPU config'''


config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)
mixed_precision.set_global_policy('mixed_float16')

print(tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))

base_savename = 'model150x90_'


'''initialize and prepare data'''


def data_init(tensor_p: 'path', labels_p: 'path'):
    tensor = np.load(tensor_p).astype('float32') / 255
    labels = np.load(labels_p)
    # half the size
    tensor = reduce(tensor, 'b (h h2) (w w2) c -> b h w c', 'max', h2=2, w2=2)
    return (tensor, labels)


def data_prep(tensor: 'np.array', labels: 'np.array', n_splits: 'int'):
    trte_split = train_test_split(tensor, labels, test_size=0.3,
                                  random_state=1234, stratify=labels)
    skf = StratifiedKFold(n_splits=n_splits, random_state=4321, shuffle=True)
    train_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                                   fill_mode='nearest')
    test_gen = ImageDataGenerator()
    return(trte_split, skf, train_gen, test_gen)


'''simple sequential model'''


def model_init():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 90, 3),
                            padding='same'))  # padding='same' because some bees are on the edge
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_regularizer='l2'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def model_name(fold_no):
    return base_savename + str(fold_no) + '.h5'


def model_exec(splitted_data, kfold, train_generator, test_generator):
    pic_train, _, labels_train, _ = splitted_data
    histories = []
    fold_no = 1
    for tri, tei in kfold.split(pic_train, labels_train):
        pic_tr, pic_val = pic_train[tri], pic_train[tei]
        labels_tr, labels_val = labels_train[tri], labels_train[tei]
        train_datagen = train_generator.flow(pic_tr, labels_tr, batch_size=20)
        val_datagen = test_generator.flow(pic_val, labels_val, batch_size=20)

        model = init_model()
        checkpoint = keras.callbacks.ModelCheckpoint('saved_models/' + model_name(fold_no),
                                                     monitor='val_accuracy', verbose=2,
                                                     save_best_only=True, mode='max')

        history = model.fit(train_datagen, epochs=80, batch_size=10, validation_data=val_datagen,
                            verbose=0, callbacks=[checkpoint])
        histories.append(history)

        model.load_weights("saved_models/" + base_savename + str(fold_no) + ".h5")
        labels_pred = [1 if p > 0.5 else 0 for p in model.predict(pic_test)]
        accuracy = balanced_accuracy_score(labels_test, labels_pred)
        f1 = f1_score(labels_test, labels_pred)
        print(f'fold no: {fold_no}, accuracy: {accuracy}, f1: {f1}')

        tf.keras.backend.clear_session()
        fold_no += 1
        return histories


'''basic model inspection'''


def hisory_plot(histories):
    avg_loss = np.mean([h.history['loss'] for h in histories], axis=0)
    avg_valloss = np.mean([h.history['val_loss'] for h in histories], axis=0)
    avg_acc = np.mean([h.history['accuracy'] for h in histories], axis=0)
    avg_valacc = np.mean([h.history['val_accuracy'] for h in histories], axis=0)

    fig, ax = plt.subplots()
    ax.plot(avg_loss)
    ax.plot(avg_valloss)
    ax.legend(['train_loss', 'validation_loss'], loc='upper left')
    return fig


def model_eval(model_name: 'path', splitted_data):
    _, pic_test, _, _ = splitted_data
    model = models.load_model(model_name)
    soft_predict = model.predict(pic_test)

    thresh_eff = {tr / 10: [1 if p > tr / 10 else 0 for p in soft_predict] for tr in range(2, 10)}
    thresh_eval_dict = {}

    for key, val in thresh_eff.items():
        accuracy = np.round(balanced_accuracy_score(labels_test, val), 2)
        f1 = np.round(f1_score(labels_test, val), 2)
        precision = np.round(precision_score(labels_test, val), 2)
        recall = np.round(recall_score(labels_test, val), 2)
        thresh_eval_dict[key] = {'accuracy': accuracy, 'f1': f1, 'precision': precision,
                                 'reacall': recall}
    return thresh_eval_dict


if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_path = os.path.join(this_dir, 'bees_tensor.npy')
    labels_path = os.path.join(this_dir, 'labels.npy')

    tensor, labels = data_init(tensor_path, labels_path)
    splitted_data, kfold, train_generator, test_generator = data_prep(tensor, labels, n_splits=3)
    training_history = model_exec(splitted_data, kfold, train_generator, test_generator)

    # may change every execution
    the_best_model = os.path.join(this_dir, 'saved_models/model150x90_3.h5')

    plot = history_plot(training_history)
    plt.show()

    model_evaluation = model_eval(the_best_model)

    # prints evaluation of test dataset with various thresholds
    for key, val in model_evaluation.items():
        print(f'For threshold={key}:')
        for k, v in val.items():
            print(f'{k}={v}')
