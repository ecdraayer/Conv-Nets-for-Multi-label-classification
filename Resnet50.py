import datetime
import numpy
import pandas
from sklearn.metrics import fbeta_score
import numpy as np # linear algebra
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
import cv2
from sklearn.model_selection import KFold

start_time = datetime.datetime.now()

# path to the data files
path = os.getcwd()
training_data = pandas.read_csv(path + "/Train/train_v2.csv")
testing_data = pandas.read_csv(path + '/sample_submission_v2.csv')

training_images = []
training_tags = []
testing_images = []

# obtaining all the unique tags used
tags = set()
for v in training_data['tags'].values:
    for tag in v.split(' '):
        tags.add(tag)

tags = list(tags)
tags_count = len(tags)

# default values of threshold
thresholds = {'blow_down': 0.2,
              'bare_ground': 0.2,
              'conventional_mine': 0.2,
              'blooming': 0.2,
              'cultivation': 0.2,
              'artisinal_mine': 0.2,
              'haze': 0.2,
              'primary': 0.2,
              'slash_burn': 0.2,
              'habitation': 0.2,
              'clear': 0.2,
              'road': 0.2,
              'selective_logging': 0.2,
              'partly_cloudy': 0.2,
              'agriculture': 0.2,
              'water': 0.2,
              'cloudy': 0.2
              }

total_threshold = {key: 0 for (key, value) in thresholds.items()}
tag_to_index = {l: i for i, l in enumerate(tags)}
index_to_tag = {i: l for i, l in enumerate(tags)}

# reading the training data
for i, c in training_data.values[:18000]:
    training_images += [cv2.resize(cv2.imread('Train/train-jpg/' + i + '.jpg'), (197, 197))]
    classes = numpy.zeros(tags_count)
    for t in c.split(' '):
        classes[tag_to_index[t]] = 1
    training_tags += [classes]

# reading the testing data
for i, c in testing_data.values:
    if i.startswith('test'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg/' + i + '.jpg'), (197, 197))]
    elif i.startswith('file'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg-additional/' + i + '.jpg'), (197, 197))]

# normalizing values and converting lists to numpy arrays
training_images = numpy.array(training_images) / 255
training_tags = numpy.array(training_tags, numpy.uint8)
testing_images = numpy.array(testing_images) / 255


# method to find the optimized thresholds for each tags
def get_optimal_thresholds(ground_truth, prediction_values):
    # initial values of each tag thresholds set to 0.2
    local_thresholds = [0.2] * tags_count

    # for each tags, find what threshold gives best f-score
    for i in range(tags_count):
        index = 0
        max_score = 0
        for j in range(100):
            j /= 100
            local_thresholds[i] = j

            y_pred = numpy.zeros_like(prediction_values)
            for l in range(tags_count):
                y_pred[:, l] = (prediction_values[:, l] > local_thresholds[l]).astype(numpy.int)
            score = fbeta_score(ground_truth, y_pred, beta=2, average='samples')

            if score > max_score:
                index = j
                max_score = score
        local_thresholds[i] = index
        print(index_to_tag[i], index, max_score)

    for i in range(len(tags)):
        thresholds[tags[i]] = local_thresholds[tag_to_index[tags[i]]]
        total_threshold[tags[i]] += thresholds[tags[i]]
    return thresholds


k = 3
num_fold = 0
sum_score = 0

yfull_test = []
yfull_train = []

kf = KFold(n_splits=k, shuffle=True, random_state=1)

for train_index, test_index in kf.split(training_tags):

    Training_Images = training_images[train_index]
    Training_Tags = training_tags[train_index]
    X_valid = training_images[test_index]
    Y_valid = training_tags[test_index]
    num_fold += 1

    kfold_weights_path = os.path.join(path + '/models/' + 'weights_kfold_' + str(num_fold) + '.h5')

    # build the ResNet network
    base_model = ResNet50(include_top=False, input_shape=(197, 197, 3))
    print('Model loaded.')

    for layer in base_model.layers:
        layer.trainable = False

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

    top_model.add(Dropout(0.3))
    top_model.add(Dense(17, activation='sigmoid'))
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

    model.fit(x=Training_Images, y=Training_Tags, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=2, epochs=10, callbacks=callbacks,
              shuffle=True)

    if os.path.isfile(kfold_weights_path):
        model.load_weights(kfold_weights_path)

    p_valid = model.predict(X_valid, batch_size=128, verbose=2)
    print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
    print("Optimizing prediction threshold")
    print(get_optimal_thresholds(Y_valid, p_valid))

    p_test = model.predict(training_images, batch_size=128, verbose=2)
    yfull_train.append(p_test)

    p_test = model.predict(testing_images, batch_size=128, verbose=2)
    yfull_test.append(p_test)

result = numpy.array(yfull_test[0])
for i in range(1, k):
    result += numpy.array(yfull_test[i])

for key, value in total_threshold.items():
    total_threshold[key] /= k
result /= k
result = pandas.DataFrame(result, columns=tags)

thresholds = total_threshold

preds = []
for i in range(result.shape[0]):
    a = result.ix[[i]]
    for index, col_name in enumerate(a):
        a[col_name] = numpy.where(a[col_name] > thresholds[col_name], True, False)
    a = a.transpose()
    a = a.loc[a[i] == True]
    preds.append(' '.join(list(a.index)))

testing_data['tags'] = preds
testing_data.to_csv('submission_keras.csv', index=False)

print("Optimized threshold values: ", total_threshold)

end_time = datetime.datetime.now()
print("Time taken to execute: ")
print(end_time - start_time)
