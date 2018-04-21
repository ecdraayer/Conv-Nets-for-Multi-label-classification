import datetime
import numpy
import pandas
import os
import cv2
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


start_time = datetime.datetime.now()

# path to the data files
path = os.getcwd()
training_data = pandas.read_csv(path + "/Train/train_v2.csv")
testing_data = pandas.read_csv(path + '/sample_submission_v2.csv')

training_images = []
training_tags = []
testing_images = []

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

label_count = {key: 0 for (key, value) in thresholds.items()}

# obtaining all the unique tags used
tags = set()
for v in training_data['tags'].values:
    for tag in v.split(' '):
        label_count[tag] += 1
        tags.add(tag)

tags = list(tags)
tags_count = len(tags)



total_threshold = {key: 0 for (key, value) in thresholds.items()}
tag_to_index = {l: i for i, l in enumerate(tags)}
index_to_tag = {i: l for i, l in enumerate(tags)}

# reading the training data
for i, c in training_data.values:
    training_images += [cv2.resize(cv2.imread('Train/train-jpg/' + i + '.jpg'), (32, 32))]
    classes = numpy.zeros(tags_count)
    for t in c.split(' '):
        classes[tag_to_index[t]] = 1
    training_tags += [classes]

# reading the testing data
for i, c in testing_data.values:
    if i.startswith('test'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg/' + i + '.jpg'), (32, 32))]
    elif i.startswith('file'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg-additional/' + i + '.jpg'), (32, 32))]


# for filename in os.listdir(os.getcwd() + '/Test/test-jpg/'):
#     if filename.endswith('.jpg'):
#         testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg/' + filename), (32, 32))]
#
# for filename in os.listdir(os.getcwd() + '/Test/test-jpg-additional/'):
#     if filename.endswith('.jpg'):
#         testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg-additional/' + filename), (32, 32))]
# testing_images = [cv2.resize(cv2.imread('Test/test-jpg/' + i + '.jpg'), (32, 32)) for i, c in testing_data.values]

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
yfull_train =[]

kf = KFold(n_splits=k, shuffle=True, random_state=1)

for train_index, test_index in kf.split(training_tags):
    Training_Images = training_images[train_index]
    Training_Tags = training_tags[train_index]
    X_valid = training_images[test_index]
    Y_valid = training_tags[test_index]
    num_fold += 1
    kfold_weights_path = os.path.join(path +'/models/'+'weights_' + str(num_fold) + '.h5')
    baseline_model = Sequential()
    baseline_model.add(BatchNormalization(input_shape=(32, 32, 3)))
    baseline_model.add(Conv2D(8, (1, 1), activation='relu'))
    baseline_model.add(Conv2D(16, (2, 2), activation='relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
    baseline_model.add(Conv2D(32, (3, 3), activation='relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
    baseline_model.add(Dropout(0.25))
    baseline_model.add(Conv2D(64, (3, 3), activation='relu'))
    baseline_model.add(MaxPooling2D(pool_size=(2, 2)))
    baseline_model.add(Dropout(0.25))
    baseline_model.add(Flatten())
    baseline_model.add(Dense(256, activation='relu'))
    baseline_model.add(Dropout(0.5))
    baseline_model.add(Dense(tags_count, activation='sigmoid'))
    baseline_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]
    baseline_model.fit(x = Training_Images, y= Training_Tags, validation_data=(X_valid, Y_valid),
              batch_size=128, verbose=2, nb_epoch=10, callbacks=callbacks,
              shuffle=True)
    if os.path.isfile(kfold_weights_path):
        baseline_model.load_weights(kfold_weights_path)
    p_valid = baseline_model.predict(X_valid, batch_size = 128, verbose=2)
    print(fbeta_score(Y_valid, numpy.array(p_valid) > 0.2, beta=2, average='samples'))
    print("Optimizing prediction threshold")
    print(get_optimal_thresholds(Y_valid, p_valid))
    p_test = baseline_model.predict(training_images, batch_size = 128, verbose=2)
    yfull_train += [p_test]
    p_test = baseline_model.predict(testing_images, batch_size = 128, verbose=2)
    yfull_test += [p_test]


for key, value in total_threshold.items():
    total_threshold[key] /= k

result = numpy.array(yfull_test[0])
for i in range(1, k):
    result += numpy.array(yfull_test[i])

result /= k
result = pandas.DataFrame(result, columns=tags)


result_on_training_images = numpy.array(yfull_train[0])
for i in range(1, k):
    result_on_training_images += numpy.array(yfull_train[i])

result_on_training_images /= k
result_on_training_images = pandas.DataFrame(result_on_training_images, columns=tags)

thresholds = total_threshold

preds = []
preds_on_training_images = []

for i in range(result.shape[0]):
    a = result.ix[[i]]
    for index, col_name in enumerate(a):
        a[col_name] = numpy.where(a[col_name] > thresholds[col_name], True, False)
    a = a.transpose()
    a = a.loc[a[i] == True]
    preds.append(' '.join(list(a.index)))

testing_data['tags'] = preds
testing_data.to_csv('submission_keras.csv', index=False)


for i in range(result_on_training_images.shape[0]):
    a = result_on_training_images.ix[[i]]
    for index, col_name in enumerate(a):
        a[col_name] = numpy.where(a[col_name] > thresholds[col_name], True, False)
    a = a.transpose()
    a = a.loc[a[i] == True]
    preds_on_training_images.append(' '.join(list(a.index)))

training_data['tags'] = preds_on_training_images
training_data.to_csv('predictions_on_training_images.csv', index=False)

validation_data = pandas.read_csv(path + '/Train/train_v2.csv')

label_count_true_positive = {key: 0 for (key, value) in thresholds.items()}
label_count_true_negative = {key: 0 for (key, value) in thresholds.items()}
label_count_false_positive = {key: 0 for (key, value) in thresholds.items()}
label_count_false_negative = {key: 0 for (key, value) in thresholds.items()}

with open('predictions_on_training_images.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    index = 0
    first = True
    for row in readCSV:
        if first:
            first = False
            continue
        t = row[1].split(' ')
        true_tags = validation_data['tags'].values[index].split(' ')
        for label in tags:
            if label not in true_tags and label not in t:
                label_count_true_negative[label] += 1
            if label in true_tags and label in t:
                label_count_true_positive[label] += 1
            if label not in true_tags and label in t:
                label_count_false_positive[label] += 1
            if label in true_tags and label not in t:
                label_count_false_negative[label] += 1
        index += 1

print ("Accuracies: ")
for key, value in thresholds.items():
    print(key, (label_count_true_positive[key] + label_count_true_negative[key]) / len(preds_on_training_images))

print ("Precision: ")
for key, value in thresholds.items():
    if label_count_true_positive[key] != 0:
        print(key, label_count_true_positive[key] / (label_count_true_positive[key] + label_count_false_positive[key]))
    else:
        print(key, 0.0)

print ("Recall: ")
for key, value in thresholds.items():
    if label_count_true_positive[key] != 0:
        print(key, label_count_true_positive[key] / (label_count_true_positive[key] + label_count_false_negative[key]))
    else:
        print(key, 0.0)

print("Optimized threshold values: ", total_threshold)

end_time = datetime.datetime.now()
print("Time taken to execute: ")
print(end_time - start_time)
