import datetime
import numpy
import pandas
import os
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras import optimizers

start_time = datetime.datetime.now()

RESOLUTION = 139

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
for i, c in training_data.values:
    training_images += [cv2.resize(cv2.imread('Train/train-jpg/' + i + '.jpg'), (RESOLUTION, RESOLUTION))]
    classes = numpy.zeros(tags_count)
    for t in c.split(' '):
        classes[tag_to_index[t]] = 1
    training_tags += [classes]

# reading the testing data
for i, c in testing_data.values:
    if i.startswith('test'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg/' + i + '.jpg'), (RESOLUTION, RESOLUTION))]
    elif i.startswith('file'):
        testing_images += [cv2.resize(cv2.imread(os.getcwd() + '/Test/test-jpg-additional/' + i + '.jpg'), (RESOLUTION, RESOLUTION))]


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

base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(RESOLUTION, RESOLUTION, 3))

print('Model loaded.')

for layer in base_model.layers:
    layer.trainable = False

for train_index, test_index in kf.split(training_tags):
    Training_Images = training_images[train_index]
    Training_Tags = training_tags[train_index]
    X_valid = training_images[test_index]
    Y_valid = training_tags[test_index]
    num_fold += 1
    # print('Start KFold number {} from {}'.format(num_fold, k))
    # print('Split train: ', len(training_images), len(training_tags))
    # print('Split valid: ', len(X_valid), len(Y_valid))
    kfold_weights_path = os.path.join(path +'/models/'+'weights_kfold_' + str(num_fold) + '.h5')
    
    x = base_model.output
    fine_tune = Flatten(input_shape=base_model.output_shape[1:])(x)
    fine_tune = Dense(128, activation='relu')(fine_tune)
    fine_tune = BatchNormalization()(fine_tune)
    out = Dense(17, activation='sigmoid')(fine_tune)
    baseline_model = Model(inputs=base_model.input, outputs=out)
    '''
    baseline_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    '''
    baseline_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
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

print ("Optimized threshold values: ", total_threshold)

end_time = datetime.datetime.now()
print ("Time taken to execute: ")
print (end_time - start_time)
