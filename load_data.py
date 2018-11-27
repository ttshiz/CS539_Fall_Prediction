import json
import random
from collections import Counter

FALL_LABELS = set([b'FOL', b'FKL', b'BSC', b'SDL'])

KEY_LIST = ["x_min", "y_min", "z_min", "x_max", "y_max", "z_max", "x_std", "y_std","z_std"
            , "x_mean", "y_mean", "z_mean", "x_slope", "y_slope", "z_slope", "x_zc"
            , "y_zc", "z_zc", "x_mmd", "y_mmd", "z_mmd"]                         

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i +n]

def window_label_multi(labels):
    intersect = set(labels).intersection(FALL_LABELS)
    if intersect:
        return intersect.pop()
    else:
        return Counter(labels).most_common(1)[0][0]

def window_X(window):
    wlist = []
    for w in window:
        wlist.extend([w[k] for k in KEY_LIST])
    return wlist

def convert(datum, num_slices):
    recording_X = []
    recording_Y = []
    for i in range(0, len(datum)-(num_slices-1)):
        window = datum[i:i +num_slices]
        recording_Y.insert(0, window_label_multi([w["label"] for w in window]))
        recording_X.insert(0, window_X(window))
    return (recording_X, recording_Y)

def convert_list(recordings, num_slices):
    recordings_X = []
    recordings_Y = []
    for recording in recordings:
        recording_X, recording_Y = convert(recording, num_slices)
        recordings_X.extend(recording_X)
        recordings_Y.extend(recording_Y)
    return (recordings_X, recordings_Y)

#returns k datasets split into X and Y
def split_data_kfold(filename, num_slices, k):
    with open(filename) as fp:
        dataset = json.load(fp)
    del dataset["slice_size"]
    datalist = list(dataset.values())
    random.shuffle(datalist)
    datasets = list()
    chunksize = len(datalist)//k
    for i in range(0, k):
        start = i*chunksize
        end = (start + chunksize) if i < (k-1) else len(datalist)
        datasets.insert(0, convert_list(datalist[start:end], num_slices))
    return datasets

# this function is an iterator that takes the output of split_data_kfold
# and outputs train and test sets as tuples
def validation_cases(dataset):
    for j in range(len(dataset)):
        X_test, Y_test = dataset[j]
        rest =  dataset[:j]+dataset[(j+1):]
        X_train = list()
        Y_train = list()
        for r_x, r_y in rest:
            X_train.extend(r_x)
            Y_train.extend(r_y)
        yield ((X_train, Y_train), (X_test, Y_test))

            
