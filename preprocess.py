import numpy as np
import sklearn as sk

from collections import Counter

FALL_LABELS = set([b'FOL', b'FKL', b'BSC', b'SDL'])

#one dimension
def zero_crossings(slice):
    return np.where(np.diff(np.signbit(slice)))[0].size

#Wafaa's metric
def min_max_distance(slice):
    return np.sqrt(np.square(np.amax(slice) - np.amin(slice))
                   + (np.square(np.argmax(slice) - np.argmin(slice))))
    
# for each dimension still need to set
def process_slice(slice):
    slice_features = dict()
    slice_features["x_min"] = np.amin(slice["acc_x"])
    slice_features["y_min"] = np.amin(slice["acc_y"])
    slice_features["z_min"] = np.amin(slice["acc_z"])
    
    slice_features["x_max"] = np.amax(slice["acc_x"])
    slice_features["y_max"] = np.amax(slice["acc_y"])
    slice_features["z_max"] = np.amax(slice["acc_z"])
    
    slice_features["x_std"] = np.std(slice["acc_x"])
    slice_features["y_std"] = np.std(slice["acc_y"])
    slice_features["z_std"] = np.std(slice["acc_z"])
    
    slice_features["x_mean"] = np.mean(slice["acc_x"])
    slice_features["y_mean"] = np.mean(slice["acc_y"])
    slice_features["z_mean"] = np.mean(slice["acc_z"])

    slice_features["x_slope"] = np.mean(np.diff(slice["acc_x"]))
    slice_features["y_slope"] = np.mean(np.diff(slice["acc_y"]))
    slice_features["z_slope"] = np.mean(np.diff(slice["acc_z"]))

    slice_features["x_zc"] = zero_crossings(slice["acc_x"])
    slice_features["y_zc"] = zero_crossings(slice["acc_y"])
    slice_features["z_zc"] = zero_crossings(slice["acc_z"])

    slice_features["x_mmd"] = min_max_distance(slice["acc_x"])
    slice_features["y_mmd"] = min_max_distance(slice["acc_y"])
    slice_features["z_mmd"] = min_max_distance(slice["acc_z"])

    # label each timeslice with the label of the majority class unless it is a fall
    falls = FALL_LABELS.intersection(set(slice["label"]))
    slice_features["label"] = falls.pop() if falls else Counter(
        slice["label"]).most_common(1)[0][0]

    return slice_features
 
def process_file(file_name, slice_size):
    try:
        data = np.genfromtxt(file_name, dtype=None, delimiter=',', names=True)
        feature_list = []
        start_time = data[0]["timestamp"]
        end_index = data.size - 1

        snum = 1
        slice_start = 0
        slice_end = 0
        while(slice_end < end_index):
            # if we still need to find the index of the end of the slice
            if (data[slice_end]["timestamp"] < start_time + snum*slice_size):
                slice_end = slice_end + 1
            # otherwise consider slice complete and process
            else:
                feature_list.append(process_slice(data[slice_start:slice_end]))
                snum = snum + 1
                slice_start = slice_end
        return feature_list
    except:
        print("Error Processing "+file_name)
        return False
