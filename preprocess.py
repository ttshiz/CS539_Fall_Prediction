import numpy as np
import sklearn as sk

#one dimension
def zero_crossings(slice):
    return np.where(np.diff(np.signbit(slice)))[0].size

#one dimension
def avg_slope(slice):
    return np.mean(np.diff(slice))

# for each dimension still need to set
def process_slice(slice):
    return [np.amin(slice["acc_x"]), np.amin(slice["acc_y"]), np.amin(slice["acc_z"])
            , np.amax(slice["acc_x"]), np.amax(slice["acc_y"]), np.amax(slice["acc_z"])
            , np.std(slice["acc_x"]), np.std(slice["acc_y"]), np.std(slice["acc_z"])
            , np.mean(slice["acc_x"]), np.mean(slice["acc_y"]), np.mean(slice["acc_z"])
            , np.mean(np.diff(slice["acc_x"])), np.mean(np.diff(slice["acc_y"]))
            , np.mean(np.diff(slice["acc_z"]))
            , zero_crossings(slice["acc_x"]), zero_crossings(slice["acc_y"])
            , zero_crossings(slice["acc_z"])
            , avg_slope(slice["acc_x"]), avg_slope(slice["acc_y"])
            , avg_slope(slice["acc_z"])
            , slice["label"][0]]
 
def process_file(file_name, slice_size):
    try:
        data = np.genfromtxt(file_name, dtype=None, delimiter=',', names=True)
        feature_list = []
        start_time = data[0]["timestamp"]
        end_index = data.size - 1

        snum = 1
        slice_start = 0
        slice_start_label = data[0]["label"]
        slice_end = 0
        slice_end_label = slice_start_label
        while(slice_end < end_index):
            # if we still need to find the index of the end of the slice
            if (data[slice_end]["timestamp"] < start_time + snum*slice_size) and (
                    slice_start_label == slice_end_label):
                slice_end = slice_end + 1
                slice_end_label = data["label"][slice_end]
            # otherwise consider slice complete and process
            else:
                feature_list.append(process_slice(data[slice_start:slice_end]))
                snum = snum + 1
                slice_start = slice_end
                slice_start_label = slice_end_label
        return feature_list
    except:
        print("Error Processing "+file_name)
        return False
