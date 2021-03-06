import operator
import load_data
import util
import numpy as np
#import pandas as pd
import math
from itertools import chain
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

setup = util.setup
calc_metrics = util.calc_metrics
matrix_n_report = util.matrix_n_report
calc_cross_val_aggregates = util.calc_cross_val_aggregates

# Trains HMM on X_train set, predicts labels on X_test set, dumps model to file,
# and returns predicted labels
def run_hmmlG(X_train, Y_train, X_test, Y_test, prepro_param, num_states=3
              , num_iter=10000000, covar_type='diag', rand_seed=None, run_count=1):
    labels = set(Y_train+Y_test)
    model = dict()
    for l in labels:
        # train an HMM for each class
        # and for each datum in X_test
        model[l] = hmm.GaussianHMM(n_components=num_states, covariance_type=covar_type
                                  , random_state=rand_seed, n_iter=num_iter)
        X_l = list()
        for (x, y) in zip(X_train, Y_train):
            if y == l:
                X_l.insert(0, x)
        model[l].fit(np.array([[s] for s in chain.from_iterable(X_l)]), [len(x) for x in X_l])
    ## calculate the probability in each of the models
    ## then set the predicted label to the class of the HMM with the highest probability
    Y_predict = list()
    #X_test_np = np.array([[s] for s in chain.from_iterable(X_test)])
    #X_test_lens = [len(x) for x in X_test]
    for t in X_test:
        br = -100000
        lbl = None
        for l in labels:
            r = model[l].score(np.array([[v] for v in t]), [len(t)])
            #print(l, r)
            if r > br:
                br = r
                lbl = l
        Y_predict.append(lbl)
    Y_res = Y_predict
    #Y_res = list()
    #for Y in pd.DataFrame(Y_predict).to_dict('record'):
    #    print(Y)
    #    Y_res.insert(0, max(Y.items(), key=operator.itemgetter(1))[0])
    #Y_res.reverse()
    filename = "hmm_pre_" + prepro_param + "states_" + str(num_states) + "iter_"
    filename = filename + str(num_iter) + "run_" + str(run_count) + ".pkl"
    joblib.dump(model, filename)
    print(Y_res)
    return Y_res

# Runs k-fold Cross Validation on preprocessed data found in filename
# with num_slices the number of preprocessed slices to include in each sample
def do_cross_validation(filename, num_slices, k):
    data_gen = setup(filename, num_slices, k)
    scores = dict()
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    run_num = 1
    for ((X_train, Y_train), (X_test, Y_test)) in data_gen:
        prepro_param = filename[13:-5]
        Y_predicted = run_one_HMM(X_train, Y_train, X_test, Y_train, prepro_param, run_num)
        a, f, p, r = calc_metrics(Y_test, Y_predicted)
        accuracies.append(a)
        f1s.append(f)
        precisions.append(p)
        recalls.append(r)
        run_num = run_num +1
        break # comment out for full run, only here for testing
    scores["accuracies"] = accuracies
    scores["f1s"] = f1s
    scores["precisions"] = precisions
    scores["recalls"] = recalls
    return scores

run_one_HMM = run_hmmlG

def main():
    max_slice_size_file =  "preprocessed_6.0E+09.json"
    files = ["preprocessed_5.0E+09.json", "preprocessed_2.5E+09.json"
             ,  "preprocessed_1.0E+09.json", "preprocessed_5.0E+08.json"
             ,  "preprocessed_2.5E+08.json", "preprocessed_5.0E+07.json"
             ,  "preprocessed_2.5E+07.json",  "preprocessed_5.0E+06.json"]
    num_slices = dict()
    # time_slice(nanoseconds) * number_of_slices <= 6 seconds
    num_slices["preprocessed_6.0E+09.json"] = [1]
    num_slices["preprocessed_5.0E+09.json"] = [1]
    num_slices["preprocessed_2.5E+09.json"] = [1, 2]
    num_slices["preprocessed_1.0E+09.json"] = [1, 2, 4]
    num_slices["preprocessed_5.0E+08.json"] = [1, 2, 4, 8]
    num_slices["preprocessed_2.5E+08.json"] = [1, 2, 4, 8, 16]
    num_slices["preprocessed_5.0E+07.json"] = [1, 2, 4, 8, 16, 32]
    num_slices["preprocessed_2.5E+07.json"] = [1, 2, 4, 8, 16, 32, 64]
    num_slices["preprocessed_5.0E+06.json"] = [1, 2, 4, 8, 16, 32, 64, 128]
    
    single_filename = "preprocessed_1.0E+09.json"

    k = 10
    
    s = do_cross_validation(single_filename, num_slices[single_filename][0], k)
    # want calc_cross_valaggregates for each run associated with run parameters
    #mean_acc, mean_f1, mean_prec, mean_rec, max_acc, max_f1, max_prec, max_rec = calc_cross_val_aggregates(s)
    #stout = "\t mean \t max \n" + "\t" + mean_acc + "\t" + max_acc + "\n"
    #stout = stout + "\t" + mean_prec + "\t" + max_prec + "\n"
    #stout = stout +"\t" + mean_rec+ "\t"+ max_rec + "\n"
    #stout = stout + "\t" + mean_f1 + "\t" + max_f1 +"\n"
    #print(stout)
    # want confusion matrix for best pair of file and num_slices
    # time provided runs:
    #num_slices_reach = [256, 512, 1024]
    #reach_files = ["preprocessed_5.0E+06.json"]
    return s

if __name__ == "__main__":
    main()
