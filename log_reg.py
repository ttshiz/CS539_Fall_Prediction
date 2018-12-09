import load_data
import run_hmm
import numpy as np
import math
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
# Trains logistic regression on X_train and Y_train sets, predicts labels on X_test
# and Y_test sets, dumps model to file and returns predicted labels
def run_one_LR(X_train, Y_train, X_test, Y_test, prepro_param, rand_seed=None
               , solver='lbfgs', max_iter=1000, multi_class='multinomial'
               , verbose=1, n_jobs=1, run_count=1):
    model = LogisticRegression(random_state=rand_seed, solver=solver, max_iter=max_iter
                               , multi_class=multi_class, n_jobs=n_jobs)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    print("Writing file....")
    
    filename = "lr_pre_" + prepro_param + "solver_" + str(solver) + "iter_"
    filename = filename + str(max_iter) + "run_" + str(run_count) + ".pkl"
    joblib.dump(model, filename)
    
    print(Y_predict)
    return Y_predict

# Runs k-fold Cross Validation on preprocessed data found in filename
# with num_slices the number of preprocessed slices to include in each sample
def do_cross_validation(filename, num_slices, k):
    data_gen = run_hmm.setup(filename, num_slices, k)
    scores = dict()
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    run_num = 1
    for ((X_train, Y_train), (X_test, Y_test)) in data_gen:
        prepro_param = filename[13:-5]
        Y_predicted = run_one_LR(X_train, Y_train, X_test, Y_train, prepro_param, run_num)
        a, f, p, r = run_hmm.calc_metrics(Y_test, Y_predicted)
        accuracies.append(a)
        f1s.append(f)
        precisions.append(p)
        recalls.append(r)
        run_num = run_num +1
        #break # comment out for full run, only here for testing
    scores["accuracies"] = accuracies
    scores["f1s"] = f1s
    scores["precisions"] = precisions
    scores["recalls"] = recalls
    return scores

def calc_cross_val_aggregates(scores):
    mean_acc = np.mean(scores["accuracies"])
    mean_f1 = np.mean(scores["f1s"])
    mean_prec = np.mean(scores["precisions"])
    mean_rec = np.mean(scores["recalls"])

    max_acc = np.max(scores["accuracies"])
    max_f1 = np.max(scores["f1s"])
    max_prec = np.max(scores["precisions"])
    max_rec = np.max(scores["recalls"])
    return mean_acc, mean_f1, mean_prec, mean_rec, max_acc, max_f1, max_prec, max_rec

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
    print(s)
    return s

if __name__ == "__main__":
    main()
