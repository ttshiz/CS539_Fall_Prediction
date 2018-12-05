import load_data
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# Loads data from file and returns generator for cross-validation
def setup(filename, num_slices, k=10):
    split_dataset = load_data.split_data_kfold(filename, num_slices, k)
    cross_val_generator = load_data.validation_cases(split_dataset)
    return cross_val_generator


# Calculates metrics based on predicted and actual labels using global true positive,
# false negative, and false positive counts where applicable
def calc_metrics(Y_truth, Y_predict):
    #acc = accuracy_score(Y_truth, Y_predict)
    #f1 = f1_score(Y_truth, Y_predict, average='micro')
    #prec = precision_score(Y_truth, Y_predict, average='micro')
    #rec = recall_score(Y_truth, Y_predict, average='micro')
    #return acc, f1, prec, rec
    return None

def matrix_n_report(Y_truth, Y_predict, filename=None):
    #Calculated dictionary of classification metrics mapping label: set of metrics
    cr = classification_report(Y_truth, Y_predict, output_dict=True)
    cm = confusion_matrix(Y_truth, Y_predict)
    return cr, cm
    

# Trains HMM on X_train set, predicts labels on X_test set, dumps model to file,
# and returns predicted labels
def run_one_HMM(X_train, X_test, prepro_param, num_states=3, num_iter=10, covar_type='diag'
                , rand_seed=None, run_count=1):
    model = hmm.GaussianHMM(n_components=num_states, covariance_type=covar_type
                            , random_state=rand_seed, n_iter=num_iter)
    model.fit(X_train)
    Y_predict = model.predict(X_test)
    filename = "hmm_pre_" + prepro_param + "states_" + str(num_states) + "iter_"
    filename = filename + str(num_iter) + "run_" + str(run_count) + ".pkl"
    joblib.dump(model, filename)
    return Y_predict

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
        Y_predicted = run_one_HMM(X_train, X_test, prepro_param, run_num)
        #a, f, p, r = calc_metrics(Y_test, Y_predicted)
        #accuracies.extend(0, a)
        #f1s.extend(0, f)
        #precisions.extend(0, p)
        #recalls.extend(0, r)
        run_num = run_num +1
        break # comment out for full run, only here for testing
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
    filename = "preprocessed_1.0E+09.json"
    num_slices = 1
    k = 10
    s = do_cross_validation(filename, num_slices, k)
    #mean_acc, mean_f1, mean_prec, mean_rec, max_acc, max_f1, max_prec, max_rec = calc_cross_val_aggregates(s)
    #stout = "\t mean \t max \n" + "\t" + mean_acc + "\t" + max_acc + "\n"
    #stout = stout + "\t" + mean_prec + "\t" + max_prec + "\n"
    #stout = stout +"\t" + mean_rec+ "\t"+ max_rec + "\n"
    #stout = stout + "\t" + mean_f1 + "\t" + max_f1 +"\n"
    #print(stout)
    return s
