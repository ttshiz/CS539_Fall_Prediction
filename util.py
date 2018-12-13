import load_data
import numpy as np
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
    acc = accuracy_score(Y_truth, Y_predict)
    f1 = f1_score(Y_truth, Y_predict, average='macro')
    prec = precision_score(Y_truth, Y_predict, average='macro')
    rec = recall_score(Y_truth, Y_predict, average='macro')
    f1_w = f1_score(Y_truth, Y_predict, average='weighted')
    prec_w = precision_score(Y_truth, Y_predict, average='weighted')
    rec_w = recall_score(Y_truth, Y_predict, average='micro')
    f1_micro = f1_score(Y_truth, Y_predict, average='micro')
    prec_micro = precision_score(Y_truth, Y_predict, average='micro')
    rec_micro = recall_score(Y_truth, Y_predict, average='micro')
    #met_dict = classification_report(Y_truth, Y_predict, output_dict=True)
    #met_dict["accuracy"] = acc
    #return met_dict
    return acc, f1, prec, rec, f1_w, prec_w, rec_w, f1_micro, prec_micro, rec_micro
  
# Calculates classification report and confusion matrix given predicted, and true labels  
def matrix_n_report(Y_truth, Y_predict, filename=None):
    #Calculated dictionary of classification metrics mapping label: set of metrics
    cr = classification_report(Y_truth, Y_predict, output_dict=True)
    cm = confusion_matrix(Y_truth, Y_predict)
    return cr, cm

# Calculates mean and max results across the multiple runs of cross validation
def calc_cross_val_aggregates(scores):
    score_dict = dict()
    score_dict["mean_acc"] = np.mean(scores["accuracies"])
    score_dict["mean_f1"] = np.mean(scores["f1s"])
    score_dict["mean_prec"] = np.mean(scores["precisions"])
    score_dict["mean_rec"] = np.mean(scores["recalls"])
    score_dict["mean_f1_w"] = np.mean(scores["f1s_weighted"])
    score_dict["mean_prec_w"] = np.mean(scores["precisions_weighted"])
    score_dict["mean_rec_w"] = np.mean(scores["recalls_weighted"])
    score_dict["mean_f1_micro"] = np.mean(scores["f1s_micro"])
    score_dict["mean_prec_micro"] = np.mean(scores["precisions_micro"])
    score_dict["mean_rec_micro"] = np.mean(scores["recalls_micro"])

    score_dict["max_acc"] = np.max(scores["accuracies"])
    score_dict["max_f1"] = np.max(scores["f1s"])
    score_dict["max_prec"] = np.max(scores["precisions"])
    score_dict["max_rec"] = np.max(scores["recalls"])
    score_dict["max_f1_w"] = np.max(scores["f1s_weighted"])
    score_dict["max_prec_w"] = np.max(scores["precisions_weighted"])
    score_dict["max_rec_w"] = np.max(scores["recalls_weighted"])
    score_dict["max_f1_micro"] = np.max(scores["f1s_micro"])
    score_dict["max_prec_micro"] = np.max(scores["precisions_micro"])
    score_dict["max_rec_micro"] = np.max(scores["recalls_micro"])
    return score_dict
