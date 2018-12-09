import load_data
import math
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
    f1 = f1_score(Y_truth, Y_predict, average='micro')
    prec = precision_score(Y_truth, Y_predict, average='micro')
    rec = recall_score(Y_truth, Y_predict, average='micro')
    return acc, f1, prec, rec
    #return None

def matrix_n_report(Y_truth, Y_predict, filename=None):
    #Calculated dictionary of classification metrics mapping label: set of metrics
    cr = classification_report(Y_truth, Y_predict, output_dict=True)
    cm = confusion_matrix(Y_truth, Y_predict)
    return cr, cm


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
