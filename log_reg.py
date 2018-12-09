import load_data
import util
import json
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

setup = util.setup
calc_metrics = util.calc_metrics
matrix_n_report = util.matrix_n_report
calc_cross_val_aggregates = util.calc_cross_val_aggregates

# Trains logistic regression on X_train and Y_train sets, predicts labels on X_test
# and Y_test sets, dumps model to file and returns predicted labels
def run_one_LR(X_train, Y_train, X_test, Y_test, prepro_param, rand_seed=None
               , solver='lbfgs', max_iter=1000, multi_class='multinomial'
               , verbose=1, n_jobs=16, run_count=1):
    model = LogisticRegression(random_state=rand_seed, solver=solver, max_iter=max_iter
                               , multi_class=multi_class, n_jobs=n_jobs)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    print("Running cross val "+str(run_count)+"....")    
    #filename = "lr_pre_" + prepro_param + "solver_" + str(solver) + "iter_"
    #filename = filename + str(max_iter) + "run_" + str(run_count) + ".pkl"
    #joblib.dump(model, filename)
    #print(Y_predict)
    return Y_predict

# Runs k-fold Cross Validation on preprocessed data found in filename
# with num_slices the number of preprocessed slices to include in each sample
def do_cross_validation(filename, num_slices, k):
    print("Cross Validating: "+filename)
    data_gen = setup(filename, num_slices, k)
    scores = dict()
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    f1s_weighted = []
    precisions_weighted = []
    recalls_weighted = []
    f1s_micro = []
    precisions_micro = []
    recalls_micro = []
    run_num = 1
    for ((X_train, Y_train), (X_test, Y_test)) in data_gen:
        prepro_param = filename[13:-5]
        Y_predicted = run_one_LR(X_train, Y_train, X_test, Y_train, prepro_param
                                 , run_count=run_num)
        a, f, p, r, f_w, p_w, r_w, f_m, p_m, r_m = calc_metrics(Y_test, Y_predicted)
        accuracies.append(a)
        f1s.append(f)
        precisions.append(p)
        recalls.append(r)
        f1s_weighted.append(f_w)
        precisions_weighted.append(p_w)
        recalls_weighted.append(r_w)
        f1s_micro.append(f_m)
        precisions_micro.append(p_m)
        recalls_micro.append(r_m)
        run_num = run_num +1
        #print("accuracy: "+str(a)+" f1score: "+str(f)+" precision: "+str(p)+" recall: "
        #      +str(r)+" f1score weighted: "+str(f_w)+" precision weighted: "+str(p_w)
        #      +" recall weighted: "+str(r_w)+" f1score micro: "+str(f_m)
        #      +" precision micro: "+str(p_m)+" recall micro: "+str(r_m))
        #break # comment out for full run, only here for testing
    scores["accuracies"] = accuracies
    scores["f1s"] = f1s
    scores["precisions"] = precisions
    scores["recalls"] = recalls
    scores["f1s_weighted"] = f1s_weighted
    scores["precisions_weighted"] = precisions_weighted
    scores["recalls_weighted"] = recalls_weighted
    scores["f1s_micro"] = f1s_micro
    scores["precisions_micro"] = precisions_micro
    scores["recalls_micro"] = recalls_micro
    return scores

def print_by_cross_val_run(file_dict):
    stout = "\t mean \t max \n"+"acc \t"+str(file_dict["mean_acc"])+"\t"
    stout = stout + str(file_dict["max_acc"]) +"\n"
                                
    stout = stout + "prec \t" + str(file_dict["mean_prec"]) + "\t"
    stout = stout + str(file_dict["max_prec"]) + "\n"
                                
    stout = stout +"rec \t" + str(file_dict["mean_rec"]) + "\t"
    stout = stout + str(file_dict["max_rec"]) + "\n"
            
    stout = stout + "f1 \t" + str(file_dict["mean_f1"]) + "\t"
    stout = stout + str(file_dict["max_f1"]) +"\n"
            
    stout = stout + "prec_weighted \t" + str(file_dict["mean_prec_w"]) + "\t"
    stout = stout + str(file_dict["max_prec_w"]) + "\n"
            
    stout = stout +"rec_weighted \t" + str(file_dict["mean_rec_w"]) + "\t"
    stout = stout + str(file_dict["max_rec_w"]) + "\n"
            
    stout = stout + "f1_weighted \t" + str(file_dict["mean_f1_w"]) + "\t"
    stout = stout + str(file_dict["max_f1_w"]) +"\n"
            
    stout = stout + "prec_micro \t" + str(file_dict["mean_prec_micro"]) + "\t"
    stout = stout + str(file_dict["max_prec_micro"]) + "\n"
            
    stout = stout +"rec_micro \t" + str(file_dict["mean_rec_micro"]) + "\t"
    stout = stout + str(file_dict["max_rec_micro"]) + "\n"
            
    stout = stout + "f1_micro \t" + str(file_dict["mean_f1_micro"]) + "\t"
    stout = stout + str(file_dict["max_f1_micro"]) +"\n"

    print(stout)
            
def main():
    max_slice_size_file =  "preprocessed_6.0E+09.json"
    files = ["preprocessed_5.0E+09.json", "preprocessed_2.5E+09.json"
             ,  "preprocessed_1.0E+09.json"#, "preprocessed_5.0E+08.json"
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

    #all_file_dict = dict()
    try:
        with open("results_summary.json", "r") as infile:
            all_file_dict = json.loads(infile.read())
    except:
        all_file_dict = dict()
    for filename in files:
        if filename in all_file_dict:
            continue
        file_dict = dict()
        try:
            for num_s in num_slices[filename]:
                s = do_cross_validation(filename, num_s, k)
                #print("scores "+str(s)) #enable for very verbose
                with open("metrics_slice_num_"+str(num_s)+"_"+filename, "w") as fp:
                    json.dump(s, fp)
                fp.close()
                print("aggregates:")
                # want calc_cross_valaggregates for each run associated with run parameters
                file_dict = calc_cross_val_aggregates(s)
                #print_by_cross_val_run(file_dict) #enable for verbose
                #break; # comment out for full run
                all_file_dict[filename] = file_dict
        except ValueError:
            print("Error on file"+filename)
            continue
            #break; # comment out for full run
        with open("results_summary.json", "w") as fp:
            json.dump(all_file_dict, fp)
          
    # want confusion matrix for best pair of file and num_slices
    # time provided runs:
    #num_slices_reach = [256, 512, 1024]
    #reach_files = ["preprocessed_5.0E+06.json"]
    #print(s)
    return s

if __name__ == "__main__":
    main()
