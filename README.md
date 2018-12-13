# CS539 
# Efficient Early Fall Detection Technique

## Introduction and Rationale
Falls play an important role in loss of self-governance, fatalities and  injuries among the elderly and remarkably affect the costs of national health systems. It is estimated from the the World Health Organization (WHO) that  28– 42%  of elderly fall each year. Accordingly, there has been  a dramatic increment in the developing assortment of research on fall detection system over the most recent years. However, the vast majority of the previous work in fall recognition required specific equipment and programming which is costly to keep up. Moreover, there are still restrictions on their accuracy and flexibility, avoiding routine utilization of this procedure in clinical practice. Mauldin et al. have reviewed 57 projects that used wearable devices to detect falls in elderly. Yet, solely 7.1% of these projects revealed testing their models in a real-world setting[1]. Furthermore,  while a few examinations that utilized sensors and wearable  devices  connected to observed subjects have demonstrated the ability to accomplish higher identification accuracy; those devices are not not well-accepted by elderly  due to their obtrusiveness and restricted portability. Finally, very few of the studies have even attempted to try to predict the falls. Predicting falls successfully could mean lives saved even before damage.
Considering aforementioned challenges, we aim to design, develop, and perform preliminary testing of a novel and efficient technique that can be implemented in an embedded hardware device such as smartphone to identify and possibly even predict falls before they happen. The work will be focus on the use of fast and efficient techniques based on new statistical features and classification will be performed using the most commonly used machine-learning based classifiers in literature.

## Dataset Selection and Description
Since there is not much research into fall prediction or extremely early detection using smartphones or strategies that would be extensible to the sensors found in smartphones such as wearable Fall Detection Systems (FDS), the authors expanded the dataset search criteria to include datasets used or collected for fall detection studies.  Furthermore, since smartphone data or a similar enough set of sensors was desired, the subset of fall detection studies or datasets that was most appropriate was ones where they tested FDS systems or collected the requisite data for such testing.  The FARSEEING real-world fall repository was considered because it has real, not simulated, fall accelerometer measurements.  The drawbacks of this dataset is that only 20 fall events are available upon request and further collaboration for full access is likely outside the scope of a term project. If access to the 20 fall events is made available, it may be used as testing data at a future date.

### Dataset
We used the second realease of the MobiAct dataset, collected by T.E.I. of Crete to develop our techniques and perform our study.  This dataset includes 4 different types of falls and 12 different Activities of Daily Life/Living (ADLs), performed by a total of 66 subjects with more than 3200 trials, all captured by smartphone[2].

#### To obtain the dataset:
The dataset we used in this project can be found at:
https://bmi.teicrete.gr/en/the-mobifall-and-mobiact-datasets-2/

#### Example Data Instance
An example reading from the time series data from one data collection trial from the MobiAct dataset is as follows: 
1913880219000, 0.90021986, -9.557653, -1.4939818
1914086499000, 0.7565677, -9.5385, -1.13964
1914287283000, 1.0151415, -9.490616, -1.292869
The first entry in each tuple is the time stamp in nanoseconds, followed by the acceleration force along the x, y, and z axes respectively in meters per second squared.
## Algorithm Selection
### Preprocessing
Our goals for preprocessing were two fold.  First to only use light/easy to compute methods to make the algorithm easily portable to a smartphone and not drain batteries too quickly.  Second to maintain the high accuracies we found in existing research.  To do this we decided to slice the data in to uniform lengths of time and compute simple features like the mean of the slice for each of these slices.  We would then feed a fixed number of these slices to the machine learning algorithms as a learning data instance. To explore the possible combinations of slice length (in nanoseconds) and the number of slices, we employed an exponential grid search.  We started the search from the larger slice lengths that we knew can work well from previous research, and worked our way down to smaller lengths, that we did not find in our literature search.
### Main Algorithm
#### Selection
There were several factors considered in algorithm selection. Given the sizes of the available datasets overfitting is a major concern.  Furthermore, since there are various smartphones on the market and for the longevity of the usefulness of the tool, the techniques used will need to be tolerant to this variability.  Some of the top candidates for the main algorithm were Decision Trees (DTs), Recurrent Neural Networks (RNNs), Support Vector Machines (SVMs), and Hidden Markov Models (HMMs).  RNNs are considered due to their ability to perform reasonably on temporal data approach. Similarly since, some of the previous work on fall detection has found some success with SVMs and HMMs they may also be a good candidate for prediction.  Finally we considered Decision Trees since they were a nice easily computable model and we wanted to determine even they could perform well.  All of these techniques had the potential for better results than basic statistical thresholding, since they may reduce false positives.
#### Implementation
In the end, we performed the majority of our study using Decision Trees and Logistic Regression.  We favored these methods due to the quality of the available libraries, given we were focusing on preprocessing techniques.  We also started implementing our own Hidden Markov Model for Gaussian Classification, using some of the framework for the (now unsupported) hmmlearn library but this did not achieve production quality before the deadline.
We used multi-class classification with a post-processing filter into the binary classification, to attempt to minimize the number of false positives in the end results. 
### Metrics
To evaluate the tool, accuracy and F1 Scoring were the primary metrics calculated, with others calculated so verify conclusions.  These metrics were calculated for each run in k-fold cross-validation, with k=10.
## Results and Discussion
We achieved mid-to-high 80% accuracies in the multi-class problem and near 100% accuracies in the binary classification.  Showing there is good potential for computationally-simple machine learning algorithms with mimimal preprocessing to provide the basis for early-fall detection algorithms.  Also since the accuracies were so high, this may imply that there is room for improvement in making the slice sizes even smaller.

## Dependencies and Running the code
This project was developed in python3, while attempts were made to ensure pypy compatibility, there is not full compliance given the current state of numpypy compatibility, thus we recommend using the basic python3 to run the files.
This project is dependent on sklearn and numpy as well as several standard libraries (i.e. csv).

To preprocess the dataset the way we did run python3 preprocess.py in the directory above the dataset folder. Warning preprocessing will take several minutes to a few hours depending on the computer it is run on, similar warnings apply to the training of the following algorithms.

For the decisionTree file, it can be run using ipython3 in terminal by calling DTclass(filename, num_slices, k=10) method.
For example: dt.DTclass("preprocessed_1.0E+09.json", 1, 10)

To run the logistic regression  run python3 log_reg.py in the same directory where preprocess.py was run.

There are comments in the code that can be toggled to vary the level of verbosity, among other quality of life things, and the files util.py and load_data.py provide helper functions and utilities for the other files.

## Future Work
Closer on the horizon, the next steps would include: improving the numerical stability of the preprocessing functions to allow us to look at smaller time slices, exploring more of the preprocessing space to further see how small we can go while maintaining accuracy and low false positives, and getting our Hidden Markov Model up to submission quality.
Further out there is work to be done in: reinforcing the models with data from other datasets and making the robust to more types of smartphones, obtaining access to and integrating datasets with naturally recorded samples, collection of a larger dataset, and integration with smartphones.
## References
[1] Mauldin, T.R.; Canby, M.E.; Metsis, V.; Ngu, A.H.H.; Rivera, C.C. SmartFall: A Smartwatch-Based Fall Detection System Using Deep Learning. Sensors 2018, 18, 3363.
[2] Chatzaki C., Pediaditis M., Vavoulas G., Tsiknakis M. (2017) Human Daily Activity and Fall Recognition Using a Smartphone’s Acceleration Sensor. In: Rocker C., O’Donoghue J., Ziefle M., Helfert M., Molloy W. (eds) Information and Communication Technologies for Ageing Well and e-Health. ICT4AWE 2016. Communications in Computer and Information Science, vol 736, pp 100-118. Springer, Cham, DOI 10.1007/978-3-319-62704-5_7.