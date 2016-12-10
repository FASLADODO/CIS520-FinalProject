# CIS520-FinalProject

## Code explanation: 
For each model, we typically have two files specific to that
model. One is a script (i.e. SVM.m) and the other is a function of the form
getYHatXXX.m (i.e. getYHatSVM.m). The reason we did this is because we have two
functions, crossValidate.m and crossValError.m (the latter of which is a wrapper
around the former that returns the training and validation errors), 
which take in the observations, their labels, and a function
handle. This function handle is one of the getYHatXXX functions, and takes in
training observations, training labels, and validation observations. Then, it
trains a model on the training data and generates yhats for the training data
and the validation data. 

Thus, going back to the fact that each model has a script and a function, we
used the scripts mainly for loading the data and running the crossValError 
function to get the training and validation errors from cross validation. These
scripts were also sometimes modified to do hyperparameter tuning, as we could
just wrap repeated calls to crossValError in for loops that iterated over
possible hyperparameters.

## Generative Method: 
Model: Naive Bayes
Files: NaiveBaes.m, getYHatNB.m
How to run: run('NaiveBaes.m')

## Discriminative Method: 
Model: SVM
Files: SVM.m, getYHatSVM.m, SVMTuned.m
How to run: run('SVM.m')

## Instance Based Method: 
Model: KNN
Files: KNN.m, getYHatKNN.m
How to run: run('KNN.m')

## Semi-Supervised Dimensionality Reduction: 
Model: PCA
Files: dim_reduce.m
How to run: dim_reduce(X, desiredNumberOfComponents)
