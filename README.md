# CIS520-FinalProject
## Team Members:
Seth Bartynski (sbarty), Trevin Gandhi (gandhit), Brady Neal (nealb)

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

## Notes:
- All getYHatXXX functions are in the Models/ folder
- Functions such as cross validation and plotting are in the Utils/ folder
- All train and validation errors below were the results from 10-fold
cross validation

## Generative Method: 
Model: Naive Bayes  
Files: NaiveBaes.m, getYHatNB.m  
How to run: run('NaiveBaes.m')  
Train error: 0.307481  
Validation error: 0.326000  

## Discriminative Method: 
Model: SVM  
Files: SVM.m, getYHatSVM.m, SVMTuned.m  
How to run: run('SVM.m')  
Train error: 0.136691  
Validation error: 0.204444  

## Instance Based Method: 
Model: KNN  
Files: KNN.m, getYHatKNN.m  
How to run: run('KNN.m')  
Train error: 0.208691  
Validation error: 0.271333  

## Semi-Supervised Dimensionality Reduction: 
Model: PCA  
Files: dim_reduce.m, getSemiSupervisedProjections.m  
How to run: dim_reduce(X, desiredNumberOfComponents), 
            getSemiSupervisedProjections()


## Other things we tried:
- Neural Networks (NNTuning.m, NeuralNetworkCortexsys.m, getOneHotY.m,
getYHatNN.m) in UnusedModels/NeuralNetworks and Tuning/

- GloVe (loadGloVe.m, GloVe.m, gloveTransform.m, evaluateGloVe.m)
in UnusedModels/GloVe

- Random Forest (RandomForest.m, RandomForestTuned.m, random_forest.m,
getYHatRandomForest.m) in UnusedModels/ and Tuning/

- Logistic Regression (LogisticTuned.m) in Tuning/

- Discriminant Ensemble over Random Subspaces (Ensemble.m, getYHatEnsemble.m,
discrimEnsemble.m) in UnusedModels/Ensembling

- RBF SVM (RBFTuned.m) in Tuning/

- Stacking (Stacking.m, getYHatStacking.m) in UnusedModels/Ensembling

- TF/IDF (tfidf.m) in UnusedModels/TFIDF

- Boosting (Boosting.m) in UnusedModels/Ensembling

- GMM (GMM.m, getYHatGMM.m) in UnusedModels/

- Isomap (Isomap.m, IsomapTuning.m, L2_distance.m) in UnusedModels/Isomap
