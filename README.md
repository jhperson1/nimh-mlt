- Scientific Goal: Understand how mood affects gambling
- Data Set: Mood Manipulation Task
- Statistical Goal: Examine how neural networks and logistic regressions compare in modeling the data.
- Computational Goal: Create a predictive and interpretable model of gambling probability that works for any trial for any subject.

Subjects:

[Explore Subject Characteristics] * necessary step for future analyses
- Start: Raw Data from Mood Manipulation Interface Task
- Data Analysis: gp.py
- End: csv file called 'gps_[experiment].csv' which lists subjects and their characteristics. The subject ids in this file will be used for future analyses that look at cross-subject validation for example

Feature Selection:

[Create Main Features]
- Start: Raw Data from Mood Manipulation Interface Task
- Data Analysis: use norm_features.ipynb.
-- Maps all features to the same general space of centered at 0 and with standard deviation 1
-- 'normed' option: Converts raw point values of the task to normalized point values, such that 0 remains in the same place but the raw point values have standard deviation 1
-- features calculated: The subject characteristics will be age, gender, and diagnosis. The trial parameters pertaining to the current trial will an indicator (with 1 if for this trial, choosing to gamble will always yield more money than choosing not to gamble), current expected reward (average of not gamble reward option and two gamble reward options), and gambling range (difference between higher gamble reward option and lower gamble reward option). The trial parameters pertaining to the outcomes of past trials are an exponential sum of past reward prediction errors and an exponential sum of past outcomes. To elaborate, reward prediction error is actual reward outcome - expected reward outcome while the exponential sums are weighed so more recent trials have large weights closer to 1 and more previous trials have smaller weights closer to 0.
- End: x and y csv files where x contains main features to predict gambling probability (0 - 1) and y contains the target gambles themselves (1's or 0's)

[Calculate Quadratic Interactions]
- Start: x csv file where x contains main features to predict gambling probability (0 - 1) and y contains the target gambles themselves (1's or 0's)
- Data Analysis: interactions.py to compute quadratic interactions of main features
- End: x csv file with interactions, npy file with all column names

[Standardize All Features]
- Start: any x csv file
- Data Analysis: standardize.py to z-score all the features
- End: x csv file with interactions, npy file with all column names, all features are standardized. This is useful for stability selection which has stability results/ rankings that are very sensitive to standard deviation of features.

[Stability Plots and Regularization Plots]
- Start: x csv file with interactions, npy file with all column names, all features are standardized.
- Data Analysis: lr_stab2_features_reg.py (create regularization plot with the functions reggraph(x, y, NFEATS) then reggraphsave(x,y); create stability plot with functions stabgraph(x, y) or stabgraphsavenolegend(x, y) if you don't want the legent then stabgraphsave(x, y) or stabgraphsavenolegend(x, y)
 or stabgraphsavefeat(x, y, l) if you want a specific list of them)
- End: npy files for the stability values and regression weights at different levels of L1 regularization

[Average Stability Calculations]
- Start: npy files for the stability values at different levels of L1 regularization
- Data Analysis: areas.py and areas_to_npy.py
- End: csv file with the names and average stabilities of all features (averaged across lambda values), npy file with ranking of indices of the top most stable (on average) features

[Picking Number of Most Stable Features]
- Start: ranking of indices of most stable features
- Data Analysis: lr_acc_features.py
- End: graph of LOOCV accuracy of models trained on 0 features, 1 top most stable feature, top 2 most stable features, top 3 most stable features, ..., all the way to 45 features (9 main features + 36 interactions). You can pick a sufficient number of features by seeing which # of features achieves the global maximum LOOCV accuracy
- pro tip: you can go back and visualize stability and regularization plots for only these features by using running the following commands in lr_stab2_features.py to use NFEATS:
--ind_a = np.load("areas_areas_indices_" + ET + ".npy")
--ind_m = np.load("areas_maxes_indices_" + ET + ".npy")
--print ind_a
--print ind_m
--l = ind_a[:NFEATS]
--print len(l)
--stabgraphsavefeat(x, y, l)
--reggraphsavefeat(x, y, l)



Modeling:

Logistic Regression:

[Estimate LOOCV accuracy]
- Run lr_folds_features.py
- End: csv file of results

[Estimate Transfer Accuracy]:
- Run lr_test_acc_features.py
- End: csv file of results

[Estimate Weights of a Model Trained on Entire Data Set]

- Run modelweightcomparison.py
- End: npy files with weights of logistic regression

[Visualize Weights of a Model Trained on Entire Data Set]
- Run rankmodelweights.py
- End: csv file of model weights

[Compare Weights of a Model Trained on Entire Data Set]

- Run modelweightcomparisonheatmap.py
- End: heatmap with weights of logistic regression



Neural Network:

[Estimate LOOCV accuracy]
- Run nn_folds.py
- End: csv file of results

[Estimate Weights of a Model Trained on Entire Data Set and Bootstrap Resamplings]
- Run nn_boot_hessians_gradients.py
-- [DOESNT WORK YET] setting the seed = 0 gets us the original dataset
-- setting the seed to > 0 gets us some bootstrap resampling of the original dataset
- End: npy files with hessian variables

[Visualize average weights of quadratic terms of models, Averaged Across 100 Bootstraps, using Hessian matrices]

- Run visualize_hessian.py
- End: heatmaps of prevalence, positive ratios, negative ratios and average hessian values

[Visualize average weights of linear main effect terms of models, Averaged Across 100 Bootstraps, using gradients]

- [DOESNT WORK YET] Run visualize_gradient.py
- End: heatmaps of prevalence, positive ratios, negative ratios and average gradient values