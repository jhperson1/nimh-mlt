Scientific Goal: Understand how mood affects gambling
Data Set: Mood Manipulation Task
Statistical Goal: Examine how neural networks and logistic regressions compare in modeling the data.
Computational Goal: Create a predictive and interpretable model of gambling probability that works for any trial for any subject.

Feature Selection

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
- End: csv file with the names and average stabilities of all features (averaged across lambda values), npy file with the average stabilities of all features






Modeling: LOOCV


Modeling: Transfer


Modeling: Entire Data Set



