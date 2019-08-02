Scientific Goal: Understand how mood affects gambling
Data Set: Mood Manipulation Task
Statistical Goal: Examine how neural networks and logistic regressions compare in modeling the data.
Computational Goal: Create a predictive and interpretable model of gambling probability that works for any trial for any subject.

Feature Selection

- Start: Raw Data from Mood Manipulation Interface Task
- Data Analysis: use norm_features.ipynb.
-- Maps all features to the same general space of centered at 0 and with standard deviation 1
-- 'normed' option: Converts raw point values of the task to normalized point values, such that 0 remains in the same place but the raw point values have standard deviation 1
-- features calculated: The subject characteristics will be age, gender, and diagnosis. The trial parameters pertaining to the current trial will an indicator (with 1 if for this trial, choosing to gamble will always yield more money than choosing not to gamble), current expected reward (average of not gamble reward option and two gamble reward options), and gambling range (difference between higher gamble reward option and lower gamble reward option). The trial parameters pertaining to the outcomes of past trials are an exponential sum of past reward prediction errors and an exponential sum of past outcomes. To elaborate, reward prediction error is actual reward outcome - expected reward outcome while the exponential sums are weighed so more recent trials have large weights closer to 1 and more previous trials have smaller weights closer to 0.
- End: x and y csv files where x contains main features to predict gambling probability (0 - 1) and y contains the target gambles themselves (1's or 0's)

- Start: x csv file where x contains main features to predict gambling probability (0 - 1) and y contains the target gambles themselves (1's or 0's)
- Data Analysis: interactions.py to compute quadratic interactions of main features
- End: x csv file with interactions, npy file with all column names

- Start: x csv file with interactions, npy file with all column names
- Data Analysis: standardize.py to z-score all the features
- End: x csv file with interactions, npy file with all column names, all features are standardized. This is useful for stability selection which has stability results/ rankings that are very sensitive to standard deviation of features.

- Start: x csv file with interactions, npy file with all column names, all features are standardized.
- Data Analysis: lr_stab2_features_reg.py
- End:


Modeling: LOOCV


Modeling: Transfer


Modeling: Entire Data Set



