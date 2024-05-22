# Name: GradientBoosting.py
# Team: AI Wizards
# Author: Dexter Kong
# Date: 5/5/2024
# Description: Implementation of Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import Data - Change Filepath if needed.
xTestLoc = "gp/test_x.data"
yTestLoc = "gp/test_y.data"
xTrainLoc = "gp/train_x.data"
yTrainLoc = "gp/train_y.data"
xValidLoc = "gp/valid_x.data"
yValidLoc = "gp/valid_y.data"
xTest = pd.read_csv(xTestLoc)
yTest = pd.read_csv(yTestLoc)
xTrain = pd.read_csv(xTrainLoc)
yTrain = pd.read_csv(yTrainLoc)
xValid = pd.read_csv(xValidLoc)
yValid = pd.read_csv(yValidLoc)

# Create Gradientboost Regressor obj
startInitialRun = input("Start initial run? (Y/N): ")
if startInitialRun == "Y" or startInitialRun == 'y':
    print("Beginning Initial Run...")
    gReg = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

    model = gReg.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    rTwo = r2_score(yPred, yTest)
    print("r2: " + str(rTwo))

    # Plot the model
    fImportance = model.feature_importances_
    # Importances are relative to max importance
    fImportance = 100.0 * (fImportance / fImportance.max())
    sIdx = np.argsort(fImportance)
    pos = np.arange(sIdx.shape[0]) + .5
    plt.barh(pos, fImportance[sIdx], align="center")
    plt.yticks(pos, model.feature_names_in_)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

    mAbsError = mae(yValid, yPred)
    print("mae: " + str(mAbsError))

    print("Initial Run End.")

print("Begin Hypertuning...")

# Hypertuning
LR = {'learning_rate':[0.2,0.25,0.3,0.35,0.4], 'n_estimators':[13,14,15,16,17], 'max_depth':[2, 4, 6, 8, 10]}
# Modify LR for different parameters, Note that setting n_estimaters to the 1000s resulted in the program taking 20+ minutes to complete
tuning = GridSearchCV(estimator = GradientBoostingRegressor(), param_grid=LR, scoring='r2')
tuning.fit(xTrain, yTrain)
tbp = tuning.best_params_
tbs = tuning.best_score_
tPred = tuning.predict(xTest)
tuningAbs = mae(yValid, tPred)
print("Tuning Best Parameters (r2): " + str(tbp))
print("Tuning Best Score (r2): " + str(tbs))
print("tPred: " + str(tuningAbs))
trt = r2_score(tPred, yTest)
print("r2: " + str(trt))

# plot the hypertuned model
tImportance = tuning.best_estimator_.feature_importances_
tImportance = 100.0 * (tImportance / tImportance.max())
tIDx = np.argsort(tImportance)
tPos = np.arange(tIDx.shape[0]) + .5
plt.barh(tPos, tImportance[tIDx], align="center")
plt.yticks(tPos, tuning.feature_names_in_)
plt.xlabel('Relative Importance')
plt.title('Hypertuned Varirable Importance')
plt.show()


print("Hypertuning End.")
print("Program End.")