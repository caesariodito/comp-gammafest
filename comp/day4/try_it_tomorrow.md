Here is an example code that demonstrates how to use some of the techniques mentioned earlier to reduce overfitting in XGBoost:

import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
# X, y = load_data()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Set parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'eval_metric': 'auc'
}

# Train model with early stopping
num_rounds = 1000
watchlist = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(params, dtrain, num_rounds, watchlist,
                  early_stopping_rounds=10, verbose_eval=True)
In this example, we first load the data and split it into training and validation sets using the train_test_split function from scikit-learn. Then we convert the data into XGBoost’s DMatrix format.

Next, we set the parameters for the XGBoost model. We set the max_depth parameter to control the depth of the trees and the subsample and colsample_bytree parameters to control subsampling of rows and columns. We also set the reg_alpha and reg_lambda parameters to control L1 and L2 regularization.

Finally, we train the model using the xgb.train function and specify an early_stopping_rounds parameter to stop training if the performance on the validation set does not improve for 10 rounds. We also pass a watchlist to the xgb.train function to monitor the performance on both the training and validation sets during training.