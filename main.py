import sys
import time
import glob
import datetime
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset2 = pd.read_csv('music.csv')
drop_list = ['location', 'artist.name', 'similar', 'release.id', 'artist_mbtags_count', 'artist.id', 'song.id', 'release.name', 'terms', 'title', 'artist_mbtags']

prep = preprocessing.LabelEncoder()
dataset2['familiarity'] = prep.fit_transform(dataset2['familiarity'].fillna(dataset2['familiarity'].median()))
dataset2 = dataset2.drop(drop_list, axis = 1)
dataset2 = dataset2.dropna()
data = dataset2.drop('song.hotttnesss', axis=1)

data = dataset2[['artist.hotttnesss', 'tempo', 'duration', 'loudness', 'longitude', 'latitude', 'beats_start', 'familiarity']]
labels = dataset2['song.hotttnesss']
labels = labels.values
labels[labels >= 0.5] = 1
labels[labels < 0.5] = 0

X_train, X_test, y_train, y_test = train_test_split(data.values, labels, test_size = 0.33)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics

def generate_roc(k, data, labels, model, model_name, neural_network):
    cv = StratifiedKFold(n_splits= k)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    for train_i, test_i in cv.split(data.values, labels):
        if neural_network == False:
            probas = model.fit(data.values[train_i], labels[train_i]).predict_proba(data.values[test_i])
        else:
            probas = model.fit(data.values[train_i], labels[train_i]).predict_proba(data.values[test_i])
        fpr, tpr, thresholds = roc_curve(labels[test_i], probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.5,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()

model_NBC = GaussianNB()
model_NBC.fit(X_train, y_train)
print('Gaussian Naive Bayes Accuracy: {}'.format(model_NBC.score(X_test, y_test)))
generate_roc(5, data, labels, model_NBC, 'Gaussian Naive Bayes', False)

model_RFC = RandomForestClassifier(n_estimators=10, max_depth=4)
model_RFC.fit(X_train, y_train)
print('Random Forest Accuracy: {}'.format(model_NBC.score(X_test, y_test)))
generate_roc(5, data, labels, model_RFC, 'Random Forest', False)

# Tunning Random Forest

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(i) for i in np.linspace(200, 2000, 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model_RFC_base = RandomForestClassifier()
model_rfc_random = RandomizedSearchCV(estimator = model_RFC_base, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
model_rfc_random.fit(X_train, y_train)
print("Best params for random forest: {} \n Best Accuracy: {}".format(model_rfc_random.best_params_, model_rfc_random.best_score_))

tuned_model_RFC = RandomForestClassifier(n_estimators=600, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=40, bootstrap=True)
tuned_model_RFC.fit(X_train, y_train)
tuned_acc = tuned_model_RFC.score(X_test, y_test)
model_RFC_base.fit(X_train, y_train)
base_acc = model_RFC_base.score(X_test, y_test)

predictors = ['artist.hotttnesss', 'tempo', 'duration', 'loudness', 'longitude', 'latitude', 'beats_start', 'familiarity']
target = 'song.hotttnesss'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=True, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['song.hotttnesss'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['song.hotttnesss'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['song.hotttnesss'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

# XGBoost for feature importance
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
model_xgb.score(X_test, y_test)
feat_imp = pd.Series(model_xgb.get_booster().get_fscore()).sort_values(ascending=False)
k = feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

#Tunning n_estimators
model_xgb = XGBClassifier(
        learning_rate = 0.1,
        n_estimators = 1000)
modelfit(model_xgb, dataset2, predictors)

#After Tunning
model_xgb = XGBClassifier(
        learning_rate = 0.1,
        n_estimators = 105)
model_xgb.fit(X_train, y_train)
model_xgb.score(X_test, y_test)
generate_roc(5, data, labels, model_xgb, "XGB", False)


import tensorflow as tf
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, input_dim=8 ,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,
             loss='binary_crossentropy',
             metrics=['accuracy']
             )
nn_history = model.fit(X_train, y_train, epochs=150, batch_size=10)
val_loss, val_acc = model.evaluate(X_test, y_test)
print('Validation accuracy: {} | Training Accuracy: {}'.format(val_acc, np.mean(nn_history['acc'])))

# Tunning number of epochs
z = list(range(20,300,20))
acc = []
for i in range(200, 300, 20):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=8 ,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'],
                )
    model.fit(X_train, y_train, epochs=i, batch_size=10)
    val_loss, val_acc = model.evaluate(X_test, y_test)
    acc.append(val_acc)

plt.plot(z, acc)
plt.show()


# Display locations of songs on the world map
import os
os.environ["PROJ_LIB"] = "C:\\Users\\haihu\\Anaconda3\\Library\\share"
from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(14, 8))
earth = Basemap()
earth.bluemarble(alpha=0.3)

data = dataset2[['song.hotttnesss','longitude', 'latitude']]

hot_data = data.loc[data['song.hotttnesss'] == 1].values
non_hot = data.loc[data['song.hotttnesss'] == 0].values

plt.scatter(non_hot[:,1], non_hot[:,2], color='green', label='non-popular')
plt.scatter(hot_data[:,1], hot_data[:,2], color='red', label='popular')
plt.legend(loc='lower left')
plt.show()