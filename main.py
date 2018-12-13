import os
import sys
import time
import glob
import datetime
import sqlite3
import numpy as np
import pandas as pd
from scipy import interp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

msd_subset_path = '/home/hai/Documents/School/Master/Fall2018/ML/Final/ml_final/MillionSongSubset'
msd_subset_data_path = os.path.join(msd_subset_path, 'data')
msd_subset_addf_path = os.path.join(msd_subset_path, 'AdditionalFiles')
assert os.path.isdir(msd_subset_path)

import hdf5_getters as GETTERS

def strtimedelta(starttime, stoptime):
    return str(datetime.timedelta(seconds=stoptime-starttime))

def apply_to_all_files(basedir, func = lambda x:x, ext = '.h5'):
    count = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        count += len(files)
        for f in files:
            func(f)
    return count

print("Number of song files: " + str(apply_to_all_files(msd_subset_data_path)))

all_artist_names = set()
def get_artist_name(filename):
    h5 = GETTERS.open_h5_file_read(filename)
    artist_name = GETTERS.get_artist_name(h5)
    all_artist_names.add(artist_name)
    h5.close()

all_popularity = []
def get_hotness(filename):
    h5 = GETTERS.open_h5_file_read(filename)
    popularity = GETTERS.get_song_hotttnesss(h5)
    all_popularity.append(popularity)
    h5.close()

all_timbre = []
def get_timbre(filename):
    h5 = GETTERS.open_h5_file_read(filename)
    timbre = GETTERS.get_segments_timbre(h5)
    all_timbre.append(timbre)
    h5.close()

t1 = time.time()
apply_to_all_files(msd_subset_data_path, func=get_hotness)
t2 = time.time()
print("Found : " + str(len(all_popularity)) + " in " + strtimedelta(t1,t2))

for k in range(5):
    # print(str(list(all_song_names)[k]) + " " + str(list(all_popularity)[k]))
    print(list(all_timbre)[k])

tim = all_timbre[0]

# conn = sqlite3.connect(os.path.join(msd_subset_addf_path, 'subset_track_metadata.db'))

# q = "SELECT DISTINCT artist_name FROM songs"
# t1 = time.time()
# res = conn.execute(q)
# all_artist_names_sqlite = res.fetchall()
# t2=time.time()
# print("All artist names in : " + strtimedelta(t1,t2))

k = np.array(all_popularity)
dataset = pd.read_csv('SongCSV.csv')
dataset2 = pd.read_csv('music.csv')

# dataset2 = dataset2.drop('time_signature_confidence')
# dataset2 = dataset2.drop('time_signature')

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
dataset2['song.hotttnesss'].to_csv('/home/hai/Documents/School/Master/Fall2018/ML/Final/ml_final/hotness.csv')
dataset2.to_csv('/home/hai/Documents/School/Master/Fall2018/ML/Final/ml_final/csv_data.csv')

X_train, X_test, y_train, y_test = train_test_split(data.values, labels, test_size = 0.33)

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

cv = StratifiedKFold(n_splits=6)
clf = GaussianNB()
clf = svm.SVC(kernel='rbf',probability=True)
clf = RandomForestClassifier(n_estimators=10, max_depth=4)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
pred = clf.predict(X_test)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(data.values, labels):
    probas_ = clf.fit(data.values[train], labels[train]).predict_proba(data.values[test])
    fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.5,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

f_score = f1_score(y_test, pred)

# feat_imp = pd.Series(clf.get_booster().get_fscore()).sort_values(ascending=False)
# print(feat_imp)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
# plt.show()

# XGBoost for feature importance
model_xgb = XGBClassifier(n_estimators=50, max_depth=5)
model_xgb.fit(X_train, y_train)



import tensorflow as tf
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_acc)



# Location
data = dataset2[['song.hotttnesss','longitude', 'latitude']]
hot_data = data.loc[data['song.hotttness'] == 1]

hot_data = data.loc[data['song.hotttnesss'] == 1].values
non_hot = data.loc[data['song.hotttnesss'] == 0].values

plt.scatter(hot_data[:,1], hot_data[:,2])
plt.scatter(non_hot[:,1], non_hot[:,2])
plt.show()