def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots

from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn

image_directory = 'occlusion_train'
X, y = get_images.get_images(image_directory) #image, label
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 68, False) #image, label

Xtest, ytest = get_images.get_images('occlusion_test')
Xtest, ytest = get_landmarks.get_landmarks(Xtest, ytest, 'landmarks/', 68, False)

print(f"dimensions: {X.ndim}")

clf = ORC(SVC(probability=True)) 
clf.fit(X, y)
matching_scores_svc = clf.predict_proba(Xtest)

clf = ORC(knn()) 
clf.fit(X, y)
matching_scores_knn = clf.predict_proba(Xtest)

matching_scores = (matching_scores_svc + matching_scores_knn) / 2.0

gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores = pd.DataFrame(matching_scores, columns=classes)

for i in range(len(ytest)): #ytest
    scores = matching_scores.loc[i]
    mask = scores.index.isin([ytest[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])

performance_plots.performance(gen_scores, imp_scores, 'kNN-feature_fusion', 100)



    
    
