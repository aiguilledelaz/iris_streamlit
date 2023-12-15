from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

iris = datasets.load_iris()
features = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

model = RandomForestClassifier()
model.fit(features, target)

import pickle
pickle.dump(model, open('models/model_iris', 'wb'))