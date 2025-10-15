# Step 1 - Import all required binaries for applying SVM regressor on abalone data
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2 - Read data into a data frame
names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv('/content/abalone.data', names=names, sep=',')

df.head()

df = df.drop('Sex', axis=1)

df.columns

# Step 4 - Define X and y
X = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
       'Viscera weight', 'Shell weight']]
y = df['Rings']

# Step 5 - Split X and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

sv = SVR(kernel='rbf')

sv.fit(X_train, y_train)

sv.score(X_test, y_test)

y_pred = sv.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(results.head())

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual vs. Predicted Rings (SVR with RBF Kernel)")
plt.show()
