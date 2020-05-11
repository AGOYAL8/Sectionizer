###################################Classifiers##############################################

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
from sklearn import metrics

# load the dataset
df = pd.read_csv('C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/sectionizer_final.csv', sep=',', encoding= 'unicode_escape')
print(df.head())

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

y = df['same_section']
x = df.drop(['same_section'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("*********RF Algorithm******")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
fittedModel = model.fit(x_train, y_train)
predictions = fittedModel.predict(x_test)
print("Classification Report")
print(classification_report(y_test, predictions))
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
#
#
print("*********Naive Bayes Algorithm******")
model = BernoulliNB()
fittedModel = model.fit(x_train, y_train)
predictions = fittedModel.predict(x_test)
print("Classification Report")
print(classification_report(y_test, predictions))
print("Confusion Matrix")
print(confusion_matrix(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
#
print("*********Logistic Regression:*******")

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(x_train, y_train)
logpredictions = logreg.predict(x_test)
print("Classification Report")
print(classification_report(y_test, logpredictions))
print("Confusion Matrix")
print(confusion_matrix(y_test, logpredictions))
print("Accuracy score", accuracy_score(y_test, logpredictions))


print("********* Perceptron ***********")
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=0)
ppn.fit(x_train, y_train)
ppn_predict = ppn.predict(x_test)
print("Classification Report")
print(classification_report(y_test, ppn_predict))
print("Confusion Matrix")
print(confusion_matrix(y_test, ppn_predict))
print("Accuracy score", accuracy_score(y_test, ppn_predict))

print("*******Decision Tree Classifier*********")
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)
print("Classification Report")
print(classification_report(y_test, clf_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test, clf_pred))
print("Accuracy score", accuracy_score(y_test, clf_pred))


print("*********Ridge***********")

rdg = RidgeClassifier().fit(x_train, y_train)
rdg_pred = rdg.predict(x_test)
print("Classification Report")
print(classification_report(y_test, rdg_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test, rdg_pred))
print("Accuracy score", accuracy_score(y_test, rdg_pred))


print("*********LASSO***********")
from sklearn.linear_model import Lasso
lasso = Lasso()
lsr = lasso.fit(x_train, y_train)
lsr_pred = lsr.predict(x_test)
print("Classification Report")
print(classification_report(y_test, lsr_pred.round()))
print("Confusion Matrix")
print(confusion_matrix(y_test, lsr_pred.round()))
print("Accuracy score", accuracy_score(y_test, lsr_pred.round()))
#
