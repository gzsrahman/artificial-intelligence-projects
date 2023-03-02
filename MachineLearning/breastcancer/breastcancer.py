from sklearn import datasets
import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score
import warnings

# The returned warnings are meaningless within the scope of this project
# Thus, I'm choosing to ignore them
warnings.filterwarnings("ignore")

# Loading breast cancer dataset
cancer = datasets.load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

# The dataset measures info regarding tumors and whether or not the person has breast cancer
# We have more information than we really need regarding the tumors, some of which may interrupt
# the proficiency of our model. Thus, we're going to restrict our inputs to the average radius,
# texture, perimeter, area, and smoothness. These inputs are going to be used to predict whether
# the target has cancer or not, where y = 0 indicates a malignant tumor (a.k.a. breast cancer)
# and y = 1 indicates a benign tumor
columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[columns]
y = cancer['target']

# Splitting our training/test 70/30
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state=42)

# Let's fit a logistic regression model to our training data
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

# We'll use our model to predict y values based on our test values
y_pred = log_reg.predict(X_test)

# The accuracy score is really just Accuracy = (TP+TN)/(TP+FP+FN+TN)
# Where T/F N/P correspond to True/False Negative/Positive
print("Our logistic regression model had an accuracy score of: " + str(100 * accuracy_score(y_test, y_pred)) + "%")

# Demonstrating precision, recall, and f1-score so that we can measure the proficiency of
# our machine learning model
print(classification_report(y_test, y_pred, labels=[0, 1]))

# Noting how many points we got wrong
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
info = confusion_matrix(y_test, y_pred)
print("There were " + str(info[0][1]) + " false negatives and " + str(info[1][0]) + " false positives.")


# Plotting our predicted T/F values versus the true T/F values for cancer classification
# This offers information about how many true positives and true negatives we found (good),
# as well as how many false positives and false negatives we found (bad)
plot_confusion_matrix(log_reg, X_test, y_test)
plt.show()