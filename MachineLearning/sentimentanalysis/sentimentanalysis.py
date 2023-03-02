# Please run the following in your terminal in order to make the data available
# for the program to run
print("Please type \"pip install datasets\" in your terminal to attain the necessary data")
print("before running this code")

# Now we'll get our data and some fun tools
from datasets import load_dataset
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import seaborn as sns 

# Let's store our data, which is comprised of ['sentence'] and ['label']
# Essentially, it stores sentences and whether or not the corresponding
# sentence is labeled 0, 1, or 2, corresponding to "negative", "neutral",
# or "positive"
dataset = load_dataset("financial_phrasebank", 'sentences_allagree')
emod = dataset['train'].to_pandas()

# It's called 'train', but this is actually all of the data
# We still have to split it into train/test ourselves:
emotrain, emotest = ms.train_test_split(emod, test_size=0.2, random_state=5)

# Let's carve out what we know about the training and testing data
print("There are " + str(emotrain.shape[0]) + " documents in the training set.")
print("There are " + str(emotest.shape[0]) + " documents in the test set.")

# Function that counts emotion labels based on value
def numEmo(data):
  numneg = 0
  numneu = 0
  numpos = 0
  for i in data['label']:
    if i == 0:
      numneg += 1
    if i == 1:
      numneu += 1
    if i == 2:
      numpos += 1
  return numneg, numneu, numpos

# The training negative, neutral, and positive values versus the testing values
trneg, trneu, trpos = numEmo(emotrain)  
teneg, teneu, tepos = numEmo(emotest)

# Printing counted values
print("There are " + str(trneg) + " negative values, " + str(trneu) + \
      " neutral values " + "and " + str(trpos) + \
      " positive values in the training set.")
print("There are " + str(teneg) + " negative values, " + str(teneu) + \
      " neutral values " + "and " + str(tepos) +  \
      " positive values in the test set.")

# We want to use a Random Forest Classifier model to fit the data, but we need
# to prepare the data for this model. Thus, we'll use CountVectorizer() to make
# this textual data make sense to an RFC model and we'll use Pipeline() to make
# them work together
text_clf = Pipeline([('vect', CountVectorizer()),('clf', RandomForestClassifier(random_state=5))])

# Here we'll fit our model to the training data and then create predicted values
# for the test data
text_clf.fit(emotrain['sentence'], emotrain['label'])
emopred = text_clf.predict(emotest['sentence'])

# Here we'll compare the predicted sentiment values to the actual sentiment values
# to test the accuracy of our model
emoacc = metrics.accuracy_score(emotest['label'], emopred)
print("The accuracy score of the Random Forest Classifier is: ", end="")
print(str(emoacc))

teminus = 0
teneu = 0
tepos = 0
for i in emotest['label']:
  teminus += bool(i == 0)
  teneu += bool(i == 1)
  tepos += bool(i == 2)

prminus = 0
prneu = 0
prpos = 0
for i in range(len(emopred)):
  prminus += int(emopred[i]==0)
  prneu += int(emopred[i]==1)
  prpos += int(emopred[i]==2)

print("There are " + str(prminus) + " negative values, " + str(prneu) + \
      " neutral values " + "and " + str(prpos) + \
      " positive values in the predicting set.")
print("There are " + str(teminus) + " negative values, " + str(teneu) + \
      " neutral values " + "and " + str(tepos) +  \
      " positive values in the test set.")

# Made a function that judges the estimates of my predictions so my print
# statements are dynamically changing
def accStr(val1, val2):
  if val1 == val2:
    return "accurately estimated"
  if val1 > val2:
    return "overestimated"
  else:
    return "underestimated"

# Reflecting on my predicted values versus the actual values in the test data
print("This indicates that I " + accStr(prminus,teminus) + " negative values.")
print("Also, I " + accStr(prneu,teneu) + " neutral values.")
print("Finally, I " + accStr(prpos,tepos) + " positive values.")

# Predicting probabilities of each value and then making density plot
print("See below a plot of the predicted probability values in this model:")
probapred = text_clf.predict_proba(emotest['sentence'])
sns.kdeplot(data = probapred, legend = True)