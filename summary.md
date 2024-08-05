## introduction
this repository aimed to summerize important things in the book of "approaching any maching learning problems" written by Abhishek Thakur. 

## Supervised vs unsupervised learning
**Supervised data**: always has one or multiple targets associated with it.<br>
**Unsupervised data** : does not have any target variable.

If the target is categorical, the problem becomes a classification problem. And if the target is a real number, the problem is defined as a regression problem. Thus, supervised problems can be divided into two sub-classes:

**Classification**: predicting a category, e.g. dog or cat.<br>
**Regression**: predicting a value, e.g. house prices.

Another type of machine learning problem is the *unsupervised* type. Unsupervised
datasets do not have a target associated with them and in general, are more
challenging to deal with when compared to supervised problems.
### example of unsupervised dataset: 
Let’s say you work in a financial firm which deals with credit card transactions.
There is a lot of data that comes in every second. The only problem is that it is
difficult to find humans who will mark each and every transaction either as a valid
or genuine transaction or a fraud. When we do not have any information about a
transaction being fraud or genuine, the problem becomes an unsupervised problem.
To tackle these kinds of problems we have to think about how many clusters can
data be divided into. Clustering is one of the approaches that you can use for
problems like this, but it must be noted that there are several other approaches
available that can be applied to unsupervised problems. For a fraud detection
problem, we can say that data can be divided into two classes (fraud or genuine).After a
clustering algorithm is applied, we should be able to distinguish between the two
assumed targets. To make sense of unsupervised problems, we can also use
numerous decomposition techniques such as **Principal Component Analysis(PCA)**, **t-distributed Stochastic Neighbour Embedding (t-SNE)**, etc.

Most of the time, it’s also possible to convert a supervised dataset to unsupervised
to see how they look like when plotted. For example, **MNIST** dataset which is a very popular dataset of handwritten digits, and it is a supervised problem in which you are given the images of the numbers and the correct label associated with them. You have to build a model that can identify which digit is it when provided only with the image. This dataset can easily be converted to an unsupervised setting for basic visualization. If we do a t-Distributed Stochastic Neighbour Embedding (t-SNE) decomposition of this dataset, we can see that we can separate the images to some extent just by doing with two components on the image pixels. This is one way of visualizing unsupervised datasets. We can also do k-means clustering on the same dataset and see how it performs in an unsupervised setting. One question that arises all the time is how to find the optimal number of clusters in k-means clustering. Well, there is no right answer. You have to find the number by cross-validation.

## Cross-validation
We did not build any models in the previous chapter. The reason for that is simple.
Before creating any kind of machine learning model, we must know what cross validation is and how to choose the best cross-validation depending on your datasets.
So, what is **cross-validation**, and why should we care about it?
We can find multiple definitions as to what cross-validation is. Mine is a one-liner:
cross-validation is a step in the process of building a machine learning model which
helps us ensure that our models fit the data accurately and also ensures that we do
not overfit. But this leads to another term: **overfitting**.

To explain overfitting, I think it’s best if we look at a dataset. There is a red wine-
quality dataset 2 which is quite famous. This dataset has 11 different attributes that
decide the quality of red wine. Based on these different attributes, we are required to predict the quality of red wine which is a value between 0 and 10.
We can treat this problem either as a classification problem or as a regression
problem since wine quality is nothing but a real number between 0 and 10. For
simplicity, let’s choose classification. This dataset, however, consists of only six
types of quality values. We will thus map all quality values from 0 to 5.
quality_mapping = {
3: 0,
4: 1,
5: 2,
6: 3,
7: 4,
8: 5
}
When we look at this data and consider it a classification problem, a lot of
algorithms come to our mind that we can apply to it, So, let’s start with something simple that we can visualize too: **decision trees**.
Before we begin to understand what overfitting is, let’s divide the data into two
parts. This dataset has 1599 samples. We keep 1000 samples for training and 599
as a separate set. 

```python
# use sample with frac=1 to shuffle the dataframe
# we reset the indices since they change after
# shuffling the dataframe

df = df.sample(frac=1).reset_index(drop=True)

df_train = df.head(1000)

df_test = df.tail(599)
```

We will now train a decision tree model on the training set. For the decision tree model, I am going to use scikit-learn.

```python

# import from scikit-learn
from sklearn import tree
from sklearn import metrics
# initialize decision tree classifier class
# with a max_depth of 3
clf = tree.DecisionTreeClassifier(max_depth=3)
# choose the columns you want to train on
# these are the features for the model
cols = ['fixed acidity',
'volatile acidity',
'citric acid',
'residual sugar',
'chlorides',
'free sulfur dioxide',
'total sulfur dioxide',
'density',
'pH',
'sulphates',
'alcohol']
# train the model on the provided features
# and mapped quality from before
clf.fit(df_train[cols], df_train.quality)
```
Note that I have used a max_depth of 3 for the decision tree classifier. I have left
all other parameters of this model to its default value.
Now, we test the accuracy of this model on the training set and the test set

```python
# generate predictions on the training set
train_predictions = clf.predict(df_train[cols])
# generate predictions on the test set
test_predictions = clf.predict(df_test[cols])
# calculate the accuracy of predictions on
# training data set
train_accuracy = metrics.accuracy_score(
df_train.quality, train_predictions
)
# calculate the accuracy of predictions on
# test data set
test_accuracy = metrics.accuracy_score(
df_test.quality, test_predictions
)

```

The training and test accuracies are found to be 58.9% and 54.25%. Now we
increase the max_depth to 7 and repeat the process. This gives training accuracy of
76.6% and test accuracy of 57.3%. Here, we have used accuracy, mainly because it
is the most straightforward metric. It might not be the best metric for this problem.
What about we calculate these accuracies for different values of max_depth and
make a plot?
We see that the best score for test data is obtained when max_depth has a value of
14 . As we keep increasing the value of this parameter, test accuracy remains the
same or gets worse, but the training accuracy keeps increasing. It means that our
simple decision tree model keeps learning about the training data better and better
with an increase in max_depth, but the performance on test data does not improve
at all.

This is called **overfitting**.
The model fits perfectly on the training set and performs poorly when it comes to
the test set. This means that the model will learn the training data well but will not
generalize on unseen samples. In the dataset above, one can build a model with very
high max_depth which will have outstanding results on training data, but that kind
of model is not useful as it will not provide a similar result on the real-world samples
or live data.
Another definition of overfitting would be when the *test loss increases as we keep improving training loss. This is very common when it comes to neural networks*.

Whenever we train a neural network, we must monitor loss during the training time
for both training and test set. If we have a very large network for a dataset which is
quite small (i.e. very less number of samples), we will observe that the loss for both
training and test set will decrease as we keep training. However, at some point, test
loss will reach its minima, and after that, it will start increasing even though training
loss decreases further. We must stop training where the validation loss reaches its
minimum value.
**Occam’s razor** in simple words states that one should not try to complicate things
that can be solved in a much simpler manner. In other words, the simplest solutions
are the most generalizable solutions. In general, whenever your model does not
obey Occam’s razor, it is probably overfitting.

Now we can go back to **cross-validation**.<br>
While explaining about overfitting, I decided to divide the data into two parts. I
trained the model on one part and checked its performance on the other part. Well,
this is also a kind of cross-validation commonly known as a hold-out set. We use
this kind of (cross-) validation when we have a large amount of data and model
inference is a time-consuming process.<br>
There are many different ways one can do cross-validation, and it is the most critical
step when it comes to building a good machine learning model which is
generalizable when it comes to unseen data. Choosing the right cross-validation
depends on the dataset you are dealing with, and one’s choice of cross-validation
on one dataset may or may not apply to other datasets. However, there are a few
types of cross-validation techniques which are the most popular and widely used.
These include:<br>
*k-fold cross-validation<br>
*stratified k-fold cross-validation<br>
*hold-out based validation<br>
*leave-one-out cross-validation<br>
*group k-fold cross-validation<br>
Cross-validation is dividing training data into a few parts. We train the model on
some of these parts and test on the remaining parts. <br>
when you get a dataset to build machine learning models, you separate them into two different sets: training and validation. Many people also split it into a third set and call it a test set. We will, however, be using only two sets. As you can see, we divide the samples and the targets associated with them. We can divide the data into k different sets which are exclusive of each other. This is known as **k-fold cross-validation**.<br>
We can split any data into k-equal parts using KFold from scikit-learn. Each sample
is assigned a value from 0 to k-1 when using k-fold cross validation.<br>
```python
# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
# Training data is in a CSV file called train.csv
df = pd.read_csv("train.csv")
# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# initiate the kfold class from model_selection module
kf = model_selection.KFold(n_splits=5)
# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
df.loc[val_, 'kfold'] = fold
# save the new csv with kfold column
df.to_csv("train_folds.csv", index=False)
```
The next important type of cross-validation is **stratified k-fold**. If you have a
skewed dataset for binary classification with 90% positive samples and only 10%
negative samples, you don't want to use random k-fold cross-validation. Using
simple k-fold cross-validation for a dataset like this can result in folds with all
negative samples. In these cases, we prefer using **stratified k-fold cross-validation**.
Stratified k-fold cross-validation keeps the ratio of labels in each fold constant. So,
in each fold, you will have the same 90% positive and 10% negative samples. Thus,
whatever metric you choose to evaluate, it will give similar results across all folds.<br>
It’s easy to modify the code for creating k-fold cross-validation to create stratified
k-folds. We are only changing from model_selection.KFold to
model_selection.StratifiedKFold and in the kf.split(...) function, we specify the
target column on which we want to stratify. We assume that our CSV dataset has a
column called “target” and it is a classification problem.<br>
```python   
# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
# Training data is in a csv file called train.csv
df = pd.read_csv("train.csv")
# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# fetch targets
y = df.target.values
# initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)
# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
df.loc[v_, 'kfold'] = f
# save the new csv with kfold column
df.to_csv("train_folds.csv", index=False)
```

Note that we continue on the code above. So, we have converted the target values.
Looking at figure 6 we can say that the quality is very much skewed. Some classes have a lot of samples, and some don’t have that many. If we do a simple k-fold, we won’t have an equal distribution of targets in every fold. Thus, we choose stratified k-fold in this case.<br>
But what should we do if we have a large amount of data?<br>
we can opt for a **hold-out based validation.**
In many cases, we have to deal with small datasets and creating big validation sets
means losing a lot of data for the model to learn. In those cases, we can opt for a
type of k-fold cross-validation where k=N, where N is the number of samples in the
dataset. This means that in all folds of training, we will be training on all data
samples except 1. The number of folds for this type of cross-validation is the same
as the number of samples that we have in the dataset.<br>
One should note that this type of cross-validation can be costly in terms of the time
it takes if the model is not fast enough, but since it’s only preferable to use this
cross-validation for small datasets, it doesn’t matter much. <br>

## validation for regression problems
Now we can move to regression. The good thing about regression problems is that
we can use all the cross-validation techniques mentioned above for regression
problems except for stratified k-fold. That is we cannot use stratified k-fold directly,
but there are ways to change the problem a bit so that we can use stratified k-fold
for regression problems. Mostly, simple k-fold cross-validation works for any
regression problem. However, if you see that the distribution of targets is not
consistent, you can use stratified k-fold.<br>
To use **stratified k-fold for a regression problem**, we have first to divide the target
into bins, and then we can use stratified k-fold in the same way as for classification
problems. There are several choices for selecting the appropriate number of bins. If
you have a lot of samples( > 10k, > 100k), then you don’t need to care about the
number of bins. Just divide the data into 10 or 20 bins. If you do not have a lot of
samples, you can use a simple rule like **Sturge’s Rule** to calculate the appropriate
number of bins.
### Sturge’s rule:
                        `Number of Bins = 1 + log₂(N)`


Let’s make a sample regression dataset and try to apply stratified k-fold as shown
in the following python snippet.<br>

```python 
# stratified-kfold for regression
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
def create_folds(data):
# we create a new column called kfold and fill it with -1
data["kfold"] = -1
# the next step is to randomize the rows of the data
data = data.sample(frac=1).reset_index(drop=True)
# calculate the number of bins by Sturge's rule
# I take the floor of the value, you can also
# just round it
num_bins = int(np.floor(1 + np.log2(len(data))))
# bin targets
data.loc[:, "bins"] = pd.cut(
data["target"], bins=num_bins, labels=False
)
# initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)
# fill the new kfold column
# note that, instead of targets, we use bins!
for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
data.loc[v_, 'kfold'] = f
# drop the bins column
data = data.drop("bins", axis=1)
# return dataframe with folds
return data
if __name__ == "__main__":
# we create a sample dataset with 15000 samples
# and 100 features and 1 target
X, y = datasets.make_regression(
n_samples=15000, n_features=100, n_targets=1
)
# create a dataframe out of our numpy arrays
df = pd.DataFrame(
X,
columns=[f"f_{i}" for i in range(X.shape[1])]
)
df.loc[:, "target"] = y
# create folds
df = create_folds(df).
```

Cross-validation is the first and most essential step when it comes to building
machine learning models. If you want to do feature engineering, split your data first.
If you're going to build models, split your data first. If you have a good cross-
validation scheme in which validation data is representative of training and real-
world data, you will be able to build a good machine learning model which is highly
generalizable.<br>
The types of cross-validation presented in this chapter can be applied to almost any
machine learning problem. Still, you must keep in mind that cross-validation also
depends a lot on the data and you might need to adopt new forms of cross-validation
depending on your problem and data.<br>
For example, let’s say we have a problem in which we would like to build a model
to detect skin cancer from skin images of patients. Our task is to build a binary
classifier which takes an input image and predicts the probability for it being benign
or malignant.<br>
In these kinds of datasets, you might have multiple images for the same patient in
the training dataset. So, to build a good cross-validation system here, you must have
stratified k-folds, but you must also make sure that patients in training data do not
appear in validation data. Fortunately, scikit-learn offers a type of cross-validation
known as GroupKFold. Here the patients can be considered as groups. But
unfortunately, there is no way to combine GroupKFold with StratifiedKFold in
scikit-learn. So you need to do that yourself.<br>


## Evaluation metrics
When it comes to machine learning problems, you will encounter a lot of different
types of metrics in the real world. Sometimes, people even end up creating metrics
that suit the business problem. see some of the most common metrics that you can use when starting with your very first few projects.
we will only focus on supervised.<br>
If we talk about classification problems, the most common metrics used are:
- Accuracy
- Precision (P)
- Recall (R)
- F1 score (F1)
- Area under the ROC (Receiver Operating Characteristic) curve or simply
AUC (AUC)
- Log loss
- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)<br>
When it comes to regression, the most commonly used evaluation metrics are:<br>
- Mean absolute error (MAE)
- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Root mean squared logarithmic error (RMSLE)
- Mean percentage error (MPE)
- Mean absolute percentage error (MAPE)
- R 2<br>

Knowing about how the aforementioned metrics work is not the only thing we have
to understand. We must also know when to use which metrics, and that depends on what kind of data and targets you have. I think it’s more about the targets and less about the data.<br>

When we have an equal number of positive and negative samples in a binary
classification metric, we generally use accuracy, precision, recall and f1.<br>
**Accuracy**: It is one of the most straightforward metrics used in machine learning.
It defines how accurate your model is. For the problem described above, if you
build a model that classifies 90 images accurately, your accuracy is 90% or 0.90. If
only 83 images are classified correctly, the accuracy of your model is 83% or 0.83.
Simple.<br>

```python

def accuracy(y_true, y_pred):
"""
Function to calculate accuracy
:param y_true: list of true values
:param y_pred: list of predicted values
:return: accuracy score
"""
# initialize a simple counter for correct predictions
correct_counter = 0
# loop over all elements of y_true
# and y_pred "together"
for yt, yp in zip(y_true, y_pred):
if yt == yp:
# if prediction is equal to truth, increase the counter
correct_counter += 1
# return accuracy
# which is correct predictions over the number of samples
return correct_counter / len(y_true)

```

We can also calculate accuracy using scikit-learn.<br>

Now, let’s say the dataset a bit such that there are 180 chest x-ray images
which do not have pneumothorax and only 20 with pneumothorax.Even in this
case, we will create the training and validation sets with the same ratio of positive
to negative (pneumothorax to non- pneumothorax) targets. In each set, we have 90
non- pneumothorax and 10 pneumothorax images. If you say that all images in the
validation set are non-pneumothorax, what would your accuracy be? Let’s see; you
classified 90% of the images correctly. So, your accuracy is 90%.<br>
You didn’t even build a model and got an accuracy of 90%. That seems kind of
useless. If we look carefully, we will see that the dataset is skewed, i.e., the number
of samples in one class outnumber the number of samples in other class by a lot. In
these kinds of cases, it is not advisable to use accuracy as an evaluation metric as it
is not representative of the data. So, you might get high accuracy, but your model
will probably not perform that well when it comes to real-world samples, and you
won’t be able to explain to your managers why.<br>

In these cases, it’s better to look at other metrics such as precision.
Before learning about precision, we need to know a few terms. Here we have
assumed that chest x-ray images with pneumothorax are positive class (1) and
without pneumothorax are negative class (0).<br>

**True positive (TP)**: Given an image, if your model predicts the image has
pneumothorax, and the actual target for that image has pneumothorax, it is
considered a true positive.<br>

**True negative (TN)**: Given an image, if your model predicts that the image does not
have pneumothorax and the actual target says that it is a non-pneumothorax image,
it is considered a true negative.<br>
In simple words, if your model correctly predicts positive class, it is true positive,
and if your model accurately predicts negative class, it is a true negative.<br>
**False positive (FP)**: Given an image, if your model predicts pneumothorax and the
actual target for that image is non- pneumothorax, it a false positive.<br>
**False negative (FN)**: Given an image, if your model predicts non-pneumothorax
and the actual target for that image is pneumothorax, it is a false negative.<br>

In simple words, if your model incorrectly (or falsely) predicts positive class, it is
a false positive. If your model incorrectly (or falsely) predicts negative class, it is a
false negative.<br>

Let’s look at implementations of these, one at a time.<br>

``` python
def true_positive(y_true, y_pred):
"""
Function to calculate True Positives
:param y_true: list of true values
:param y_pred: list of predicted values
:return: number of true positives
"""
# initialize
tp = 0
for yt, yp in zip(y_true, y_pred):
if yt == 1 and yp == 1:
tp += 1
return tp
def true_negative(y_true, y_pred):
"""
Function to calculate True Negatives
:param y_true: list of true values
:param y_pred: list of predicted values
:return: number of true negatives
"""
# initialize
tn = 0
for yt, yp in zip(y_true, y_pred):
if yt == 0 and yp == 0:
tn += 1
return tn
def false_positive(y_true, y_pred):
"""
Function to calculate False Positives
:param y_true: list of true values
:param y_pred: list of predicted values
:return: number of false positives
"""
# initialize
fp = 0
for yt, yp in zip(y_true, y_pred):
if yt == 0 and yp == 1:
fp += 1
return fp
def false_negative(y_true, y_pred):
"""
Function to calculate False Negatives
:param y_true: list of true values
:param y_pred: list of predicted values
:return: number of false negatives
"""
# initialize
fn = 0
for yt, yp in zip(y_true, y_pred):
if yt == 1 and yp == 0:
fn += 1
return fn

```
The way I have implemented these here is quite simple and works only for binary
classification.<br>

If we have to define accuracy using the terms described above, we can write:<br>

                Accuracy Score = (TP + TN) / (TP + TN + FP + FN)

Now, we can move to other important metrics.<br>
First one is precision. Precision is defined as:<br>

                Precision = TP / (TP + FP)


Now, since we have implemented TP, TN, FP and FN, we can easily implement
precision in python.<br>

``` python
def precision(y_true, y_pred):
"""
Function to calculate precision
:param y_true: list of true values
:param y_pred: list of predicted values
:return: precision score
"""
tp = true_positive(y_true, y_pred)
fp = false_positive(y_true, y_pred)
precision = tp / (tp + fp)
return precision

```


Next, we come to recall. Recall is defined as:<br>

                        Recall = TP / (TP + FN)


```python
def recall(y_true, y_pred):
"""
Function to calculate recall
:param y_true: list of true values
:param y_pred: list of predicted values
:return: recall score
"""
tp = true_positive(y_true, y_pred)
fn = false_negative(y_true, y_pred)
recall = tp / (tp + fn)
return recall
```

For a “good” model, our precision and recall values should be high.<br>

Most of the models predict a probability, and when we predict, we usually choose
this threshold to be 0.5. This threshold is not always ideal, and depending on this
threshold, your value of precision and recall can change drastically. If for every
threshold we choose, we calculate the precision and recall values, we can create a
plot between these sets of values. This plot or curve is known as the precision-recall
curve.

Before looking into the precision-recall curve, let’s assume two lists.<br>

``` python
In [X]: y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
...:
1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
In [X]: y_pred = [0.02638412, 0.11114267, 0.31620708,
...:
0.0490937, 0.0191491, 0.17554844,
...:
0.15952202, 0.03819563, 0.11639273,
...:
0.079377,
0.08584789, 0.39095342,
...:
0.27259048, 0.03447096, 0.04644807,
...:
0.03543574, 0.18521942, 0.05934905,
...:
0.61977213, 0.33056815]
```

So, y_true is our targets, and y_pred is the probability values for a sample being
assigned a value of 1. So, now, we look at probabilities in prediction instead of the
predicted value (which is most of the time calculated with a threshold at 0.5).

``` python
precisions = []
recalls = []
# how we assumed these thresholds is a long story
thresholds = [0.0490937 , 0.05934905, 0.079377,
0.08584789, 0.11114267, 0.11639273,
0.15952202, 0.17554844, 0.18521942,
0.27259048, 0.31620708, 0.33056815,
0.39095342, 0.61977213]
# for every threshold, calculate predictions in binary
# and append calculated precisions and recalls
# to their respective lists
for i in thresholds:
temp_prediction = [1 if x >= i else 0 for x in y_pred]
p = precision(y_true, temp_prediction)
r = recall(y_true, temp_prediction)
precisions.append(p)
recalls.append(r)
```
You will notice that it’s challenging to choose a value of threshold that gives both
good precision and recall values. If the threshold is too high, you have a smaller
number of true positives and a high number of false negatives. This decreases your
recall; however, your precision score will be high. If you reduce the threshold too
low, false positives will increase a lot, and precision will be less.<br>
Both precision and recall range from 0 to 1 and a value closer to 1 is better.<br>
F1 score is a metric that combines both precision and recall. It is defined as a simple
weighted average (harmonic mean) of precision and recall. If we denote precision
using P and recall using R, we can represent the F1 score as:

                        F1 = 2PR / (P + R)

A little bit of mathematics will lead you to the following equation of F1 based on
TP, FP and FN

                        F1 = 2TP / (2TP + FP + FN)

A Python implementation is simple because we have already implemented these.

``` python
def f1(y_true, y_pred):
"""
Function to calculate f1 score
:param y_true: list of true values
:param y_pred: list of predicted values
:return: f1 score
"""
p = precision(y_true, y_pred)
r = recall(y_true, y_pred)
score = 2 * p * r / (p + r)
return score
```

F1 score also ranges from 0 to 1, and a perfect prediction model has an F1 of 1. When dealing with datasets that have skewed targets, we should look at F1 (or precision and recall) instead of accuracy.<br>

Then there are other crucial terms that we should know about.
The first one is **TPR or True Positive Rate**, which is the same as recall.

                            TPR = TP / (TP + FN)

TPR or recall is also known as **sensitivity**.

And **FPR or False Positive Rate**, which is defined as:
                            FPR = FP / (TN + FP)    

And **1 - FPR** is known as **specificity or True Negative Rate or TNR**.<br>

These are a lot of terms, but the most important ones out of these are only TPR and
FPR.

We can  get a TPR and FPR value for each threshold. we have TPR on the y-axis and FPR
on the x-axis.<br>
This curve is also known as the **Receiver Operating Characteristic (ROC)**. And
if we calculate the area under this ROC curve, we are calculating another metric
which is used very often when you have a dataset which has skewed binary targets.<br>

This metric is known as the **Area Under ROC Curve** or **Area Under Curve** or
just simply **AUC**. There are many ways to calculate the area under the ROC curve.<br>

AUC values range from 0 to 1.<br>

- AUC = 1 implies you have a perfect model. Most of the time, it means that
you made some mistake with validation and should revisit data processing
and validation pipeline of yours. If you didn’t make any mistakes, then
congratulations, you have the best model one can have for the dataset you
built it on.
- AUC = 0 implies that your model is very bad (or very good!). Try inverting
the probabilities for the predictions, for example, if your probability for the
positive class is p, try substituting it with 1-p. This kind of AUC may also
mean that there is some problem with your validation or data processing.
- AUC = 0.5 implies that your predictions are random. So, for any binary
classification problem, if I predict all targets as 0.5, I will get an AUC of
0.5.