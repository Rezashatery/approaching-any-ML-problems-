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
0.5.<br>

But what does AUC say about our model?

Suppose you get an AUC of 0.85 when you build a model to detect pneumothorax
from chest x-ray images. This means that if you select a random image from your
dataset with pneumothorax (positive sample) and another random image without
pneumothorax (negative sample), then the pneumothorax image will rank higher
than a non-pneumothorax image with a probability of 0.85.<br>


After calculating probabilities and AUC, you would want to make predictions on
the test set. Depending on the problem and use-case, you might want to either have
probabilities or actual classes. If you want to have probabilities, it’s effortless. You
already have them. If you want to have classes, you need to select a threshold. In
the case of binary classification, you can do something like the following.<br>
                            Prediction = Probability >= Threshold

Which means, that prediction is a new list which contains only binary variables. An
item in prediction is 1 if the probability is greater than or equal to a given threshold
else the value is 0.<br>
And guess what, you can use the ROC curve to choose this threshold! The ROC
curve will tell you how the threshold impacts false positive rate and true positive
rate and thus, in turn, false positives and true positives. You should choose the
threshold that is best suited for your problem and datasets.<br>

AUC is a widely used metric for skewed binary classification tasks in the industry,
and a metric everyone should know about. Once you understand the idea behind
AUC, as explained in the paragraphs above, it is also easy to explain it to non-
technical people who would probably be assessing your models in the industry.<br>

Another important metric you should learn after learning AUC is log loss. In case
of a binary classification problem, we define log loss as:<br>
    Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )

Where target is either 0 or 1 and prediction is a probability of a sample belonging
to class 1.log loss punishes you for being very sure and very wrong.

``` python
import numpy as np
def log_loss(y_true, y_proba):
"""
Function to calculate log loss
:param y_true: list of true values
:param y_proba: list of probabilities for 1
:return: overall log loss
"""
# define an epsilon value
# this can also be an input
# this value is used to clip probabilities
epsilon = 1e-15
# initialize empty list to store
# individual losses
loss = []
# loop over all true and predicted probability values
for yt, yp in zip(y_true, y_proba):
# adjust probability
# 0 gets converted to 1e-15
# 1 gets converted to 1-1e-15
# Why? Think about it!
yp = np.clip(yp, epsilon, 1 - epsilon)
# calculate loss for one sample
temp_loss = - 1.0 * (
yt * np.log(yp)
+ (1 - yt) * np.log(1 - yp)
)
# add to loss list
loss.append(temp_loss)
# return mean loss over all samples
return np.mean(loss)
```

Implementation of log loss is easy. Interpretation may seem a bit difficult. You must remember that log loss penalizes a lot more than other metrics.<br>
For example, if you are 51% sure about a sample belonging to class 1, log loss
would be:<br>
                    - 1.0 * ( 1 * log(0.51) + (1 - 1) * log(1 – 0.51) ) = 0.67<br>
And if you are 49% sure for a sample belonging to class 0, log loss would be:<br>
                    - 1.0 * ( 0 * log(0.49) + (1 - 0) * log(1 – 0.49) ) = 0.67

So, even though we can choose a cut off at 0.5 and get perfect predictions, we will
still have a very high log loss. So, when dealing with log loss, you need to be very
careful; any non-confident prediction will have a very high log loss.<br>
There are three different ways to calculate this which might get confusing from time
to time. Let’s assume we are interested in precision first. We know that precision
depends on true positives and false positives.<br>
- Macro averaged precision: calculate precision for all classes individually
and then average them
- Micro averaged precision: calculate class wise true positive and false
positive and then use that to calculate overall precision
- Weighted precision: same as macro but in this case, it is weighted average
depending on the number of items in each class.<br>

see how macro-averaged precision is implemented.<br>
``` python
import numpy as np
def macro_precision(y_true, y_pred):
"""
Function to calculate macro averaged precision
:param y_true: list of true values
:param y_pred: list of predicted values
:return: macro precision score
"""
# find the number of classes by taking
# length of unique values in true list
num_classes = len(np.unique(y_true))
# initialize precision to 0
precision = 0
# loop over all classes
for class_ in range(num_classes):
# all classes except current are considered negative
temp_true = [1 if p == class_ else 0 for p in y_true]
temp_pred = [1 if p == class_ else 0 for p in y_pred]
# calculate true positive for current class
tp = true_positive(temp_true, temp_pred)
# calculate false positive for current class
fp = false_positive(temp_true, temp_pred)
# calculate precision for current class
temp_precision = tp / (tp + fp)
# keep adding precision for all classes
precision += temp_precision
# calculate and return average precision over all classes
precision /= num_classes
return precision

```
Similarly, we have micro-averaged precision score.<br>

``` python
import numpy as np
def micro_precision(y_true, y_pred):
"""
Function to calculate micro averaged precision
:param y_true: list of true values
:param y_pred: list of predicted values
:return: micro precision score
"""
# find the number of classes by taking
# length of unique values in true list
num_classes = len(np.unique(y_true))
# initialize tp and fp to 0
tp = 0
fp = 0
# loop over all classes
for class_ in range(num_classes):
# all classes except current are considered negative
temp_true = [1 if p == class_ else 0 for p in y_true]
temp_pred = [1 if p == class_ else 0 for p in y_pred]
# calculate true positive for current class
# and update overall tp
tp += true_positive(temp_true, temp_pred)
# calculate false positive for current class
# and update overall tp
fp += false_positive(temp_true, temp_pred)
# calculate and return overall precision
precision = tp / (tp + fp)
return precision
```
let’s look at the implementation of weighted precision:
``` python
from collections import Counter
import numpy as np
def weighted_precision(y_true, y_pred):
"""
Function to calculate weighted averaged precision
:param y_true: list of true values
:param y_pred: list of predicted values
:return: weighted precision score
"""
# find the number of classes by taking
# length of unique values in true list
num_classes = len(np.unique(y_true))
# create class:sample count dictionary
# it looks something like this:
# {0: 20, 1:15, 2:21}
class_counts = Counter(y_true)
# initialize precision to 0
precision = 0
# loop over all classes
for class_ in range(num_classes):
# all classes except current are considered negative
temp_true = [1 if p == class_ else 0 for p in y_true]
temp_pred = [1 if p == class_ else 0 for p in y_pred]
# calculate tp and fp for class
tp = true_positive(temp_true, temp_pred)
fp = false_positive(temp_true, temp_pred)
# calculate precision of class
temp_precision = tp / (tp + fp)
# multiply precision with count of samples in class
weighted_precision = class_counts[class_] * temp_precision
# add to overall precision
precision += weighted_precision
# calculate overall precision by dividing by
# total number of samples
overall_precision = precision / len(y_true)
return overall_precision
```
implementation of weighted_f1:

``` python
    from collections import Counter
import numpy as np
def weighted_f1(y_true, y_pred):
"""
Function to calculate weighted f1 score
:param y_true: list of true values
:param y_proba: list of predicted values
:return: weighted f1 score
"""
# find the number of classes by taking
# length of unique values in true list
num_classes = len(np.unique(y_true))
# create class:sample count dictionary
# it looks something like this:
# {0: 20, 1:15, 2:21}
class_counts = Counter(y_true)
# initialize f1 to 0
f1 = 0
# loop over all classes
for class_ in range(num_classes):
# all classes except current are considered negative
temp_true = [1 if p == class_ else 0 for p in y_true]
temp_pred = [1 if p == class_ else 0 for p in y_pred]
# calculate precision and recall for class
p = precision(temp_true, temp_pred)
r = recall(temp_true, temp_pred)
# calculate
if p + r !=
temp_f1
else:
temp_f1
f1 of class
0:
= 2 * p * r / (p + r)
= 0
# multiply f1 with count of samples in class
weighted_f1 = class_counts[class_] * temp_f1
# add to f1 precision
f1 += weighted_f1
# calculate overall F1 by dividing by
# total number of samples
overall_f1 = f1 / len(y_true)
return overall_f1   
```
### multi-class problems
Thus, we have precision, recall and F1 implemented for multi-class problems. You
can similarly convert AUC and log loss to multi-class formats too. This format of
conversion is known as one-vs-all. I’m not going to implement them here as the
implementation is quite similar to what we have already discussed.<br>
In binary or multi-class classification, it is also quite popular to take a look at
confusion matrix. Don’t be confused; it’s quite easy. A confusion matrix is nothing
but a table of TP, FP, TN and FN. Using the confusion matrix, you can quickly see
how many samples were misclassified and how many were classified correctly.
One might argue that the confusion matrix should be covered quite early in this
chapter, but I chose not to do it. If you understand TP, FP, TN, FN, precision, recall
and AUC, it becomes quite easy to understand and interpret confusion matrix.<br>
We can also expand the binary confusion matrix to a multi-class confusion matrix.
How would that look like? If we have N classes, it will be a matrix of size NxN.
For every class, we calculate the total number of samples that went to the class in
concern and other classes.<br>
A perfect confusion matrix should only be filled diagonally from left to right.<br>
Confusion matrix gives an easy way to calculate different metrics that we have
discussed before. Scikit-learn offers an easy and straightforward way to generate a
confusion matrix.<br>
<br>
So, until now, we have tackled metrics for binary and multi-class classification.
Then comes another type of classification problem called multi-label
classification. In multi-label classification, each sample can have one or more
classes associated with it. One simple example of this type of problem would be a
task in which you are asked to predict different objects in a given image.<br>
The metrics for this type of classification problem are a bit different. Some suitable
and most common metrics are:<br>

- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)
- Log loss

Let’s start with **precision at k or P@k**. One must not confuse this precision with
the precision discussed earlier. If you have a list of original classes for a given
sample and list of predicted classes for the same, precision is defined as the number
of hits in the predicted list considering only top-k predictions, divided by k.
If that’s confusing, it will become apparent with python code.<br>
```python
def pk(y_true, y_pred, k):
"""
This function calculates precision at k
for a single sample
:param y_true: list of values, actual classes
:param y_pred: list of values, predicted classes
:param k: the value for k
:return: precision at a given value k
"""
# if k is 0, return 0. we should never have this
# as k is always >= 1
if k == 0:
return 0
# we are interested only in top-k predictions
y_pred = y_pred[:k]
# convert predictions to set
pred_set = set(y_pred)
# convert actual values to set
true_set = set(y_true)
# find common values
common_values = pred_set.intersection(true_set)
# return length of common values over k
return len(common_values) / len(y_pred[:k])
```

With code, everything becomes much easier to understand.
Now, we have average precision at k or AP@k. AP@k is calculated using P@k.
For example, if we have to calculate AP@3, we calculate P@1, P@2 and P@3 and
then divide the sum by 3.<br>
Let’s see its implementation.<br>
```python
def apk(y_true, y_pred, k):
"""
This function calculates average precision at k
for a single sample
:param y_true: list of values, actual classes
:param y_pred: list of values, predicted classes
:return: average precision at a given value k
"""
# initialize p@k list of values
pk_values = []
# loop over all k. from 1 to k + 1
for i in range(1, k + 1):
# calculate p@i and append to list
pk_values.append(pk(y_true, y_pred, i))
# if we have no values in the list, return 0
if len(pk_values) == 0:
return 0
# else, we return the sum of list over length of list
return sum(pk_values) / len(pk_values)
```

Please note that I have omitted many values from the output, but you get the point.
So, this is how we can calculate AP@k which is per sample. In machine learning,
we are interested in all samples, and that’s why we have mean average precision
at k or MAP@k. MAP@k is just an average of AP@k and can be calculated easily
by the following python code.<br>
``` python
def mapk(y_true, y_pred, k):
"""
This function calculates mean avg precision at k
for a single sample
:param y_true: list of values, actual classes
:param y_pred: list of values, predicted classes
:return: mean avg precision at a given value k
"""
# initialize empty list for apk values
apk_values = []
# loop over all samples
for i in range(len(y_true)):
# store apk values for every sample
apk_values.append(apk(y_true[i], y_pred[i], k=k)
)
# return mean of apk values list
return sum(apk_values) / len(apk_values)

```
P@k, AP@k and MAP@k all range from 0 to 1 with 1 being the best.<br>
Now, we come to log loss for multi-label classification. This is quite easy. You
can convert the targets to binary format and then use a log loss for each column. In
the end, you can take the average of log loss in each column. This is also known as
mean column-wise log loss. Of course, there are other ways you can implement
this, and you should explore it as you come across it.<br>

### Regression
now we can move to regression metrics.<br>
The most common metric in regression is error. Error is simple and very easy to
understand.<br>
                    Error = True Value – Predicted Value<br>
Absolute error is just absolute of the above.<br>
                    Absolute Error = Abs ( True Value – Predicted Value )<br>

Then we have mean absolute error (MAE). It’s just mean of all absolute errors.

``` python
import numpy as np
def mean_absolute_error(y_true, y_pred):
"""
This function calculates mae
:param y_true: list of real numbers, true values
:param y_pred: list of real numbers, predicted values
:return: mean absolute error
"""
# initialize error at 0
error = 0
# loop over all samples in the true and predicted list
for yt, yp in zip(y_true, y_pred):
# calculate absolute error
# and add to error
error += np.abs(yt - yp)
# return mean error
return error / len(y_true)
```

Similarly, we have squared error and **mean squared error (MSE)**.
                Squared Error = ( True Value – Predicted Value )^2

And mean squared error (MSE) can be implemented as follows.

``` python
def mean_squared_error(y_true, y_pred):
"""
This function calculates mse
:param y_true: list of real numbers, true values
:param y_pred: list of real numbers, predicted values
:return: mean squared error
"""
# initialize error at 0
error = 0
# loop over all samples in the true and predicted list
for yt, yp in zip(y_true, y_pred):
# calculate squared error
# and add to error
error += (yt - yp) ** 2
# return mean error
return error / len(y_true)
```
MSE and RMSE (root mean squared error) are the most popular metrics used in
evaluating regression models.<br>
                                RMSE = SQRT ( MSE )<br>


And an absolute version of the same (and more common version) is known as mean
absolute percentage error or MAPE.
```python
import numpy as np
def mean_abs_percentage_error(y_true, y_pred):
"""
This function calculates MAPE
:param y_true: list of real numbers, true values
:param y_pred: list of real numbers, predicted values
:return: mean absolute percentage error
"""
# initialize error at 0
error = 0
# loop over all samples in true and predicted list
for yt, yp in zip(y_true, y_pred):
# calculate percentage error
# and add to error
error += np.abs(yt - yp) / yt
# return mean percentage error
return error / len(y_true)
```
The best thing about regression is that there are only a few most popular metrics
that can be applied to almost every regression problem. And it is much easier to
understand when we compare it to classification metrics.
Let’s talk about another regression metric known as **R^2 (R-squared)**, also known
as the **coefficient of determination**.<br>

In simple words, R-squared says how good your model fits the data. R-squared
closer to 1.0 says that the model fits the data quite well, whereas closer 0 means
that model isn’t that good. R-squared can also be negative when the model just
makes absurd predictions.<br>

The formula for R-squared is shown in figure 10, but as always a python
implementation makes things more clear.

$$
R^2 = 1 - \frac{\sum_{i=1}^{N} \left(y_{ti} - y_{pi}\right)^2}{\sum_{i=1}^{N} \left(y_{ti} - y_{tmean}\right)^2}
$$

```python
import numpy as np
def r2(y_true, y_pred):
"""
This function calculates r-squared score
:param y_true: list of real numbers, true values
:param y_pred: list of real numbers, predicted values
:return: r2 score
"""
# calculate the mean value of true values
mean_true_value = np.mean(y_true)
# initialize numerator with 0
numerator = 0
# initialize denominator with 0
denominator = 0
# loop over all true and predicted values
for yt, yp in zip(y_true, y_pred):
# update numerator
numerator += (yt - yp) ** 2
# update denominator
denominator += (yt - mean_true_value) ** 2
# calculate the ratio
ratio = numerator / denominator
# return 1 - ratio
return 1 – ratio
```

there are some advanced metrics.<br>

One of them which is quite widely used is **quadratic weighted kappa**, also known
as **QWK**. It is also known as **Cohen’s kappa**. QWK measures the “agreement”
between two “ratings”. The ratings can be any real numbers in 0 to N. And
predictions are also in the same range. An agreement can be defined as how close
these ratings are to each other. So, it’s suitable for a classification problem with N
different categories/classes. If the agreement is high, the score is closer towards 1.0.
In the case of low agreement, the score is close to 0. Cohen’s kappa has a good
implementation in scikit-learn.<br>
<br>
An important metric is Matthew’s Correlation Coefficient (MCC). MCC ranges
from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction, and 0 is random
prediction. The formula for MCC is quite simple.<br>

$$
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP) \times (FN + TN) \times (FP + TN) \times (TP + FN)}}
$$
We see that MCC takes into consideration TP, FP, TN and FN and thus can be used
for problems where classes are skewed.<br>

One thing to keep in mind is that to evaluate un-supervised methods, for example,
some kind of clustering, it’s better to create or manually label the test set and keep
it separate from everything that is going on in your modelling part. When you are
done with clustering, you can evaluate the performance on the test set simply by
using any of the supervised learning metrics.<br>


Text simplification involves altering a text to make it easier to read or under-
stand without changing its original meaning. The objective is to create content
that is more accessible for specific users or systems. discover criteria based on the dataset for simple and complex sentences, and create ML model for categorizing simple and complex sentences. Also, create new dataset for simple and complex sentences based on the criteria that are found in the first dataset using GPT4o and check the ML models for categorizing simple and complex sentences. and Also use the same criteria for Italian dataset and check the ML models for categorizing simple and complex sentences.


## Arranging machine learning projects
Let’s look at the structure of the files first of all. For any project that you are doing, create a new folder. For this example, I am calling the project “project”.<br>

The inside of the project folder should look something like the following.<br>
├── input
│
    ├── train.csv
│
    └── test.csv
├── src
│
    ├── create_folds.py
│
    ├── train.py
│
    ├── inference.py
│
    ├── models.py
│
    ├── config.py
│
    └── model_dispatcher.py
├── models
│
    ├── model_rf.bin
│
    └── model_et.bin
├── notebooks
│
    ├── exploration.ipynb
│
    └── check_data.ipynb
├── README.md
└── LICENSE



Let’s see what these folders and file are about.<br>
input/: This folder consists of all the input files and data for your machine learning
project. If you are working on NLP projects, you can keep your embeddings here.
If you are working on image projects, all images go to a subfolder inside this folder.<br>
src/: We will keep all the python scripts associated with the project here. If I talk
about a python script, i.e. any *.py file, it is stored in the src folder.<br>
models/: This folder keeps all the trained models.<br>
notebooks/: All jupyter notebooks (i.e. any *.ipynb file) are stored in the notebooks
folder.<br>
README.md: This is a markdown file where you can describe your project and
write instructions on how to train the model or to serve this in a production
environment.<br>
LICENSE: This is a simple text file that consists of a license for the project, such as
MIT, Apache, etc. Going into details of the licenses is beyond the scope of this
book.<br>
Let’s assume you are building a model to classify MNIST dataset, we will be using the CSV format of the dataset.<br>
In this format of the dataset, each row of the CSV consists of the label of the image
and 784 pixel values ranging from 0 to 255. The dataset consists of 60000 images
in this format. We can use pandas to read this data format easily.<br>
We don’t need much more exploration for this dataset. We already know what we
have, and there is no need to make plots on different pixel values. We can thus use
accuracy/F1 as metrics. This is the first step when approaching a machine learning
problem: **decide the metric!**.<br>
Please note that the training CSV file is located in the input/ folder and is called
mnist_train.csv.Please note that the training CSV file is located in the input/ folder and is called
mnist_train.csv.<br>
The first script that one should create is **create_folds.py**.
This will create a new file in the input/ folder called mnist_train_folds.csv, and it’s
the same as mnist_train.csv. The only differences are that this CSV is shuffled and
has a new column called kfold.<br>
Once we have decided what kind of evaluation metric we want to use and have
created the folds, we are good to go with creating a basic model. This is done in
train.py.   
```python
# src/train.py
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
def run(fold):
# read the training data with folds
df = pd.read_csv("../input/mnist_train_folds.csv")
# training data is where kfold is not equal to provided fold
# also, note that we reset the index
df_train = df[df.kfold != fold].reset_index(drop=True)
# validation data is where kfold is equal to provided fold
df_valid = df[df.kfold == fold].reset_index(drop=True)
# drop the label column from dataframe and convert it to
# a numpy array by using .values.
# target is label column in the dataframe
x_train = df_train.drop("label", axis=1).values
y_train = df_train.label.values
# similarly, for validation, we have
x_valid = df_valid.drop("label", axis=1).values
y_valid = df_valid.label.values
# initialize simple decision tree classifier from sklearn
clf = tree.DecisionTreeClassifier()
# fit the model on training data
clf.fit(x_train, y_train)
# create predictions for validation samples
preds = clf.predict(x_valid)
# calculate & print accuracy
accuracy = metrics.accuracy_score(y_valid, preds)
print(f"Fold={fold}, Accuracy={accuracy}")
# save the model
joblib.dump(clf, f"../models/dt_{fold}.bin")
if __name__ == "__main__":
run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
```
When you look at the training script, you will see that there are still a few more
things that are hardcoded, for example, the fold numbers, the training file and the
output folder.<br>

We can thus create a config file with all this information: **config.py**.
```python
# config.py
TRAINING_FILE = "../input/mnist_train_folds.csv"
MODEL_OUTPUT = "../models/"
```
And we make some changes to our training script too. The training file utilizes the
config file now. Thus making it easier to change data or the model output.
```python
# train.py
import os
import config
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
def run(fold):
# read the training data with folds
df = pd.read_csv(config.TRAINING_FILE)
# training data is where kfold is not equal to provided fold
# also, note that we reset the index
df_train = df[df.kfold != fold].reset_index(drop=True)
# validation data is where kfold is equal to provided fold
df_valid = df[df.kfold == fold].reset_index(drop=True)
# drop the label column from dataframe and convert it to
# a numpy array by using .values.
# target is label column in the dataframe
x_train = df_train.drop("label", axis=1).values
y_train = df_train.label.values
# similarly, for validation, we have
x_valid = df_valid.drop("label", axis=1).values
y_valid = df_valid.label.values
# initialize simple decision tree classifier from sklearn
clf = tree.DecisionTreeClassifier()
# fir the model on training data
clf.fit(x_train, y_train)
# create predictions for validation samples
preds = clf.predict(x_valid)
# calculate & print accuracy
accuracy = metrics.accuracy_score(y_valid, preds)
print(f"Fold={fold}, Accuracy={accuracy}")
# save the model
joblib.dump(
clf,
os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
)
if __name__ == "__main__":
run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
```
Please note that I am not showing the difference between this training script and the
one before. Please take a careful look at both of them and find the differences
yourself. There aren’t many of them.<br>
There is still one more thing related to the training script that can be improved. As
you can see, we call the run function multiple times for every fold. Sometimes it’s
not advisable to run multiple folds in the same script as the memory consumption
may keep increasing, and your program may crash. To take care of this problem,
we can pass arguments to the training script. I like doing it using argparse.
```python
# train.py
import argparse
.
.
.
if __name__ == "__main__":
# initialize ArgumentParser class of argparse
parser = argparse.ArgumentParser()
# add the different arguments you need and their type
# currently, we only need fold
parser.add_argument(
"--fold",
type=int
)
# read the arguments from the command line
args = parser.parse_args()
# run the fold specified by command line arguments
run(fold=args.fold)
```
If you see carefully, our fold 0 score was a bit different before. This is because of
the randomness in the model.<br>
We have made quite some progress now, but if we look at our training script, we
still are limited by a few things, for example, the model. The model is hardcoded in
the training script, and the only way to change it is to modify the script. So, we will
create a new python script called model_dispatcher.py. model_dispatcher.py, as the
name suggests, will dispatch our models to our training script.<br>
```python
# model_dispatcher.py
from sklearn import tree
models = {
"decision_tree_gini": tree.DecisionTreeClassifier(
criterion="gini"
),
"decision_tree_entropy": tree.DecisionTreeClassifier(
criterion="entropy"
),
}
```
model_dispatcher.py imports tree from scikit-learn and defines a dictionary with
keys that are names of the models and values are the models themselves. Here, we
define two different decision trees, one with gini criterion and one with entropy. To
use model_dispatcher.py, we need to make a few changes to our training script.
```python
# train.py
import argparse
import os
import joblib
import pandas as pd
from sklearn import metrics
import config
import model_dispatcher
def run(fold, model):
# read the training data with folds
df = pd.read_csv(config.TRAINING_FILE)
# training data is where kfold is not equal to provided fold
# also, note that we reset the index
df_train = df[df.kfold != fold].reset_index(drop=True)
# validation data is where kfold is equal to provided fold
df_valid = df[df.kfold == fold].reset_index(drop=True)
# drop the label column from dataframe and convert it to
# a numpy array by using .values.
# target is label column in the dataframe
x_train = df_train.drop("label", axis=1).values
y_train = df_train.label.values
# similarly, for validation, we have
x_valid = df_valid.drop("label", axis=1).values
y_valid = df_valid.label.values
# fetch the model from model_dispatcher
clf = model_dispatcher.models[model]
# fir the model on training data
clf.fit(x_train, y_train)
# create predictions for validation samples
preds = clf.predict(x_valid)
# calculate & print accuracy
accuracy = metrics.accuracy_score(y_valid, preds)
print(f"Fold={fold}, Accuracy={accuracy}")
# save the model
joblib.dump(
clf,
os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
)
if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument(
"--fold",
type=int
)
parser.add_argument(
"--model",
type=str
)
args = parser.parse_args()
run(
fold=args.fold,
model=args.model
)
```

There are a few major changes to train.py:
• import model_dispatcher
• add --model argument to ArgumentParser
• add model argument to run() function
• use the dispatcher to fetch the model given the name<br>
Now, if you add a new model, all you have to do is make changes to
model_dispatcher.py. Let’s try adding random forest and see what happens to our
accuracy.<br>
```python
# model_dispatcher.py
from sklearn import ensemble
from sklearn import tree
models = {
"decision_tree_gini": tree.DecisionTreeClassifier(
criterion="gini"
),
"decision_tree_entropy": tree.DecisionTreeClassifier(
criterion="entropy"
),
"rf": ensemble.RandomForestClassifier(),
}
```
❯ python train.py --fold 0 --model rf
Fold=0, Accuracy=0.9670833333333333
<br>
And the scores look like the following.<br>
Fold=0, Accuracy=0.9674166666666667<br>
Fold=1, Accuracy=0.9698333333333333<br>
Fold=2, Accuracy=0.96575<br>
Fold=3, Accuracy=0.9684166666666667<br>
Fold=4, Accuracy=0.9666666666666667<br>


## Approaching categorical variables
Many people struggle a lot with the handling of categorical variables, and thus this
deserves a full chapter. In this chapter, I will talk about different types of categorical
data and how to approach a problem with categorical variables.<br>
What are categorical variables?<br>
Categorical variables/features are any feature type can be classified into two major
types:<br>
- Nominal
- Ordinal<br>
**Nominal variables** are variables that have two or more categories which do not
have any kind of order associated with them. For example, if gender is classified
into two groups, i.e. male and female, it can be considered as a nominal variable.<br>
**Ordinal variables**, on the other hand, have “levels” or categories with a particular
order associated with them. For example, an ordinal categorical variable can be a
feature with three different levels: low, medium and high. Order is important.<br>
As far as definitions are concerned, we can also categorize categorical variables as
**binary**, i.e., a categorical variable with only two categories. Some even talk about
a type called **cyclic** for categorical variables. Cyclic variables are present in
“cycles” for example, days in a week: Sunday, Monday, Tuesday, Wednesday,
Thursday, Friday and Saturday. After Saturday, we have Sunday again. This is a
cycle. Another example would be hours in a day if we consider them to be categories.<br>

Before we start, we need a dataset to work with (as always). One of the best free
datasets to understand categorical variables is cat-in-the-dat from Categorical
Features Encoding Challenge from Kaggle. There were two challenges, and we will
be using the data from the second challenge as it had more variables and was more
difficult than its previous version.<br>
The dataset consists of all kinds of categorical variables:
- Nominal
- Ordinal
- Cyclical
- Binary<br>

It is a binary classification problem.<br>

The target is not very important for us to learn categorical variables, but in the end,
we will be building an end-to-end model so let’s take a look at the target distribution
in figure 2. We see that the target is **skewed** and thus the best metric for this binary
classification problem would be Area Under the ROC Curve (AUC). We can use
precision and recall too, but AUC combines these two metrics. Thus, we will be
using AUC to evaluate the model that we build on this dataset.<br>

Overall, there are:<br>
- Five binary variables
- Ten nominal variables
- Six ordinal variables
- Two cyclic variables
- And a target variable<br>

We have to know that computers do not understand text data and thus, we need to
convert these categories to numbers. A simple way of doing this would be to create
a dictionary that maps these values to numbers starting from 0 to N-1, where N is
the total number of categories in a given feature.<br>
```python
mapping = {
"Freezing": 0,
"Warm": 1,
"Cold": 2,
"Boiling Hot": 3,
"Hot": 4,
"Lava Hot": 5
}
```
This type of encoding of categorical variables is known as Label Encoding, i.e.,
we are encoding every category as a numerical label.We can do the same by using LabelEncoder from scikit-learn.<br>

We can use this directly in many tree-based models:<br>

- Decision trees
- Random forest
- Extra Trees<br>
Or any kind of boosted trees model<br>
o XGBoost
o GBM
o LightGBM<br>




This type of encoding cannot be used in linear models, support vector machines or
neural networks as they expect data to be normalized (or standardized).<br>
For these types of models, we can binarize the data.<br>
Freezing --> 0 --> 0 0 0<br>
Warm --> 1 --> 0 0 1<br>
Cold --> 2 --> 0 1 0<br>
Boiling Hot --> 3 --> 0 1 1<br>
Hot --> 4 --> 1 0 0<br>
Lava Hot --> 5 --> 1 0 1<br>
This is just converting the categories to numbers and then converting them to their
binary representation. We are thus splitting one feature into three (in this case)
features (or columns). If we have more categories, we might end up splitting into a
lot more columns.<br>

It becomes easy to store lots of binarized variables like this if we store them in a
sparse format. A sparse format is nothing but a representation or way of storing
data in memory in which you do not store all the values but only the values that
matter. In the case of binary variables described above, all that matters is where we
have ones (1s).<br>

Even though the sparse representation of binarized features takes much less
memory than its dense representation, there is another transformation for
categorical variables that takes even less memory. This is known as **One Hot Encoding**.<br>
One hot encoding is a binary encoding too in the sense that there are only two
values, 0s and 1s. However, it must be noted that it’s not a binary representation.<br>

These three methods are the most important ways to handle categorical variables.
There are, however, many other different methods you can use to handle categorical
variables. An example of one such method is about converting categorical variables
to numerical variables.<br>

So which categories should we combine? Well, there isn't an easy answer to that. It
depends on your data and the types of features. Some domain knowledge might be
useful for creating features like this. But if you don’t have concerns about memory
and CPU usage, you can go for a greedy approach where you can create many such
combinations and then use a model to decide which features are useful and keep
them.<br>
Whenever you get categorical variables, follow these simple steps:
- fill the NaN values (this is very important!)
- convert them to integers by applying label encoding using LabelEncoder
of scikit-learn or by using a mapping dictionary. If you didn’t fill up NaN
values with something, you might have to take care of them in this step.<br>
- create one-hot encoding. Yes, you can skip binarization!
- go for modelling! I mean the machine learning one. Not on the ramp.<br>
Handling NaN data in categorical features is quite essential else you can get the
infamous error from scikit-learn’s LabelEncoder:<br>
ValueError: y contains previously unseen labels: [nan, nan, nan, nan,
nan, nan, nan, nan]<br>
This simply means that when you are transforming the test data, you have NaN
values in it. It’s because you forgot to handle them during training. One simple way
to handle NaN values would be to drop them. Well, it’s simple but not ideal. NaN
values may have a lot of information in them, and you will lose it if you just drop
these values. There might also be many situations where most of your data has NaN
values, and thus, you cannot drop rows/samples with NaN values. Another way of
handling NaN values is to treat them as a completely new category. This is the most
preferred way of handling NaN values. And can be achieved in a very simple
manner if you are using pandas.<br>
in this dataset there were 18075 NaN values in this column that we didn’t even consider
using previously. With the addition of this new category, the total number of
categories have now increased from 6 to 7. This is okay because now when we build
our models, we will also consider NaN. The more relevant information we have,
the better the model is.<br>

Let’s assume that ord_2 did not have any NaN values. We see that all categories in
this column have a significant count. There are no “rare” categories; i.e. the
categories which appear only a small percentage of the total number of samples.
Now, let’s assume that you have deployed this model which uses this column in
production and when the model or the project is live, you get a category in ord_2
column that is not present in train. You model pipeline, in this case, will throw an
error and there is nothing that you can do about it. If this happens, then probably
something is wrong with your pipeline in production. If this is expected, then you
must modify your model pipeline and include a new category to these six categories.<br>
This new category is known as the “rare” category. A **rare category** is a category
which is not seen very often and can include many different categories. You can
also try to “predict” the unknown category by using a nearest neighbour model.
Remember, if you predict this category, it will become one of the categories from
the training data.<br>

When we have a dataset to have such a rare category, we can build a simple model
that’s trained on all features except rare category. Thus, you will be creating a model that
predicts rare category when it’s not known or not available in training. I can’t say if this kind
of model is going to give you an excellent performance but might be able to handle
those missing values in test set or live data and one can’t say without trying just like
everything else when it comes to machine learning.<br>
If you have a fixed test set, you can add your test data to training to know about the
categories in a given feature. This is very similar to semi-supervised learning in
which you use data which is not available for training to improve your model. This
will also take care of rare values that appear very less number of times in training
data but are in abundance in test data. Your model will be more robust.<br>
Many people think that this idea overfits. It may or may not overfit. There is a
simple fix for that. If you design your cross-validation in such a way that it
replicates the prediction process when you run your model on test data, then it’s
never going to overfit. It means that the first step should be the separation of folds,
and in each fold, you should apply the same pre-processing that you want to apply
to test data. Suppose you want to concatenate training and test data, then in each
fold you must concatenate training and validation data and also make sure that your
validation dataset replicates the test set. In this specific case, you must design your
validation sets in such a way that it has categories which are “unseen” in the training
set.<br>

```python
import pandas as pd
from sklearn import preprocessing
# read training data
train = pd.read_csv("../input/cat_train.csv")
#read test data
test = pd.read_csv("../input/cat_test.csv")
# create a fake target column for test data
# since this column doesn't exist
test.loc[:, "target"] = -1
# concatenate both training and test data
data = pd.concat([train, test]).reset_index(drop=True)
# make a list of features we are interested in
# id and target is something we should not encode
features = [x for x in train.columns if x not in ["id", "target"]]
# loop over the features list
for feat in features:
# create a new instance of LabelEncoder for each feature
lbl_enc = preprocessing.LabelEncoder()
# note the trick here
# since its categorical data, we fillna with a string
# and we convert all the data to string type
# so, no matter its int or float, its converted to string
# int/float but categorical!!!
temp_col = data[feat].fillna("NONE").astype(str).values
# we can use fit_transform here as we do not
# have any extra test data that we need to
# transform on separately
data.loc[:, feat] = lbl_enc.fit_transform(temp_col)
# split the training and test data again
train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)
```
This trick works when you have a problem where you already have the test dataset.
It must be noted that this trick will not work in a live setting. For example, let’s say
you are in a company that builds a real-time bidding solution (RTB). RTB systems
bid on every user they see online to buy ad space. The features that can be used for
such a model may include pages viewed in a website. Let’s assume that features are
the last five categories/pages visited by the user. In this case, if the website
introduces new categories, we will no longer be able to predict accurately. Our
model, in this case, will fail. A situation like this can be avoided by using an
**unknown** category.<br>


In our cat-in-the-dat dataset, we already have unknowns in ord_2 column.<br>
We can treat **NONE** as unknown. So, if during live testing, we get new categories
that we have not seen before, we will mark them as **NONE**.<br>
This is very similar to natural language processing problems. We always build a
model based on a fixed vocabulary. Increasing the size of the vocabulary increases
the size of the model. Transformer models like BERT are trained on ~30000 words
(for English). So, when we have a new word coming in, we mark it as UNK
(unknown).<br>
So, you can either assume that your test data will have the same categories as
training or you can introduce a rare or unknown category to training to take care of
new categories in test data.<br>
We can now define our criteria for calling a value “rare”. Let’s say the requirement
for a value being rare in this column is a count of less than 2000. So, it seems, J and
L can be marked as rare values. With pandas, it is quite easy to replace categories
based on count threshold.<br>
We say that wherever the value count for a certain category is less than 2000,
replace it with rare. So, now, when it comes to test data, all the new, unseen
categories will be mapped to “RARE”, and all missing values will be mapped to
“NONE”.<br>
This approach will also ensure that the model works in a live setting, even if you
have new categories.<br>
Now we have everything we need to approach any kind of problem with categorical
variables in it. Let’s try building our first model and try to improve its performance
in a step-wise manner.<br>
Before going to any kind of model building, it’s essential to take care of cross-
validation. We have already seen the label/target distribution, and we know that it
is a binary classification problem with skewed targets. Thus, we will be using
StratifiedKFold to split the data here.<br>
```python
# create_folds.py
# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection
if __name__ == "__main__":
# Read training data
df = pd.read_csv("../input/cat_train.csv")
# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# fetch labels
y = df.target.values
# initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)
# fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
df.loc[v_, 'kfold'] = f
# save the new csv with kfold column
df.to_csv("../input/cat_train_folds.csv", index=False)
```
We can now check our new folds csv to see the number of samples per fold:
```python
In [X]: import pandas as pd
In [X]: df = pd.read_csv("../input/cat_train_folds.csv")
In [X]: df.kfold.value_counts()
Out[X]:
4
120000
3
120000
2
120000
1
120000
0
120000
Name: kfold, dtype: int64
```

All folds have 120000 samples. This is expected as training data has 600000
samples, and we made five folds. So far, so good.<br>
One of the simplest models we can build is by one-hot encoding all the data and
using logistic regression.<br>
```python
# ohe_logres.py
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
def run(fold):
# load the full training data with folds
df = pd.read_csv("../input/cat_train_folds.csv")
# all columns are features except id, target and kfold columns
features = [
f for f in df.columns if f not in ("id", "target", "kfold")
]
# fill all NaN values with NONE
# note that I am converting all columns to "strings"
# it doesn’t matter because all are categories
for col in features:
df.loc[:, col] = df[col].astype(str).fillna("NONE")
# get training data using folds
df_train = df[df.kfold != fold].reset_index(drop=True)
# get validation data using folds
df_valid = df[df.kfold == fold].reset_index(drop=True)
# initialize OneHotEncoder from scikit-learn
ohe = preprocessing.OneHotEncoder()
# fit ohe on training + validation features
full_data = pd.concat(
[df_train[features], df_valid[features]],
axis=0
)
ohe.fit(full_data[features])
# transform training data
x_train = ohe.transform(df_train[features])
# transform validation data
x_valid = ohe.transform(df_valid[features])
# initialize Logistic Regression model
model = linear_model.LogisticRegression()
# fit model on training data (ohe)
model.fit(x_train, df_train.target.values)
# predict on validation data
# we need the probability values as we are calculating AUC
# we will use the probability of 1s
valid_preds = model.predict_proba(x_valid)[:, 1]
# get roc auc score
auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
# print auc
print(auc)
if __name__ == "__main__":
# run function for fold = 0
# we can just replace this number and
# run this for any fold
run(0)
```
We have created a function that splits data into training and validation, given a fold
number, handles NaN values, applies one-hot encoding on all the data and trains a
simple Logistic Regression model.<br>

There are a few warnings. It seems logistic regression did not converge for the max
number of iterations. We didn’t play with the parameters, so that is fine. We see
that AUC is ~ 0.785.<br>
Please note that we are not making a lot of changes and that’s why I have shown
only some lines of the code; some of which have changes.<br>
This gives:<br>
```python
❯ python -W ignore ohe_logres.py
Fold = 0, AUC = 0.7847865042255127
Fold = 1, AUC = 0.7853553605899214
Fold = 2, AUC = 0.7879321942914885
Fold = 3, AUC = 0.7870315929550808
Fold = 4, AUC = 0.7864668243125608
```
We see that AUC scores are quite stable across all folds. The average AUC is
0.78631449527. Quite good for our first model!<br>
Many people will start this kind of problem with a tree-based model, such as
random forest. For applying random forest in this dataset, instead of one-hot
encoding, we can use label encoding and convert every feature in every column to
an integer as discussed previously.<br>
And this is a reason why we should always start with simple models first. A fan of
random forest would begin with it here and will ignore logistic regression model
thinking it’s a very simple model that cannot bring any value better than random
forest. That kind of person will make a huge mistake. In our implementation of
random forest, the folds take a much longer time to complete compared to logistic
regression. So, we are not only losing on AUC but also taking much longer to
complete the training. Please note that inference is also time-consuming with
random forest and it also takes much larger space.<br>
If we want, we can also try to run random forest on sparse one-hot encoded data,
but that is going to take a lot of time. We can also try reducing the sparse one-hot
encoded matrices using singular value decomposition. This is a very common
method of extracting topics in natural language processing.<br>
Please note that we do not need to normalize data when we use tree-based models.<br>

One more way of feature engineering from categorical features is to use target
encoding. However, you have to be very careful here as this might overfit your
model. Target encoding is a technique in which you map each category in a given
feature to its mean target value, but this must always be done in a cross-validated
manner. It means that the first thing you do is create the folds, and then use those
folds to create target encoding features for different columns of the data in the same
way you fit and predict the model on folds. So, if you have created 5 folds, you
have to create target encoding 5 times such that in the end, you have encoding for
variables in each fold which are not derived from the same fold. And then when
you fit your model, you must use the same folds again. Target encoding for unseen
test data can be derived from the full training data or can be an average of all the 5
folds.you must be very careful when using target encoding as it is too prone to overfitting. When we use target
encoding, it’s better to use some kind of smoothing or adding noise in the encoded values. Scikit-learn has contrib repository which has target encoding with smoothing, or you can create your own smoothing. Smoothing introduces some
kind of regularization that helps with not overfitting the model. It’s not very difficult.<br>


So, let’s take a look at a technique known as entity embedding. In entity embeddings, the
categories are represented as vectors. We represent categories by vectors in both
binarization and one hot encoding approaches. But what if we have tens of
thousands of categories. This will create huge matrices and will take a long time for
us to train complicated models. We can thus represent them by vectors with float
values instead.<br>
The idea is super simple. You have an embedding layer for each categorical feature.
So, every category in a column can now be mapped to an embedding (like mapping
words to embeddings in natural language processing). You then reshape these
embeddings to their dimension to make them flat and then concatenate all the
flattened inputs embeddings. Then add a bunch of dense layers, an output layer and
you are done.<br>
For some reason, I find it super easy to do using TF/Keras. So, let’s see how it’s
implemented using TF/Keras.
```python
# entity_emebddings.py
import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
def create_model(data, catcols):
"""
This function returns a compiled tf.keras model
for entity embeddings
:param data: this is a pandas dataframe
:param catcols: list of categorical column names
:return: compiled tf.keras model
"""
# init list of inputs for embeddings
inputs = []
# init list of outputs for embeddings
outputs = []
# loop over all categorical columns
for c in catcols:
# find the number of unique values in the column
num_unique_values = int(data[c].nunique())
# simple dimension of embedding calculator
# min size is half of the number of unique values
# max size is 50. max size depends on the number of unique
# categories too. 50 is quite sufficient most of the times
# but if you have millions of unique values, you might need
# a larger dimension
embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
# simple keras input layer with size 1
inp = layers.Input(shape=(1,))
# add embedding layer to raw input
# embedding size is always 1 more than unique values in input
out = layers.Embedding(
num_unique_values + 1, embed_dim, name=c
)(inp)
# 1-d spatial dropout is the standard for emebedding layers
# you can use it in NLP tasks too
out = layers.SpatialDropout1D(0.3)(out)
# reshape the input to the dimension of embedding
# this becomes our output layer for current feature
out = layers.Reshape(target_shape=(embed_dim, ))(out)
# add input to input list
inputs.append(inp)
# add output to output list
outputs.append(out)
# concatenate all output layers
x = layers.Concatenate()(outputs)
#
#
#
#
#
#
x add a batchnorm layer.
from here, everything is up to you
you can try different architectures
this is the architecture I like to use
if you have numerical features, you should add
them here or in concatenate layer
= layers.BatchNormalization()(x)
#
#
x
x
x a bunch of dense layers with dropout.
start with 1 or two layers only
= layers.Dense(300, activation="relu")(x)
= layers.Dropout(0.3)(x)
= layers.BatchNormalization()(x)
x = layers.Dense(300, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
#
#
#
y
using softmax and treating it as a two class problem
you can also use sigmoid, then you need to use only one
output class
= layers.Dense(2, activation="softmax")(x)
# create final model
model = Model(inputs=inputs, outputs=y)
# compile the model
# we use adam and binary cross entropy.
# feel free to use something else and see how model behaves
model.compile(loss='binary_crossentropy', optimizer='adam')
return model
def run(fold):
# load the full training data with folds
df = pd.read_csv("../input/cat_train_folds.csv")
# all columns are features except id, target and kfold columns
features = [
f for f in df.columns if f not in ("id", "target", "kfold")
]
# fill all NaN values with NONE
# note that I am converting all columns to "strings"
# it doesnt matter because all are categories
for col in features:
df.loc[:, col] = df[col].astype(str).fillna("NONE")
# encode all features with label encoder individually
# in a live setting you need to save all label encoders
for feat in features:
lbl_enc = preprocessing.LabelEncoder()
df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)
# get training data using folds
df_train = df[df.kfold != fold].reset_index(drop=True)
# get validation data using folds
df_valid = df[df.kfold == fold].reset_index(drop=True)
# create tf.keras model
model = create_model(df, features)
# our features are lists of lists
xtrain = [
df_train[features].values[:, k] for k in range(len(features))
]
xvalid = [
df_valid[features].values[:, k] for k in range(len(features))
]
# fetch target columns
ytrain = df_train.target.values
yvalid = df_valid.target.values
# convert target columns to categories
# this is just binarization
ytrain_cat = utils.to_categorical(ytrain)
yvalid_cat = utils.to_categorical(yvalid)
# fit the model
model.fit(xtrain,
ytrain_cat,
validation_data=(xvalid, yvalid_cat),
verbose=1,
batch_size=1024,
epochs=3
)
# generate validation predictions
valid_preds = model.predict(xvalid)[:, 1]
# print roc auc score
print(metrics.roc_auc_score(yvalid, valid_preds))
# clear session to free up some GPU memory
K.clear_session()
if __name__ == "__main__":
run(0)
run(1)
run(2)
run(3)
run(4)
```
You will notice that this approach gives the best results and is also super-fast if you
have a GPU! This can also be improved further, and you don’t need to worry about
feature engineering as neural network handles it on its own. This is definitely worth
a try when dealing with a large dataset of categorical features. When embedding
size is the same as the number of unique categories, we have one-hot-encoding.<Br>




## Feature engineering

Feature engineering is one of the most crucial parts of building a good machine
learning model. If we have useful features, the model will perform better. There are
many situations where you can avoid large, complicated models and use simple
models with crucially engineered features. We must keep in mind that feature
engineering is something that is done in the best possible manner only when you
have some knowledge about the domain of the problem and depends a lot on the
data in concern. However, there are some general techniques that you can try to
create features from almost all kinds of numerical and categorical variables.
Feature engineering is not just about creating new features from data but also
includes different types of normalization and transformations.<Br>
our focus will be limited to numerical variables and a combination of numerical and categorical variables.<br>

Let’s start with the most simple but most widely used feature engineering
techniques. Let’s say that you are dealing with date and time data. So, we have a
pandas dataframe with a datetime type column. Using this column, we can create
features like:

- Year
- Week of year
- Month
- Day of week
- Weekend
- Hour
- And many more.<br>
And this can be done using pandas very easily.
```python
df.loc[:, 'year'] = df['datetime_column'].dt.year
df.loc[:, 'weekofyear'] = df['datetime_column'].dt.weekofyear
df.loc[:, 'month'] = df['datetime_column'].dt.month
df.loc[:, 'dayofweek'] = df['datetime_column'].dt.dayofweek
df.loc[:, 'weekend'] = (df.datetime_column.dt.weekday >=5).astype(int)
df.loc[:, 'hour'] = df['datetime_column'].dt.hour
```
So, we are creating a bunch of new columns using the datetime column. Let’s see
some of the sample features that can be created.
```python
import pandas as pd
# create a series of datetime with a frequency of 10 hours
s = pd.date_range('2020-01-06', '2020-01-10', freq='10H').to_series()
# create some features based on datetime
features = {
"dayofweek": s.dt.dayofweek.values,
"dayofyear": s.dt.dayofyear.values,
"hour": s.dt.hour.values,
"is_leap_year": s.dt.is_leap_year.values,
"quarter": s.dt.quarter.values,
"weekofyear": s.dt.weekofyear.values
}
```

This will generate a dictionary of features from a given series. You can apply this
to any datetime column in a pandas dataframe. These are some of the many date
time features that pandas offer. Date time features are critical when you are dealing
with time-series data, for example, predicting sales of a store but would like to use
a model like xgboost on aggregated features.<br>
we see that we have a date column, and we can easily extract features
like the year, month, quarter, etc. from that. Then we have a customer_id column
which has multiple entries, so a customer is seen many times (not visible in the
screenshot). And each date and customer id has three categorical and one numerical
feature attached to it. There are a bunch of features we can create from it:<br>
- What’s the month a customer is most active in
- What is the count of cat1, cat2, cat3 for a customer
- What is the count of cat1, cat2, cat3 for a customer for a given week of the year
- What is the mean of num1 for a given customer
- And so on.<br>
Using aggregates in pandas, it is quite easy to create features like these. Let’s see
how.
```python
def generate_features(df):
# create a bunch of features using the date column
df.loc[:, 'year'] = df['date'].dt.year
df.loc[:, 'weekofyear'] = df['date'].dt.weekofyear
df.loc[:, 'month'] = df['date'].dt.month
df.loc[:, 'dayofweek'] = df['date'].dt.dayofweek
df.loc[:, 'weekend'] = (df['date'].dt.weekday >=5).astype(int)
# create an aggregate dictionary
aggs = {}
# for aggregation by month, we calculate the
# number of unique month values and also the mean
aggs['month'] = ['nunique', 'mean']
aggs['weekofyear'] = ['nunique', 'mean']
# we aggregate by num1 and calculate sum, max, min
# and mean values of this column
aggs['num1'] = ['sum','max','min','mean']
# for customer_id, we calculate the total count
aggs['customer_id'] = ['size']
# again for customer_id, we calculate the total unique
aggs['customer_id'] = ['nunique']
# we group by customer_id and calculate the aggregates
agg_df = df.groupby('customer_id').agg(aggs)
agg_df = agg_df.reset_index()
return agg_df
```
Please note that in the above function, we have skipped the categorical variables,
but you can use them in the same way as other aggregates.<br>

Sometimes, for example, when dealing with time-series problems, you might have
features which are not individual values but a list of values. For example,
transactions by a customer in a given period of time. In these cases, we create
different types of features such as: with numerical features, when you are grouping
on a categorical column, you will get features like a list of values which are time
distributed. In these cases, you can create a bunch of statistical features such as:<br>
- Mean
- Max
- Min
- Unique
- Skew
- Kurtosis
- Kstat
- Percentile
- Quantile
- Peak to peak<br>
These can be created using simple numpy functions, as shown in the following
python snippet.<br>

```python
import numpy as np
feature_dict = {}
# calculate mean
feature_dict['mean'] = np.mean(x)
# calculate max
feature_dict['max'] = np.max(x)
# calculate min
feature_dict['min'] = np.min(x)
# calculate standard deviation
feature_dict['std'] = np.std(x)
# calculate variance
feature_dict['var'] = np.var(x)
# peak-to-peak
feature_dict['ptp'] = np.ptp(x)
# percentile features
feature_dict['percentile_10'] = np.percentile(x, 10)
feature_dict['percentile_60'] = np.percentile(x, 60)
feature_dict['percentile_90'] = np.percentile(x, 90)
# quantile features
feature_dict['quantile_5'] = np.quantile(x, 0.05)
feature_dict['quantile_95'] = np.quantile(x, 0.95)
feature_dict['quantile_99'] = np.quantile(x, 0.99)
```
The time series data (list of values) can be converted to a lot of features.
A python library called tsfresh is instrumental in this case.<br>

This is not all; tsfresh offers hundreds of features and tens of variations of different
features that you can use for time series (list of values) based features. In the
examples above, x is a list of values. But that’s not all. There are many other features
that you can create for numerical data with or without categorical data. A simple
way to generate many features is just to create a bunch of polynomial features. For
example, a second-degree polynomial feature from two features “a” and “b” would
include: “a”, “b”, “ab”, “a 2 ” and “b 2 ”.<br>
```python
import numpy as np
# generate a random dataframe with
# 2 columns and 100 rows
df = pd.DataFrame(
np.random.rand(100, 2),
columns=[f"f_{i}" for i in range(1, 3)]
)
```

the more the number of polynomial features and you must also remember that if you have a lot of samples in the dataset, it is going to take a while creating these kinds of features.<br>

Another interesting feature converts the numbers to categories. It’s known as
binning. Let’s check a sample histogram of a random numerical feature. We use ten bins for this figure, and we see that we can divide the data into ten parts. This is accomplished using the pandas’ cut function.
```python
# create bins of the numerical columns
# 10 bins
df["f_bin_10"] = pd.cut(df["f_1"], bins=10, labels=False)
# 100 bins
df["f_bin_100"] = pd.cut(df["f_1"], bins=100, labels=False)
```
When you bin, you can use both the bin and the original feature. Binning also enables you to treat
numerical features as categorical.<br>

Yet another interesting type of feature that you can create from numerical features
is log transformation. 
there is a feature which is a special feature with a very high variance. Compared to other features that
have a low variance (let’s assume that). Thus, we would want to reduce the variance
of this column, and that can be done by taking a log transformation.And we can apply log(1 + x) to this column to reduce its variance.<br>


Sometimes, instead of log, you can also take exponential. A very interesting case is
when you use a log-based evaluation metric, for example, RMSLE. In that case,
you can train on log-transformed targets and convert back to original using
exponential on the prediction. That would help optimize the model for the metric.
Most of the time, these kinds of numerical features are created based on intuition.
There is no formula. If you are working in an industry, you will create your
industry-specific features.<br>
When dealing with both categorical and numerical variables, you might encounter
missing values. We saw some ways to handle missing values in categorical features
in the previous chapter, but there are many more ways to handle missing/NaN
values. This is also considered feature engineering.<br>
For categorical features, let’s keep it super simple. If you ever encounter missing
values in categorical features, treat is as a new category! As simple as this is, it
(almost) always works!<br>

One way to fill missing values in numerical data would be to choose a value that
does not appear in the specific feature and fill using that. For example, let’s say 0
is not seen in the feature. So, we fill all the missing values using 0. This is one of
the ways but might not be the most effective. One of the methods that works better
than filling 0s for numerical data is to fill with mean instead. You can also try to fill
with the median of all the values for that feature, or you can use the most common
value to fill the missing values. There are just so many ways to do this.<br>

A fancy way of filling in the missing values would be to use a k-nearest neighbour
method. You can select a sample with missing values and find the nearest
neighbours utilising some kind of distance metric, for example, Euclidean distance.
Then you can take the mean of all nearest neighbours and fill up the missing value.<br>
Another way of imputing missing values in a column would be to train a regression
model that tries to predict missing values in a column based on other columns. So,
you start with one column that has a missing value and treat this column as the
target column for regression model without the missing values. Using all the other
columns, you now train a model on samples for which there is no missing value in
the concerned column and then try to predict target (the same column) for the
samples that were removed earlier. This way, you have a more robust model based
imputation.<br>
Always remember that imputing values for tree-based models is unnecessary as they
can handle it themselves. And always remember to scale or normalize your
features if you are using linear models like logistic regression or a model like SVM.<br>

## Feature selection
When you are done creating hundreds of thousands of features, it’s time for
selecting a few of them. Well, we should never create hundreds of thousands of
useless features. Having too many features pose a problem well known as the curse
of dimensionality. If you have a lot of features, you must also have a lot of training
samples to capture all the features. What’s considered a “lot” is not defined
correctly and is up to you to figure out by validating your models properly and
checking how much time it takes to train your models.<br>
The simplest form of selecting features would be to remove features with very
low variance. If the features have a very low variance (i.e. very close to 0), they
are close to being constant and thus, do not add any value to any model at all. It
would just be nice to get rid of them and hence lower the complexity. Please note
that the variance also depends on scaling of the data. Scikit-learn has an
implementation for VarianceThreshold that does precisely this.<br>
```python
from sklearn.feature_selection import VarianceThreshold
data = ...
var_thresh = VarianceThreshold(threshold=0.1)
transformed_data = var_thresh.fit_transform(data)
# transformed data will have all columns with variance less
# than 0.1 removed
```
We can also remove features which have a high correlation. For calculating the
correlation between different numerical features, you can use the Pearson
correlation.
```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
# fetch a regression dataset
data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y = data["target"]
# convert to pandas dataframe
df = pd.DataFrame(X, columns=col_names)
# introduce a highly correlated column
df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)
# get correlation matrix (pearson)
df.corr()
```
When we  see that the two features are highly correlated to each other we can remove one of them. <br>
And now we can move to some univariate ways of feature selection. Univariate
feature selection is nothing but a scoring of each feature against a given target.
Mutual information, ANOVA F-test and chi 2 are some of the most popular
methods for univariate feature selection. There are two ways of using these in scikit-
learn.<br>
- SelectKBest: It keeps the top-k scoring features
- SelectPercentile: It keeps the top features which are in a percentage
specified by the user<br>
It must be noted that you can use chi 2 only for data which is non-negative in nature.
This is a particularly useful feature selection technique in natural language
processing when we have a bag of words or tf-idf based features. It’s best to create
a wrapper for univariate feature selection that you can use for almost any new
problem.<br>

```python
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
class UnivariateFeatureSelction:
def __init__(self, n_features, problem_type, scoring):
"""
Custom univariate feature selection wrapper on
different univariate feature selection models from
scikit-learn.
:param n_features: SelectPercentile if float else SelectKBest
:param problem_type: classification or regression
:param scoring: scoring function, string
"""
# for a given problem type, there are only
# a few valid scoring methods
# you can extend this with your own custom
# methods if you wish
if problem_type == "classification":
valid_scoring = {
"f_classif": f_classif,
"chi2": chi2,
"mutual_info_classif": mutual_info_classif
}
else:
valid_scoring = {
"f_regression": f_regression,
"mutual_info_regression": mutual_info_regression
}
# raise exception if we do not have a valid scoring method
if scoring not in valid_scoring:
raise Exception("Invalid scoring function")
# if n_features is int, we use selectkbest
# if n_features is float, we use selectpercentile
# please note that it is int in both cases in sklearn
if isinstance(n_features, int):
self.selection = SelectKBest(
valid_scoring[scoring],
k=n_features
)
elif isinstance(n_features, float):
self.selection = SelectPercentile(
valid_scoring[scoring],
percentile=int(n_features * 100)
)
else:
raise Exception("Invalid type of feature")
# same fit function
def fit(self, X, y):
return self.selection.fit(X, y)
# same transform function
def transform(self, X):
return self.selection.transform(X)
# same fit_transform function
def fit_transform(self, X, y):
return self.selection.fit_transform(X, y)

ufs = UnivariateFeatureSelction(
n_features=0.1,
problem_type="regression",
scoring="f_regression"
)
ufs.fit(X, y)
X_transformed = ufs.transform(X)
```

That should take care of most of your univariate feature selection needs. Please note
that it’s usually better to create less and important features than to create hundreds
of features in the first place. Univariate feature selection may not always perform
well. Most of the time, people prefer doing feature selection using a machine
learning model. Let’s see how that is done.<br>
The simplest form of feature selection that uses a model for selection is known as
**greedy feature selection**. In greedy feature selection, the first step is to choose a
model. The second step is to select a loss/scoring function. And the third and final
step is to iteratively evaluate each feature and add it to the list of “good” features if
it improves loss/score. It can’t get simpler than this. But you must keep in mind that
this is known as greedy feature selection for a reason. This feature selection process
will fit a given model each time it evaluates a feature. The computational cost
associated with this kind of method is very high. It will also take a lot of time for
this kind of feature selection to finish. And if you do not use this feature selection
properly, then you might even end up overfitting the model.<br>


Another greedy approach is known as **recursive feature elimination (RFE)**. In the
previous method, we started with one feature and kept adding new features, but in
RFE, we start with all features and keep removing one feature in every iteration that
provides the least value to a given model. But how to do we know which feature
offers the least value? Well, if we use models like linear support vector machine
(SVM) or logistic regression, we get a coefficient for each feature which decides
the importance of the features. In case of any tree-based models, we get feature
importance in place of coefficients. In each iteration, we can eliminate the least
important feature and keep eliminating it until we reach the number of features
needed. So, yes, we have the ability to decide how many features we want to keep.<br>
When we are doing recursive feature elimination, in each iteration, we remove the
feature which has the feature importance or the feature which has a coefficient
close to 0. Please remember that when you use a model like logistic regression for
binary classification, the coefficients for features are more positive if they are
important for the positive class and more negative if they are important for the
negative class. It’s very easy to modify our greedy feature selection class to create
a new class for recursive feature elimination, but scikit-learn also provides RFE
out of the box.<br>
We saw two different greedy ways to select features from a model. But you can also
fit the model to the data and select features from the model by the feature
coefficients or the importance of features. If you use coefficients, you can select
a threshold, and if the coefficient is above that threshold, you can keep the feature
else eliminate it.<br>
Well, selecting the best features from the model is nothing new. You can choose
features from one model and use another model to train. For example, you can use
Logistic Regression coefficients to select the features and then use Random Forest
to train the model on chosen features. Scikit-learn also offers SelectFromModel
class that helps you choose features directly from a given model. You can also
specify the threshold for coefficients or feature importance if you want and the
maximum number of features you want to select.


## Hyperparameter optimization

With great models, comes the great problem of optimizing hyper-parameters to get
the best scoring model. So, what is this hyper-parameter optimization? Suppose
there is a simple pipeline for your machine learning project. There is a dataset, you
directly apply a model, and then you have results. The parameters that the model
has here are known as hyper-parameters, i.e. the parameters that control the
training/fitting process of the model. If we train a linear regression with SGD,
parameters of a model are the slope and the bias and hyperparameter is learning
rate. You will notice that I use these terms interchangeably in this chapter and
throughout this book. Let’s say there are three parameters a, b, c in the model, and
all these parameters can be integers between 1 and 10. A “correct” combination of
these parameters will provide you with the best result. So, it’s kind of like a suitcase
with a 3-dial combination lock. However, in 3 dial combination lock has only one
correct answer. The model has many right answers. So, how would you find the
best parameters? A method would be to evaluate all the combinations and see which
one improves the metric.<br>
Let’s look at the random forest model from scikit-learn.
```python
RandomForestClassifier(
n_estimators=100,
criterion='gini',
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.0,
max_features='auto',
max_leaf_nodes=None,
min_impurity_decrease=0.0,
min_impurity_split=None,
bootstrap=True,
oob_score=False,
n_jobs=None,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None,
ccp_alpha=0.0,
max_samples=None,
)
```
There are nineteen parameters, and all the combinations of all these parameters for
all the values they can assume are going to be infinite. Normally, we don’t have the
resource and time to do this. Thus, we specify a grid of parameters. A search over
this grid to find the best combination of parameters is known as grid search. We
can say that n_estimators can be 100, 200, 250, 300, 400, 500; max_depth can be
1, 2, 5, 7, 11, 15 and criterion can be gini or entropy. These may not look like a lot
of parameters, but it would take a lot of time for computation if the dataset is too
large.We can make this grid search work by creating three for loops like before and calculating the score on the validation set. It must also be noted that if you have k-fold cross-validation, you need even more loops which implies even more time to
find the perfect parameters. Grid search is therefore not very popular.<br>
Random search is faster than grid search if the number of iterations is less. Using
these two, you can find the optimal (?) parameters for all kinds of models as long
as they have a fit and predict function, which is the standard of scikit-learn.
Sometimes, you might want to use a pipeline. For example, let’s say that we are
dealing with a multiclass classification problem. In this problem, the training data
consists of two text columns, and you are required to build a model to predict the
class. Let’s assume that the pipeline you choose is to first apply tf-idf in a semi-
supervised manner and then use SVD with SVM classifier. Now, the problem is we
have to select the components of SVD and also need to tune the parameters of SVM.
How to do this is shown in the following snippet.<br>

The pipeline shown here has SVD (Singular Value Decomposition), standard
scaling and an SVM (Support Vector Machines) model. Please note that you won’t
be able to run the above code as it is as training data is not available.
When we go into advanced hyperparameter optimization techniques, we can take a
look at **minimization of functions** using different kinds of minimization
algorithms. This can be achieved by using many minimization functions such as
downhill simplex algorithm, Nelder-Mead optimization, using a Bayesian
technique with Gaussian process for finding optimal parameters or by using a
genetic algorithm. I will talk more about the application of downhill simplex and
Nelder-Mead in ensembling and stacking chapter. First, let’s see how the gaussian
process can be used for hyper-parameter optimization. These kinds of algorithms
need a function they can optimize. Most of the time, it’s about the minimization of
this function, like we **minimize loss**.<br>
So, let’s say, you want to find the best parameters for best accuracy and obviously,
the more the accuracy is better. Now we cannot minimize the accuracy, but we can
minimize it when we multiply it by -1. This way, we are minimizing the negative
of accuracy, but in fact, we are maximizing accuracy. Using &**Bayesian optimization with gaussian process** can be accomplished by using gp_minimize
function from scikit-optimize (skopt) library.<br>
There are many libraries available that offer hyperparameter optimization. scikit-
optimize is one such library that you can use. Another useful library for
hyperparameter optimization is hyperopt. hyperopt uses Tree-structured Parzen
Estimator (TPE) to find the most optimal parameters.<br>
Once you get better with hand-tuning the parameters, you might not even need any
automated hyper-parameter tuning. When you create large models or introduce a
lot of features, you also make it susceptible to overfitting the training data. To avoid
overfitting, you need to introduce noise in training data features or penalize the cost
function. This penalization is called **regularization** and helps with generalizing the
model. In linear models, the most common types of regularizations are L1 and L2.
L1 is also known as Lasso regression and L2 as Ridge regression. When it comes
to neural networks, we use dropouts, the addition of augmentations, noise, etc. to
regularize our models. Using hyper-parameter optimization, you can also find the
correct penalty to use.

## Approaching image classification & segmentation

What are the different approaches that we can apply to images? Image is nothing
but a matrix of numbers. The computer cannot see the images as humans do. It only
looks at numbers, and that’s what the images are. A grayscale image is a two-
dimensional matrix with values ranging from 0 to 255. 0 is black, 255 is white and
in between you have all shades of grey. Previously, when there was no deep learning
(or when deep learning was not popular), people used to look at pixels. Each pixel
was a feature. You can do this easily in Python. Just read the grayscale image using
OpenCV or Python-PIL, convert to a numpy array and ravel (flatten) the matrix. If
you are dealing with RGB images, then you have three matrices instead of one. But
the idea remains the same.<br>
This matrix consists of
values ranging from 0 to 255 (included) and is of size 256x256 (also known as
pixels). As you can see that the ravelled version is nothing but a vector of size M, where M
= N * N. In this case, this vector is of the size 256 * 256 = 65536.<br>
Now, if we go ahead and do it for all the images in our dataset, we have 65536
features for each sample. We can now quickly build a decision tree model or
random forest or SVM-based model on this data. The models will look at pixel
values and would try to separate positive samples from negative samples (in case
of a binary classification problem).<br>
All of you must have heard about the cats vs dogs problem. It's a classic one. But
let's try something different. If you remember, at the beginning of the chapter on
evaluation metrics, I introduced you to a dataset of pneumothorax images. So, let’s
try building a model to detect if an X-ray image of a lung has pneumothorax or not.
That is, a (not so) simple binary classification.<br>
The original dataset is about detecting where exactly pneumothorax is present, but
we have modified the problem to find if the given x-ray image has pneumothorax
or not. Don’t worry; we will cover the where part in this chapter. The dataset
consists of 10675 unique images and 2379 have pneumothorax (note that these
numbers are after some cleaning of data and thus do not match original dataset). As
a data doctor would say: this is a **classic case of skewed binary classification**.
Therefore, we choose the evaluation metric to be AUC and go for a stratified k-fold
cross-validation scheme.<br>
You can flatten out the features and try some classical methods like SVM, RF for
doing classification, which is perfectly fine, but it won't get you anywhere near state
of the art. Also, the images are of size 1024x1024. It’s going to take a long time to
train a model on this dataset. For what it’s worth, let’s try building a simple random
forest model on this data. Since the images are grayscale, we do not need to do any
kind of conversion. We will resize the images to 256x256 to make them smaller
and use AUC as a metric as discussed before.<br>
let’s take a look at one of the most famous deep learning models AlexNet and see what’s
happening there.<br>
Nowadays, you might say that it is a basic **deep convolutional neural network**,
but it is the foundation of many new deep nets (deep neural networks). We see that
the network  is a convolutional neural network with five convolution
layers, two dense layers and an output layer. We see that there is also max pooling.
What is it? Let’s look at some terms which you will come across when doing deep
learning.<br>
Figure 4 introduces two new terms: filter and strides. **Filters** are nothing but two-
dimensional matrices which are initialized by a given function. **“He initialization”** 
which is also known **Kaiming normal initialization** is a good choice for
convolutional neural networks. It is because most modern networks use **ReLU**
(Rectified Linear Units) activation function and proper initialization is required to
avoid the problem of **vanishing gradients** (when gradients approach zero and
weights of network do not change). This filter is convolved with the image.
Convolution is nothing but a summation of elementwise multiplication (cross-
correlation) between the filter and the pixels it is currently overlapping in a given
image. You can read more about convolution in any high school mathematics
textbook. We start convolution of this filter from the top left corner of the image,
and we move it horizontally. If we move it by 1 pixel, the stride is 1. If we move it
by 2 pixels, the stride is 2. And that’s what **stride** is.<br>    
Stride is a useful concept even in natural language processing, e.g. in question and
answering systems when you have to filter answer from a large text corpus. When
we are exhausted horizontally, we move the filter by the same stride downwards
vertically, starting from left again. Figure 4 also shows a filter going outside the
image. In these cases, it’s not possible to calculate the convolution. So, we skip it.
If you don’t want to skip it, you will need to **pad the image**. It must also be noted
that convolution will decrease the size of the image. Padding is also a way to keep
the size of the image the same. In figure 4, A 3x3 filter is moving horizontally and
vertically, and every time it moves, it skips two columns and two rows (i.e. pixels)
respectively. Since it skips two pixels, stride = 2. And resulting image size is [(8-3)
/ 2] + 1 = 3.5. We take the floor of 3.5, so its 3x3. You can do it by hand by moving
the filters on a pen and paper.<br>
Now, we have a 3x3 filter which is moving with a stride of 1. Size of the original image is 6x6, and we have added padding of
1 .The padding of 1 means increasing the size of the image by adding zero pixels
on each side once. In this case, the resulting image will be of the same size as the
input image, i.e. 6x6. Another relevant term that you might come across when
dealing with deep neural networks is dilation.<br>

In **dilation**, we expand the filter by N-1, where N is the value of dilation rate or
simply known as dilation. In this kind of kernel with dilation, you skip some pixels
in each convolution. This is particularly effective in segmentation tasks. Please note
that we have only talked about 2-dimensional convolutions. There are 1-d
convolutions too and also in higher dimensions. All work on the same underlying
concept.<br>
Next comes max-pooling. **Max pooling** is nothing but a filter which returns max.
So, instead of convolution, we are extracting the max value of pixels. Similarly,
**average pooling** or **mean-pooling** returns mean of pixels. They are used in the
same way as the convolutional kernel. Pooling is faster than convolution and is a
way to down-sample the image. Max pooling detects edges and average pooling
smoothens the image.<br>
Now, we are well prepared to start building our first convolutional neural
network in PyTorch. PyTorch provides an intuitive and easy way to implement deep
neural networks, and you don’t need to care about back-propagation.<br>
You can design your own convolutional neural networks for your task, and many
times it is a good idea to start from something on your own. Let’s build a network
to classify images from our initial dataset of this chapter into categories of having
pneumothorax or not. But first, let’s prepare some files. The first step would be to
create a folds file, i.e. train.csv but with a new column kfold. We will create five
folds. Since I have shown how to do this for different datasets in this book, I will
skip this part and leave it an exercise for you. For PyTorch based neural networks,
we need to create a dataset class. The objective of the dataset class is to return an
item or sample of data. This sample of data should consist of everything you need
in order to train or evaluate your model.<br>
This model seems to perform the best. However, you might be able to tune the
different parameters and image size in AlexNet to get a better score. Using
augmentations will improve the score further. Optimising deep neural networks is
difficult but not impossible. Choose Adam optimizer, use a low learning rate,
reduce learning rate on a plateau of validation loss, try some augmentations, try
preprocessing the images (e.g. cropping if needed, this can also be considered pre-
processing), change the batch size, etc. There’s a lot that you can do to optimize
your deep neural network.<br>
**ResNet** is an architecture much more complicated compared to AlexNet. ResNet
stands for Residual Neural Network and was introduced by K. He, X. Zhang, S.
Ren and J. Sun in the paper, deep residual learning for image recognition, in 2015.
ResNet consists of **residual blocks** that transfer the knowledge from one layer to
further layers by skipping some layers in between. These kinds of connections of
layers are known as **skip-connections** since we are skipping one or more layers.
Skip-connections help with the vanishing gradient issue by propagating the
gradients to further layers. This allows us to train very large convolutional neural
networks without loss of performance.Usually, the training loss increases at a given point if we are using a large neural network, but that can be prevented by using skip-connections.<br>
A residual block is quite simple to understand. You take the output from a layer,
skip some layers and add that output to a layer further in the network. The dotted
lines mean that the input shape needs to be adjusted as max-pooling is being used
and use of max-pooling changes the size of the output.<br>
ResNet comes in many different variations: 18, 34, 50, 101 and 152 layers and all
of them are available with weights pre-trained on ImageNet dataset. These days
pretrained models work for (almost) everything but make sure that you start with
smaller models, for example, begin with resnet-18 rather than resnet-50. Some other
ImageNet pre-trained models include:<br>
- Inception
- DenseNet (different variations)
- NASNet
- PNASNet
- VGG
- Xception
- ResNeXt
- EfficientNet<br>






**Segmentation** is a task which is quite popular in computer vision. In a segmentation
task, we try to remove/extract foreground from background. Foreground and background can have different definitions. We can also say that it is a pixel-wise classification task in which your job is to assign a class to each pixel in a given
image. The pneumothorax dataset that we are working on is, in fact, a segmentation
task. In this task, given the chest radiographic images, we are required to segment
pneumothorax. The most popular model used for segmentation tasks is **U-Net**.<br>
U-Nets have two parts: encoder and decoder. The encoder is the same as any
convnet you have seen till now. The decoder is a bit different. Decoder consists of
up-convolutional layers. In up-convolutions (transposed convolutions), we use
filters that when applied to a small image, creates a larger image. In PyTorch, you
can use ConvTranspose2d for this operation. It must be noted that up-convolution
is not the same as up-sampling. Up-sampling is an easy process in which we apply
a function to an image to resize it. In up-convolution, we learn the filters. We take
some parts of the encoder as inputs to some of the decoders. This is important for
the up-convolutional layers.<br>

We see that the encoder part of the U-Net is a nothing but a simple convolutional
network. We can, thus, replace this with any network such as ResNet. The
replacement can also be done with pretrained weights. Thus, we can use a ResNet
based encoder which is pretrained on ImageNet and a generic decoder. In place of
ResNet, many different network architectures can be used. Segmentation Models
Pytorch 12 by Pavel Yakubovskiy is an implementation of many such variations
where an encoder can be replaced by a pretrained model. Let’s apply a ResNet
based U-Net for pneumothorax detection problem.<br>

Most of the problems like this should have two inputs: the original image and a
mask. In the case of multiple objects, there will be multiple masks. In our
pneumothorax dataset, we are provided with RLE instead. RLE stands for run-length encoding and is a way to represent binary masks to save space. Going deep into RLE is beyond the scope of this chapter. So, let’s assume that we have an input
image and corresponding mask. Let’s first design a dataset class which outputs
image and mask images. Please note that we will create these scripts in such a way
that they can be applied to almost any segmentation problem. The training dataset
is a CSV file consisting only of image ids which are also filenames.<br>    



## Approaching text classification/regression
In general, these problems are also known as
Natural Language Processing (NLP) problems. NLP problems are also like
images in the sense that, it’s quite different. You need to create pipelines you have
never created before for tabular problems. You need to understand the business case
to build a good model. By the way, that is true for anything in machine learning.
Building models will take you to a certain level, but to improve and contribute to a
business you are building the model for, you must understand how it impacts the
business. Let’s not get too philosophical here.<br>

There are many different types of NLP problems, and the most common type is the
classification of strings. Many times, it is seen that people are doing well with
tabular data or with images, but when it comes to text, they don’t even have a clue
where to start from. Text data is no different than other types of datasets. For
computers, everything is numbers.<br>
Let’s say we start with a fundamental task of sentiment classification. We will try
to classify sentiment from movie reviews. So, you have a text, and there is a
sentiment associated with it. How will you approach this kind of problem? Apply a
deep neural network right, or maybe muppets can come and save you? No,
absolutely wrong. You start with the basics. Let’s see what this data looks like first.<br>
We start with IMDB movie review dataset 15 that consists of 25000 reviews for
positive sentiment and 25000 reviews for negative sentiment.
The concepts that I will discuss here can be applied to almost any text classification
dataset. This dataset is quite easy to understand. One review maps to one target variable.
Note that I wrote review instead of sentence. A review is a bunch of sentences. So,
until now you must have seen classifying only a single sentence, but in this problem,
we will be classifying multiple sentences.In simple words, it means that not only one sentence contributes to the sentiment, but the sentiment score is a combination of score from multiple sentences.<br>

How would you start with such a problem?
A simple way would be just to create two handmade lists of words. One list will
contain all the positive words you can imagine, for example, good, awesome, nice,
etc. and another list will include all the negative words, such as bad, evil, etc. Let’s
leave examples of bad words else I’ll have to make this book available only for 18+.
Once you have these lists, you do not even need a model to make a prediction.
These lists are also known as sentiment lexicons. A bunch of them for different
languages are available on the internet.<br> 
You can have a simple counter that counts the number of positive and negative
words in the sentence. If the number of positive words is higher, it is a positive
sentiment, and if the number of negative words is higher, it is a sentence with a
negative sentiment. If none of them are present in the sentence, you can say that the
sentence has a neutral sentiment. This is one of the oldest ways, and some people
still use it.<br>
However, this kind of approach does not take a lot into consideration. And as you
can see that our split() is also not perfect. If you use split(), a sentence like:<br>
“hi, how are you?”<br>
gets split into<br>
[“hi,”, “how”, “are”, “you?”]<br>
This is not ideal, because you see the comma and question mark, they are not split.
It is therefore not recommended to use this method if you don’t have a pre-
processing that handles these special characters before the split. Splitting a string
into a list of words is known as tokenization. One of the most popular tokenization
comes from NLTK (Natural Language Tool Kit).<br>
One of the basic models that you should always try with a classification problem in
NLP is bag of words. In bag of words, we create a huge sparse matrix that stores
counts of all the words in our corpus (corpus = all the documents = all the
sentences). For this, we will use CountVectorizer from scikit-learn.<br>

Now, we have more words in the vocabulary. Thus, we can now create a sparse
matrix by using all the sentences in IMDB dataset and can build a model. The ratio
to positive and negative samples in this dataset is 1:1, and thus, we can use accuracy
as the metric.<br>
We will use StratifiedKFold and create a single script to train five folds. Which model to use you ask? Which is the fastest model for high dimensional sparse data? Logistic regression. We will use logistic regression for this dataset to
start with and to create our first actual benchmark.<br>


all we did was use bag of words with logistic regression! This is super amazing! However, this model took a lot of time
to train, let’s see if we can improve the time by using naïve bayes classifier. Naïve
bayes classifier is quite popular in NLP tasks as the sparse matrices are huge and
naïve bayes is a simple model. To use this model, we need to change one import
and the line with the model. Let’s see how this model performs. We will use
MultinomialNB from scikit-learn.<br>
Another method in NLP that most of the people these days tend to ignore or don’t
care to know about is called **TF-IDF**. TF is term frequencies, and IDF is inverse
document frequency. It might seem difficult from these terms, but things will
become apparent with the formulae for TF and IDF.

$$
TF(t) = \frac{\text{Number of times a term } t \text{ appears in a document}}{\text{Total number of terms in the document}}
$$

$$
IDF(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents with term } t \text{ in it}}\right)
$$

And TF-IDF for a term t is defined as:

$$
\text{TF-IDF}(t) = TF(t) \times IDF(t)
$$

Similar to CountVectorizer in scikit-learn, we have TfidfVectorizer. Let’s try using
it the same way we used CountVectorizer. We see that instead of integer values, this time we get floats. Replacing
CountVectorizer with TfidfVectorizer is also a piece of cake. Scikit-learn also offers
TfidfTransformer. If you have count values, you can use TfidfTransformer and get
the same behaviour as TfidfVectorizer.<br>

Another interesting concept in NLP is n-grams. N-grams are combinations of
words in order. N-grams are easy to create. You just need to take care of the order.
To make things even more comfortable, we can use n-gram implementation from
NLTK.<br>
Similarly, we can also create 2-grams, or 4-grams, etc. Now, these n-grams become
a part of our vocab, and when we calculate counts or tf-idf, we consider one n-gram
as one entirely new token. So, in a way, we are incorporating context to some extent.
Both CountVectorizer and TfidfVectorizer implementations of scikit-learn offers n-
grams by ngram_range parameter, which has a minimum and maximum limit. By
default, this is (1, 1). When we change it to (1, 3), we are looking at unigrams,
bigrams and trigrams. The code change is minimal. Since we had the best result till
now with tf-idf, let’s see if including n-grams up to trigrams improves the model.<br>
we do not see any improvements. Maybe we can get improvements by using only up to bigrams. I’m not showing that part here.
Probably you can try to do it on your own.<br>
There are a lot more things in the basics of NLP. One term that you must be aware
of is stemming. Another is lemmatization. **Stemming and lemmatization** reduce a
word to its smallest form. In the case of stemming, the processed word is called the
stemmed word, and in the case of lemmatization, it is known as the lemma. It must
be noted that lemmatization is more aggressive than stemming and stemming is
more popular and widely used. Both stemming and lemmatization come from
linguistics. And you need to have an in-depth knowledge of a given language if you
plan to make a stemmer or lemmatizer for that language. Going into too much detail
of these would mean adding one more chapter in this book. Both stemming and
lemmatization can be done easily by using the NLTK package. Let’s take a look at
some examples for both of them. There are many different types of stemmers and
lemmatizers. I will show an example using the most common **Snowball Stemmer**
and **WordNet Lemmatizer**.<br>