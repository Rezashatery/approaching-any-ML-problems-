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
