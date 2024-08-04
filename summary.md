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
