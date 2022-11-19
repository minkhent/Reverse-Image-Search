# Reverse-Image-Search

<h2>Dataset</h2>
The original dataset is Caltech101 but I only use 10 categories of dataset.

<h2>Model</h2>
ResNet-50 model is used for feature extraction process without the top classification layers, so
I get only the bottleneck features.The ResNet-50 model generated 2,048 features from the provided
image. Each feature is floating-point values represent an image .And then all of these extracted feature
vectors length are reduced by PCA.PCA is considered one of the techniques for dimensionality reduction.It
does not eliminate redundant features; rather, it generates a new
set of features that are a linear combination of the input features.
These features are known as principal components. All of these feature vector are
clustered using KNN algorithm.
