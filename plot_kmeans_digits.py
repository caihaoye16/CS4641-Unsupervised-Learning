"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for
K-means in terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster
quality metrics to judge the goodness of fit of the cluster labels to the
ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
print(__doc__)

import os
import pickle
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection

np.set_printoptions(threshold=np.nan)
np.random.seed(42)

def bench_k_means(estimator, name, data):
    t0 = time()
    print('data size', len(data))
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

def test_kmeans(data, original):
  print(79 * '-')
  print('% 9s' % 'init'
        '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

  bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                name="k-means++", data=data)

  bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                name="random", data=data)

  # in this case the seeding of the centers is deterministic, hence we run the
  # kmeans algorithm only once with n_init=1
  if original:
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
  print(79 * '-')

def twoDVis(estimator, name):
  ###############################################################################
  # Visualize the results on PCA-reduced data
  est = estimator.fit(data)
  reduced_data = est.transform(data)

  kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
  kmeans.fit(reduced_data)

  # Step size of the mesh. Decrease to increase the quality of the VQ.
  # h = .05     # point in the mesh [x_min, x_max]x[y_min, y_max].

  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() - 0.1 * (reduced_data[:, 0].max() - reduced_data[:, 0].min()), reduced_data[:, 0].max() + 0.1 * (reduced_data[:, 0].max() - reduced_data[:, 0].min())
  y_min, y_max = reduced_data[:, 1].min() - 0.1 * (reduced_data[:, 1].max() - reduced_data[:, 1].min()), reduced_data[:, 1].max() + 0.1 * (reduced_data[:, 1].max() - reduced_data[:, 1].min())
  h = (x_max - x_min) / 500
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain labels for each point in mesh. Use last trained model.
  Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure(1)
  plt.clf()
  plt.title(name+'-reduced data visualization')
  plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
  plt.colorbar()
  plt.savefig(name+'_vis.png')
  # plt.show()
  plt.clf()
  plt.imshow(Z, interpolation='nearest',
             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
             cmap=plt.cm.Paired,
             aspect='auto', origin='lower')

  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # Plot the centroids as a white X
  centroids = kmeans.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1],
              marker='x', s=169, linewidths=3,
              color='w', zorder=10)
  plt.title('K-means clustering on the MNIST dataset ('+name+'-reduced data)\n'
            'Centroids are marked with white cross')
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())
  plt.savefig('kmeans_'+name+'.png')
  # plt.show()

def kmeans_on_reduction(estimator, n_components, name, itr = None, eps = None):
  if itr != None:
    est = estimator(n_components=n_components, max_iter=itr).fit(data)
  elif eps != None:
    est = estimator(n_components=n_components, eps = eps).fit(data)
  else:
    est = estimator(n_components=n_components).fit(data)
  reduced_data = est.transform(data)
  # print(reduced_data.shape)
  reduced_data_test = est.transform(data_test)

  with open('reduced_data_train_'+name+'.pkl', 'wb') as f:
    pickle.dump(reduced_data, f)

  with open('reduced_data_test_'+name+'.pkl', 'wb') as f:
    pickle.dump(reduced_data_test, f)

  print('Kmeans on '+name)
  if n_components != 'auto':
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_components))
  else:
    print("n_digits: %d, \t n_samples %d, \t n_features %s"
        % (n_digits, n_samples, n_components))
  test_kmeans(reduced_data, False)

  kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
  kmeans.fit(reduced_data)
  p_labels = kmeans.predict(reduced_data)
  p_labels = np.expand_dims(p_labels, axis=1)
  new_data = np.hstack((reduced_data, p_labels))
  p_labels_test = kmeans.predict(reduced_data_test)
  p_labels_test = np.expand_dims(p_labels_test, axis=1)
  new_data_test = np.hstack((reduced_data_test, p_labels_test))

  with open(name+'_Kmeans_feature_train.pkl', 'wb') as f:
    pickle.dump(new_data, f)

  with open(name+'_Kmeans_feature_test.pkl', 'wb') as f:
    pickle.dump(new_data_test, f)

  twoDVis(estimator(n_components=2), name)

train_df = pd.read_csv('train.csv', header=0)
train_df = shuffle(train_df, n_samples = 10000, random_state = 0)

X = train_df.drop(["label"], axis = 1).values.astype(float)
X = X / 255.0
# print data.shape
n_features = X.shape[1]
Y = train_df["label"].values
data, data_test, labels, labels_test = train_test_split(X, Y, test_size = 0.3, random_state = 0) 

with open('data_train.pkl', 'wb') as f:
  pickle.dump(data, f)

with open('data_test.pkl', 'wb') as f:
  pickle.dump(data_test, f)


n_samples = len(data)
n_digits = len(np.unique(labels))

# print labels

with open('labels_train.pkl', 'wb') as f:
  pickle.dump(labels, f)

with open('labels_test.pkl', 'wb') as f:
  pickle.dump(labels_test, f)

sample_size = 300


# Need to test component number here!

print('Kmeans')
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
test_kmeans(data, True)




###############################################################################
# Rerun Kmeans on PCA-reduced data

kmeans_on_reduction(PCA, 10, 'PCA')

###############################################################################
# Rerun Kmeans on PCA-reduced data
  
kmeans_on_reduction(FastICA, 10, 'ICA', itr=100000)

###############################################################################
# Rerun Kmeans on RP-reduced data
  
kmeans_on_reduction(SparseRandomProjection, 'auto', 'RP', eps=0.8)

###############################################################################
# Rerun Kmeans on VAE-reduced data

os.system('python vae.py')

with open('reduced_data_train_VAE.pkl', 'rb') as f:
  reduced_data = pickle.load(f)

with open('reduced_data_test_VAE.pkl', 'rb') as f:
  reduced_data_test = pickle.load(f)

with open('labels_train_VAE.pkl', 'rb') as f:
  labels = pickle.load(f)

with open('labels_test_VAE.pkl', 'rb') as f:
  labels_test = pickle.load(f)

n_samples = len(reduced_data)

print('Kmeans on VAE')
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples,2))
test_kmeans(reduced_data, True)

kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)
p_labels = kmeans.predict(reduced_data)
p_labels = np.expand_dims(p_labels, axis=1)
new_data = np.hstack((reduced_data, p_labels))
p_labels_test = kmeans.predict(reduced_data_test)
p_labels_test = np.expand_dims(p_labels_test, axis=1)
new_data_test = np.hstack((reduced_data_test, p_labels_test))

with open('VAE_Kmeans_feature_train.pkl', 'wb') as f:
  pickle.dump(new_data, f)

with open('VAE_Kmeans_feature_test.pkl', 'wb') as f:
  pickle.dump(new_data_test, f)

name = 'VAE'
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 0.1 * (reduced_data[:, 0].max() - reduced_data[:, 0].min()), reduced_data[:, 0].max() + 0.1 * (reduced_data[:, 0].max() - reduced_data[:, 0].min())
y_min, y_max = reduced_data[:, 1].min() - 0.1 * (reduced_data[:, 1].max() - reduced_data[:, 1].min()), reduced_data[:, 1].max() + 0.1 * (reduced_data[:, 1].max() - reduced_data[:, 1].min())
h = (x_max - x_min) / 500
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.title(name+'-reduced data visualization')
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
plt.colorbar()
plt.savefig(name+'_vis.png')
# plt.show()
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the MNIST dataset ('+name+'-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('kmeans_'+name+'.png')