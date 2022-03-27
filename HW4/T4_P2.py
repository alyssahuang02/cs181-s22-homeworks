# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# small_dataset = np.load("P2_Autograder_Data.npy")

def euclidean_dist(x1, x2):
    return np.linalg.norm(x1-x2, ord=2)

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.losses = []
        self.k_means = None
        self.groupings = None
        self.indices_clusters = None

    # X is a (N x 784) array where 784 is the dimensions of each of the N images.
    def fit(self, X):
        N = X.shape[0]
        # random init
        k_means = np.random.randn(10, 784)

        for run in range(10):
            loss = 0
            # step 1: assign each point to closest cluster
            groupings = {}
            final_run = run == 9
            if final_run:
                indices_clusters = {}
            
            for i in range(self.K):
                groupings[i] = []
                if final_run:
                    indices_clusters[i] = []

            for index, x_i in enumerate(X):
                min_dist = np.inf
                best_mean_index = 0
                for i, mean in enumerate(k_means):
                    cur_dist = euclidean_dist(x_i, mean)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        best_mean_index = i
                groupings[best_mean_index].append(x_i)
                if final_run:
                    indices_clusters[best_mean_index].append(index)
                loss += min_dist**2
            
            self.losses.append(loss)

            # step 2: compute the new means
            for i in groupings:
                x_is = groupings[i]
                mat = np.matrix(x_is)
                if mat.shape[1]!=0:
                    k_means[i] = np.mean(mat, axis=0)
        
        self.k_means = k_means
        self.groupings = groupings
        self.indices_clusters = list(indices_clusters.values())
    
    def plot_losses(self):
        plt.plot(range(10), self.losses)
        plt.title("Losses over Iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Residual Sum of Squares")
        plt.tight_layout()
        plt.savefig('kmeans_loss.png')
        plt.show()

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.k_means

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.HACmeans = None
        self.indices_clusters = None
    
    def find_closest(self, distances_between_clusters):
        # every entry of distances_between_clusters is a matrix that gives dists between pairs in groups
        group_dists = {}
        if self.linkage == 'centroid':
            group_dists = distances_between_clusters
        for key, dist in distances_between_clusters.items():
            if self.linkage == "max":
                group_dists[key] = np.max(dist.flatten())
            elif self.linkage == "min":
                group_dists[key] = np.min(dist.flatten())
        merge_indices = min(group_dists, key= lambda x: group_dists[x])
        return merge_indices
    
    def fit(self, X):
        # initialize n cluster
        N = X.shape[0]
        clusters = [[i] for i in range(X.shape[0])] # index of list represents cluster # and entry is number representing which X_i
        # run while loop until everything is in 10 clusters
        # TODO: ASK IF THIS IS OKAY OR IF WE HAVE TO GO DOWN TO 1????
        while N > 10:
            distances_between_clusters = {}
            for i in range(N):
                for j in range(i+1, N):
                    if self.linkage != "centroid":
                        dist = distance.cdist(X[clusters[i]], X[clusters[j]], 'euclidean')
                    else:
                        first = X[clusters[i]]
                        second = X[clusters[j]]
                        centroid1 = np.mean(first, axis=0)
                        # print(centroid1.shape)
                        centroid2 = np.mean(second, axis=0)
                        dist = euclidean_dist(centroid1, centroid2)
                    distances_between_clusters[(i, j)] = dist
            merge1, merge2 = self.find_closest(distances_between_clusters)
            merged = clusters[merge1] + clusters[merge2]
            clusters.pop(merge1)
            clusters.pop(merge2 - 1) # since merge2 will always be bigger (off by one cuz just popped)
            clusters.append(merged)
            N -= 1
        
        self.indices_clusters = clusters
        result = []
        for cluster in clusters:
            result.append([X[index] for index in cluster])
        self.HACmeans = result
    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        print('here')
        HACmeans = self.HACmeans # should be 10 by ? (where ? is bunch of 1x784 vectors)
        means = np.zeros((n_clusters, 784))
        for i in range(n_clusters):
            mat = np.matrix(HACmeans[i])
            means[i] = np.mean(mat, axis=0)
        return means

# Plot losses code
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(large_dataset)
KMeansClassifier.plot_losses()
kmeans_clusters = KMeansClassifier.groupings

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    if standardized:
        plt.savefig('kmeans_standardized.png')
    else:
        plt.savefig('kmeans.png')
    plt.show()

# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
large_dataset_standardized = np.zeros(large_dataset.T.shape)
# The standardized value for the i-th pixel in a particular data point would be = (pixel_value - mean) / std . 
# The mean would be the average value of the i-th pixel across all the data points. 
for i, col in enumerate(large_dataset.T):
    mean = np.mean(col)
    stdev = np.std(col)
    if stdev == 0:
        stdev = 1
    large_dataset_standardized[i] = (col - mean)/stdev

large_dataset_standardized = large_dataset_standardized.T
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10
HAC_clusters = {}
HAC_indices_clusters = {}

fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    HAC_clusters[l] = hac.HACmeans
    HAC_indices_clusters[l] = hac.indices_clusters
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.savefig('hac.png')
plt.show()

# TODO: Write plotting code for part 5
def plot_counts(dictionary, title, save_fig):
    plt.bar(dictionary.keys(), dictionary.values())
    plt.title(title)
    plt.xlabel("Cluster index")
    plt.ylabel("Number of images in cluster")
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.show()

kmeans_counts = {}
for i in kmeans_clusters:
    kmeans_counts[i] = len(kmeans_clusters[i])
    # print(kmeans_clusters[i].shape)
    # CHECK THIS BY PRINTING?????
plot_counts(kmeans_counts, 'Kmeans', 'kmeans_counts.png')

for linkage in HAC_clusters:
    hac_counts = {}
    double_list = HAC_clusters[linkage]
    for i, cluster in enumerate(double_list):
        hac_counts[i] = len(cluster)
        # ALSO CHECK VIA PRINTING
    plot_counts(hac_counts, 'HAC: ' + linkage, 'HAC_' + linkage+ '_counts.png')

# TODO: Write plotting code for part 6

def generate_confusion_matrix(indices_1, indices_2):
    matrix = np.zeros((10, 10))
    for img_idx in range(len(small_dataset)):
        i = -1
        j = -1
        for label, cluster_idxs in enumerate(indices_1):
            if img_idx in cluster_idxs:
                i = label
                break
        for label, cluster_idxs in enumerate(indices_2):
            if img_idx in cluster_idxs:
                j = label
                break
        matrix[i][j] += 1
    return matrix

KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(small_dataset)
kmeans_indices_clusters = KMeansClassifier.indices_clusters

for i, linkage in enumerate(HAC_indices_clusters):
    hac = HAC_indices_clusters[linkage]
    mat = generate_confusion_matrix(kmeans_indices_clusters, hac)
    fig = sns.heatmap(mat, annot=True)
    plt.xlabel('Kmeans')
    plt.ylabel('HAC: ' + linkage)
    plt.tight_layout()
    plt.savefig('kmeans_' + linkage + '.png')
    plt.show()

hac1 = HAC_indices_clusters['min']
hac2 = HAC_indices_clusters['max']
mat = generate_confusion_matrix(hac1, hac2)
fig = sns.heatmap(mat, annot=True)
plt.xlabel('HAC: min')
plt.ylabel('HAC: max')
plt.tight_layout()
plt.savefig('min_max.png')
plt.show()

hac1 = HAC_indices_clusters['min']
hac2 = HAC_indices_clusters['centroid']
mat = generate_confusion_matrix(hac1, hac2)
fig = sns.heatmap(mat, annot=True)
plt.xlabel('HAC: min')
plt.ylabel('HAC: centroid')
plt.tight_layout()
plt.savefig('min_centroid.png')
plt.show()

hac1 = HAC_indices_clusters['max']
hac2 = HAC_indices_clusters['centroid']
mat = generate_confusion_matrix(hac1, hac2)
fig = sns.heatmap(mat, annot=True)
plt.xlabel('HAC: max')
plt.ylabel('HAC: centroid')
plt.tight_layout()
plt.savefig('max_centroid.png')
plt.show()