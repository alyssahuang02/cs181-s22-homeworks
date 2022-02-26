import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.means = None
        self.shared_covariance = None
        self.covariance1 = None
        self.covariance2 = None
        self.covariance3 = None
        self.priors = np.zeros(3)
        self.class_0 = None
        self.class_1 = None
        self.class_2 = None

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    # Calculates the optimal means and covariances (k of each) according to results in 2.4 and 2.6
    def fit(self, X, y):
        sums = np.zeros(3)
        col1 = np.zeros(3)
        col2 = np.zeros(3)
        Xs = [[], [], []]
        for i, elem in enumerate(y):
            sums[elem] += 1
            col1[elem] += X[i][0]
            col2[elem] += X[i][1]
            Xs[elem].append(X[i])
        for i in range(3):
            col1[i] /= sums[i]
            col2[i] /= sums[i]
        means = np.vstack((col1, col2))
        self.means=means.T
        self.covariance1 = np.cov(np.matrix(Xs[0]).T, ddof=0)
        self.covariance2 = np.cov(np.matrix(Xs[1]).T, ddof=0)
        self.covariance3 = np.cov(np.matrix(Xs[2]).T, ddof=0)
        if self.is_shared_covariance:
            shared_cov = (self.covariance1 * (len(Xs[0])) + self.covariance2 * (len(Xs[1])) + self.covariance3 * (len(Xs[2])))/X.shape[0]
            self.shared_covariance = shared_cov
        else:
            self.covariance1 = np.cov(np.matrix(Xs[0]).T, ddof=0)
            self.covariance2 = np.cov(np.matrix(Xs[1]).T, ddof=0)
            self.covariance3 = np.cov(np.matrix(Xs[2]).T, ddof=0)
        
        for i in range(3):
            self.priors[i] = len(Xs[i])/len(X)

    # TODO: Implement this method!
    # Generate generative models for each class and predicts based on which model returns the highest likelihood
    def predict(self, X_pred):
        if self.is_shared_covariance:
            self.class_0 = mvn(mean=self.means[0], cov=self.shared_covariance)
            self.class_1 = mvn(mean=self.means[1], cov=self.shared_covariance)
            self.class_2 = mvn(mean=self.means[2], cov=self.shared_covariance)
        else:
            self.class_0 = mvn(mean=self.means[0], cov=self.covariance1)
            self.class_1 = mvn(mean=self.means[1], cov=self.covariance2)
            self.class_2 = mvn(mean=self.means[2], cov=self.covariance3)
        prob_0 = self.class_0.pdf(X_pred) * self.priors[0]
        prob_1 = self.class_1.pdf(X_pred) * self.priors[1]
        prob_2 = self.class_2.pdf(X_pred) * self.priors[2]
        probs = np.vstack((prob_0, prob_1, prob_2))
        preds = np.argmax(probs, axis=0)
        return preds

    # TODO: Implement this method!
    # Calculates the neg log likelihood according to our result in problem 2.1
    def negative_log_likelihood(self, X, y):
        sum = 0
        for i, x_i in enumerate(X):
            if y[i] == 0:
                sum += self.class_0.logpdf(x_i) + np.log(self.priors[0])
            elif y[i] == 1:
                sum += self.class_1.logpdf(x_i) + np.log(self.priors[1])
            else:
                sum += self.class_2.logpdf(x_i) + np.log(self.priors[2])
        return -1 * sum
