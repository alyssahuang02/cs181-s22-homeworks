import numpy as np

#from T2_P3 import X_test

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    def distance(self, x1, x2):
        return ((x1[0]-x2[0])/3)**2 + (x1[1]-x2[1])**2

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = np.zeros(len(X_pred))
        index = 0
        for x_test in X_pred:
            arr = []
            for i in range(len(self.X)):
                x_i = self.X[i]
                arr.append((self.distance(x_i, x_test), i))
            arr.sort() # sort by distance values in increasing order

            votes = []
            for i in range(self.K):
                votes.append(self.y[int(arr[i][1])])
            
            preds[index] = max(set(votes), key=votes.count)
            index += 1

        return preds

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y