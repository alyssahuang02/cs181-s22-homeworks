import numpy as np
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000
        self.losses = np.zeros(self.runs)

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    
    def one_hot_encode(self, y):
        arr = []
        for elem in y:
            row = np.zeros(3)
            row[elem] = 1
            arr.append(row)
        arr = np.vstack(arr)
        return arr

    # TODO: Implement this method!
    def fit(self, X, y):
        bias = np.matrix(np.ones(X.shape[0])).T
        X = np.hstack((X, bias))
        self.W = np.random.rand(X.shape[1], 3)
        for i in range(self.runs):
            y_hat = self.softmax(np.dot(X, self.W))
            one_hot_y = self.one_hot_encode(y)
            y_delta = y_hat - one_hot_y
            grad = X.T.dot(y_delta) + self.lam * self.W
            self.W = self.W - self.eta*2*grad
            # add here to some loss function
            loss = 0
            for elem in range(y_hat.shape[0]):
                loss -= np.dot(one_hot_y[elem], np.squeeze(np.asarray(np.log(y_hat[elem]))))
            self.losses[i] = loss

    def softmax(self, x):
        res = []
        for row in x:
            denom = np.sum(np.exp(row))
            new_row = np.exp(row) / denom
            res.append(new_row)
        res = np.vstack(res)
        return res

    # TODO: Implement this method!
    def predict(self, X_pred):
        bias = np.matrix(np.ones(X_pred.shape[0])).T
        X_pred = np.hstack((X_pred, bias))
        y_hat = self.softmax(np.dot(X_pred, self.W))
        res = np.argmax(y_hat, axis=1)
        return np.array(res)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        # print(self.losses[-1])
        plt.plot(range(self.runs), self.losses)
        plt.title("Losses over Iterations")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Negative Log-Likelihood  Loss")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
