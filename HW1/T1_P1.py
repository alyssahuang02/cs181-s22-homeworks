#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

# computes loss with formula found in problem 1.1
def compute_loss(tau):
    loss = 0
    for i in range(len(data)):
        x_i, y_i = data[i]
        minisum = 0
        for j in range(len(data)):
            x_j, y_j = data[j]
            if i != j:
                minisum += y_j * np.exp(-1*np.transpose(x_i - x_j)*(x_i-x_j)/tau)
        loss += (y_i - minisum) ** 2
    return loss

for tau in (0.01, 2, 100):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))

# kernel function as defined in the problem
def kern(x, x_prime, tau):
    return np.exp(-1*np.transpose(x - x_prime)*(x-x_prime)/tau)

# function f as defined in the problem
def f(x, tau):
    sum = 0
    for x_i, y_i in data:
        sum += kern(x_i, x, tau) * y_i
    return sum

def plot():
    x = np.arange(0, 12.1, 0.1)
    y1 = f(x, 0.01)
    y2 = f(x, 2)
    y3 = f(x, 100)
    plt.figure()
    plt.plot(x, y1, color='red', label='tau 0.01')
    plt.plot(x, y2, color='blue', label='tau 2')
    plt.plot(x, y3, color='green', label='tau 100')
    plt.title("Function by Varying Lengthscale (Tau) Values")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plot.png", facecolor="white")
    plt.show()

plot()

# computes the gradient as found in part 1.2
def compute_grad(tau):
    grad = 0
    for i in range(len(data)):
        x_i, y_i = data[i]
        minisum = 0
        for j in range(len(data)):
            x_j, y_j = data[j]
            if i != j:
                minisum += y_j * np.exp(-1*np.transpose(x_i - x_j)*(x_i-x_j)/tau)
        part_one = y_i - minisum
        part_two = 0
        for j in range(len(data)):
            x_j, y_j = data[j]
            if i != j:
                chain_one = y_j * np.exp(-1*np.transpose(x_i - x_j)*(x_i-x_j)/tau)
                part_two += chain_one*np.transpose(x_i - x_j)*(x_i-x_j)
        grad += part_one * part_two
    return -2*grad/(tau**2)

# performs gradient descent starting from tau=2    
def gradient_descent(learning_rate=0.01, iterations=1000):
    tau = 2
    for i in range(iterations):
        tau = tau - learning_rate * compute_grad(tau)
    return tau

print(gradient_descent())