#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
from operator import index
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20
        
    arr = [np.ones(xx.shape)]
    if part == "a":
        for i in range(1, 6):
            arr.append(xx**i)
    
    elif part == "b":
        miu_j = 1960
        while miu_j <= 2010:
            cur = np.exp((-(xx-miu_j)**2)/25)
            arr.append(cur)
            miu_j += 5

    elif part == "c":
        for i in range(1, 6):
            i = float (i)
            arr.append(np.cos(xx/i))

    else:
        for i in range(1, 26):
            i = float (i)
            arr.append(np.cos(xx/i))
    
    res = np.vstack(arr)
    return res.T

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years)) # basic basis

print("Losses for years")
# Plot the data and the regression line.
for part in ['a', 'b', 'c', 'd']:
    years_basis = make_basis(years, part=part, is_years=True) # basis for years (24, ?)
    w = find_weights(years_basis, republican_counts)
    grid_X = make_basis(grid_years, part=part, is_years=True) # basis for smoother plotting (200, ?)
    grid_Yhat  = np.dot(grid_X, w)
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Basis " + part.upper() + ": Year v. Number of Republicans in the Senate")
    plt.tight_layout()
    plt.savefig("1_basis_"+part+".png", facecolor="white")
    plt.show()

    predictions = np.dot(years_basis, w) # 24 vector to calculate loss
    print(sum((predictions - republican_counts)**2))

print()

# convert data to restrict to years before 1985
sunspot_counts = sunspot_counts[years<last_year]
republican_counts = republican_counts[years<last_year]
max_val = max(sunspot_counts)
grid_sunspots = np.linspace(0, max_val, 200)

print("Losses for sunspots")

for part in ['a', 'c', 'd']:
    sunspots_basis = make_basis(sunspot_counts, part=part, is_years=False) # basis for sunspots (24, ?)
    w = find_weights(sunspots_basis, republican_counts)
    grid_X = make_basis(grid_sunspots, part=part, is_years=False) # basis for smoother plotting (200, ?)
    grid_Yhat  = np.dot(grid_X, w)

    plt.plot(sunspot_counts, republican_counts, 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Basis " + part.upper() + ": Number of Sunspots v. Number of Republicans in the Senate")
    plt.tight_layout()
    plt.savefig("2_basis_"+part+".png", facecolor="white")
    plt.show()
    plt.show()

    predictions = np.dot(sunspots_basis, w) # 24 vector to calculate loss
    print(sum((predictions - republican_counts)**2))

# TODO: plot and report sum of squared error for each basis
