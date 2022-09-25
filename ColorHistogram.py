import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd

#img = cv2.imread('./images/AurNo.jpg')
#plt.figure()
#plt.xlim([0,256])
#plt.ylim([0,20000])
color = ('r','g','b')

#histogram.append(series)
#series,bins = np.histogram(img.ravel(),256,[0,256]) #histogram for all channels
#plt.plot(series,color = col)
#plt.show()
#img
    

# Encode single image into 256+256+256 histogram vector    
def encode_img(img_filename):
    img = cv2.imread(img_filename)
    hist = ()
    for i,col in enumerate(color):
        series = cv2.calcHist([img],[i],None,[256],[0,256])
        hist = np.append(hist, series)

    return hist

# save histograms to input.txt
# normalize, append name and label, save to features array
features=[]
with open("inputs.txt", "w") as f:
    directory = './yes_aurora'
    for filename in os.listdir(directory):
        img_filename = os.path.join(directory, filename)
        hist = encode_img(img_filename)
        f.write(",".join(hist.astype(np.str)))
        f.write("\n")
        hist = np.true_divide(hist-np.min(hist), np.max(hist)-np.min(hist)) # minmax normalize histogram
        hist = np.append(1, hist)
        hist = np.append(hist, "#" + img_filename)
        features.append(list(hist))
    directory = './no_aurora'
    for filename in os.listdir(directory):
        # ignore hidden file
        if not filename.startswith('.'):
            img_filename = os.path.join(directory, filename)
            hist = encode_img(img_filename)
            f.write(",".join(hist.astype(np.str)))
            f.write("\n")
            hist = np.true_divide(hist-np.min(hist), np.max(hist)-np.min(hist))
            hist = np.append(-1, hist)
            hist = np.append(hist, "#" + img_filename)
            features.append(list(hist))

# convert list to array, each row = feature vector of an image      
features=np.asarray(features)
            
#inputs = pd.read_csv("inputs.txt", header=None) # use this line of code to get color histogram of all labeled imgs

np.savetxt("features.txt", features, delimiter=",", fmt='%s')
#pd.read_csv("feature.txt", header=None) to access # use this line of code to view features in dataframe

rows = features.shape[0]
train_idx = np.random.choice(rows, math.floor(rows*0.8), replace=False)
features_train = features[train_idx]
features_validate = np.delete(features, train_idx, 0)



# for one epoch, record accuracy and training error for each batch
# W is vector of size (767,)
# features has 770 columns
def perceptron_train(features, W, batch_size, mu, n_epoch):
    rows = features.shape[0]
    train_errors = []
    accuracies = []
    
    
    for i in np.arange(n_epoch):
        batch_idx = np.random.randint(rows, size=batch_size) 
        batch_features = features[batch_idx]
        
        label = batch_features[:,0].astype(np.float)
        X = np.delete(batch_features, [0, -1],1)# remove label and name column
        X0 = np.ones(shape=(batch_size,1))
        X = np.append(X0, X, axis=1) # add X0 column
        X = X.astype(np.float) # cast as float

        activation = np.dot(X, W)
        prediction = np.sign(activation)
        accuracy = (prediction==label)
        accuracies.append(np.sum(accuracy)/batch_size)
        
        error = np.subtract(prediction, label)
        W_t = np.dot(X.transpose(), error) / batch_size
        W_t = np.multiply(mu, W_t)
        W = np.subtract(W, W_t)
        
        E = np.true_divide(np.power(error, 2), batch_size)
        train_errors.append(np.sum(E))
    
    #accuracies = accuracies/batch_size
        
    return W, accuracies, train_errors


def perceptron_validate(features, W):
    rows = features.shape[0]
    
    label = features[:,0].astype(np.float)
    X = np.delete(features, [0,-1],1)# remove label and name column
    X0 = np.ones(shape=(rows,1))
    X = np.append(X0, X, axis=1) # add X0 column
    X = X.astype(np.float) # cast as float
    
    activation = np.dot(X, W)
    prediction = np.sign(activation)
    accuracy = (prediction==label)
    accuracy = np.sum(accuracy)/rows
    
    true = prediction[prediction==label] # extract correct predictions 
    true_pos = np.sum(true==1) # number of correct yes's
    positive = np.sum(prediction==1) # number of yes predictions
    precision = true_pos / positive # correct yes / (# predict yes)
    
    recall = true_pos / np.sum(label==1) # correct yes /  (# actual yes)
    
    compare_features = np.insert(features, 2, prediction, axis=1) #insert prediction col after label col
    
    return accuracy, precision, recall, compare_features
    
#part 2
n_epoch = 1000
mus = [0.1, 0.01, 0.001, 1.5]
batch_sizes = [20, 50, 100]
#W = np.ones(shape = (769,1))
W = np.random.uniform(-1, 1, 769) # generate (769,) random numbers between -1 and 1

# try batch_size 50, all learning rates
plt.figure(1)
plt.title("Training error")
plt.figure(2)
plt.title("Accuracy")


for mu in mus:
    W = np.random.uniform(-1, 1, 769)
    W, accuracies, train_errors = perceptron_train(features_train, W, 50, mu, n_epoch)
    plt.figure(1)
    plt.plot(np.arange(n_epoch), train_errors, label = "%f"%mu)
    plt.figure(2)
    plt.plot(np.arange(n_epoch), accuracies, label = "%f"%mu)

plt.figure(1)
plt.legend(loc='upper center')
plt.figure(2)
plt.legend(loc='upper center')
plt.show()

# try learning rate 0.001, all batch sizes
plt.figure(1)
plt.title("Training error")
plt.figure(2)
plt.title("Accuracy")


for batch_size in batch_sizes:
    W = np.random.uniform(-1, 1, 769)
    W, accuracies, train_errors = perceptron_train(features_train, W, batch_size, 0.001, n_epoch)
    plt.figure(1)
    plt.plot(np.arange(n_epoch), train_errors, label = "%d"%batch_size)
    plt.figure(2)
    plt.plot(np.arange(n_epoch), accuracies, label = "%d"%batch_size)
    
plt.figure(1)
plt.legend(loc='upper center')
plt.figure(2)
plt.legend(loc='upper center')
plt.show()

# part 3
# see recall, precision etc for all combo of mu and batch_size
for mu in mus:
    for batch_size in batch_sizes:
        W = np.random.uniform(-1, 1, 769)
        W, accuracies, train_errors = perceptron_train(features_train, W, batch_size, mu, n_epoch)
        accuracy, precision, recall, compare_features = perceptron_validate(features_validate, W)
        print("Validation mu %f, batch_size %d, accuracy: %f, precision: %f, recall: %f" %(mu, batch_size, accuracy, precision, recall))


# part 4
unknown_features=[]
directory = './not_known'
for filename in os.listdir(directory):
    img_filename = os.path.join(directory, filename)
    hist = encode_img(img_filename)
    hist = np.true_divide(hist-np.min(hist), np.max(hist)-np.min(hist)) # minmax normalize histogram
    hist = np.append(hist, "#" + img_filename)
    unknown_features.append(list(hist))
unknown_features = np.asarray(unknown_features)


W = np.random.uniform(-1, 1, 769) 
W, accuracies, train_errors = perceptron_train(features_train, W, 50, 0.001, n_epoch)

def perceptron_test(features, W):
    rows = features.shape[0]
    
    names = features[:, -1]
    X = np.delete(features, -1,1)# remove name column
    X0 = np.ones(shape=(rows,1))
    X = np.append(X0, X, axis=1) # add X0 column
    X = X.astype(np.float) # cast as float
    
    activation = np.dot(X, W)
    prediction = np.sign(activation)
    compare = np.stack((prediction, names), axis=1) # return prediction and name col
    return compare
    
    
compare = perceptron_test(unknown_features, W)
np.savetxt("unknown_results.txt", compare, delimiter=",", fmt='%s')
