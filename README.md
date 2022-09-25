# Perceptron-ANN
Aurora image classification


### Directories
- `yes_aurora`: directory with images labeled as with aurora 
- `no_aurora`: directory with images labeled as without aurora 
- `not_known`: directory with unlabeled images (typically with some sign of aurora but hard to make a decision)

### Files
- `Color_histogram.py`: code for perceptron training
- `inputs.txt`: color histograms for labeled images
- `features.txt`: normalized color histograms with labels and filenames included
- `unknown_results.txt`: results of using final model on unlabeled images, includes label and filename

## Overview
The code reads in images from ‘yes_aurora’ and ‘no_aurora’ directories and extracts the color histograms which are stored in the file ‘input.txt’. The histograms are then normalized by min-max normalization, prepended with its actual label, and appended with its file name, which are then stored in an array called ‘features’, which is also saved as a text file ‘features.txt’. The code to visualize these text files commented in the code base. 

Throughout the code, the number of epoch is 1000, and weights are randomly initialized between -1 and 1 before each training process. The data are randomly split into train and validation by 80-20. Training error and accuracy plots for batch size 100 and various learning rates (0.1, 0.01, 0.001, 1.5) are plotted, then the plots for learning rate 0.1 and various batch sizes (20, 50, 100) are plotted to find the best training parameter. Then measures for the validations results are printed for each (learning rate, batch size) pair, which include accuracy, precision, and recall. 

Finally, learning rate 0.1 and batch size 100 were used to produce the final model. The resulting model is also tested on unclassified dataset and results are saved to ‘unknown_results.txt’ where one can see the filenames (in order) and the corresponding classifications.
