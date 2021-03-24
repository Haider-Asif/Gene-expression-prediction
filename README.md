# CS1850-midterm-1
Private reposity for CS1850 Midterm 1 Code for the team Waka Waka - Isaac Nathoo and Muhammad Haider Asif

# How to build

To build the appropriate setup to run the code, please follow these steps:
1) Download the data zip file into your computer.
2) Drag and drop (move) the train.npz and the eval.npz files into the data directory
3) To make sure that the zip file is not pushed please don't add it to the git directory, .npz files will automatically be ignored.

As a final check make sure before you run, you navigate to the code directory, and the .npz files are properly placed in the data directory.

# How to run the model

To run the model please navigate into the code directory of the repository and open a terminal. Inside the terminal please run the following command:

python3 model.py

This will run the model which will start with k-fold validation and then move on to training the model properly, it will then calculate the pearons' correlation coefficent for the training predictions and labels, and will also calculate the final average MSE. It will then plot the train, and k-cross validation curves, and generate the csv required for the kaggle submission.
