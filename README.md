# CS1850-project
Private reposity for CS1850 Midterm Project and Final Project Code for team WakaWaka - Isaac Nathoo and Muhammad Haider Asif

Note: the model for the midterm is model_midterm.py and the model for the final is model_final.py

# How to build

To build the appropriate setup to run the code, please follow these steps:
1) Download the data zip files into your computer.
2) Drag and drop (move) the train.npz, the eval.npz, and the seq_data.csv files into the data directory
3) To ensure that the zip file is not pushed, please do not add it to the git directory, .npz and .csv files will automatically be ignored.

As a final check make sure before you run, you navigate to the code directory, and the .npz and .csv files are properly placed in the data directory.

# How to run the model

To run the model please navigate into the code directory of the repository and open a terminal. Inside the terminal please run the following command:

python3 model_final.py

This will run the model which will start with k-fold validation and then move on to training the model properly, it will then calculate the Pearson Correlation Coefficent for the training predictions and labels, and will also calculate the final average MSE. It will then plot the training loss and k-cross validation curves, and generate the csv required for the Kaggle submission. It will also produce interpretation plots for the histone modification data and DNA sequence data of 10 random genes. Lastly, it will plot a histogram showing the gene expression value distributions.

# Results

Our code will generate five outputs: the first (train_val_plot.png) is a graph showing the average training and validation losses across the epochs for k-fold cross validation, the second is a graph (expression_counts.png) showing the distributions of the true expression values and the predicted expression values, the third is a plot (hm_saliency.png) showing the histone modification input heatmap and input\*gradient heatmap, the fourth is a plot (dna_grad.png) showing the gradients for each of the four nucleotide bases across the length of the DNA sequence, and the fifth (sample_submission.csv) is a CSV with the predicted values. These outputs will be saved in the results directory.
