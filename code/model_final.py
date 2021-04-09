import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd

def one_hot_encoding(seq_array):
    """
    :param seq_array: list of DNA sequences
    :return: np array of one-hot encodings of input DNA sequences
    """
    nuc2id = {'A' : 0, 'C' : 1, 'T' : 2, 'G' : 3}
    onehot_array = np.zeros((len(seq_array), 10000, 4))
    for seq_num, seq in enumerate(seq_array):
        for seq_idx, nucleotide in enumerate(seq):
            nuc_idx = nuc2id[nucleotide]
            onehot_array[seq_num, seq_idx, nuc_idx] = 1
    
    return onehot_array


def gene_one_hot_encoding(single_seq):
    """
    :param single_seq: single DNA sequence
    :return: np array of one-hot encoding of input DNA sequence
    """
    nuc2id = {'A' : 0, 'C' : 1, 'T' : 2, 'G' : 3, 'N' : 0}
    onehot_array = np.zeros((1, 10000, 4))
    for seq_idx, nucleotide in enumerate(single_seq):
        nuc_idx = nuc2id[nucleotide]
        onehot_array[0, seq_idx, nuc_idx] = 1
    
    return onehot_array


def get_data(train_cells,eval_cells):
    """
    Method to get the training & eval data from the npz files stored in the directory outside the code
    @param train_cells - keys to the train npz file
    @param eval_cells - keys to the eval npz file
    @return train_inputs, train_outputs, eval_inputs,eval_data - clean training data inputs, training data labels, eval data inputs, and all of eval data
    """

    # Load data
    train_data = np.load('../../train.npz')
    eval_data = np.load('../../eval.npz')
    seq_data = pd.read_csv('../../seq_data.csv', names=['gene_id', 'sequence'])


    # Combine Train Data to use information from all cells
    train_inputs = [] # Input histone mark data
    train_outputs = [] # Correct expression value
    train_seqs = []
    train_genes = []
    gene2seq = {}
    for num, cell in enumerate(train_cells):
        print(cell)
        cell_data = train_data[cell]
        hm_data = cell_data[:,:,1:6]
        exp_values = cell_data[:,0,6]
        # gene_ids = cell_data[:,0,0]
        # cell_seqs = seq_data[seq_data['gene_id'].isin(gene_ids)]['sequence'].tolist()
        # train_seqs.append(cell_seqs[:2])
        for gene in cell_data[:,0,0]:
            if num == 0:
                rowgene = seq_data.loc[seq_data['gene_id'] == gene]
                gene_seq = rowgene['sequence'].values[0]
                onehot_gene_seq = gene_one_hot_encoding(gene_seq)
                gene2seq[gene] = onehot_gene_seq
            train_genes.append(gene)
        # for gene in cell_data[:,0,0]:
            # rowgene = seq_data.loc[seq_data['gene_id'] == gene]
            # ind_seq = rowgene['sequence'].values[0]
            # train_seqs.append(ind_seq[:10])
        train_inputs.append(hm_data)
        train_outputs.append(exp_values)

    train_inputs = np.concatenate(train_inputs, axis=0)
    print(np.shape(train_inputs))
    train_outputs = np.concatenate(train_outputs, axis=0)
    print(np.shape(train_outputs))
    print(len(gene2seq))
    print(np.shape(gene2seq[5]))
    # train_seqs = np.concatenate(train_seqs, axis=0)
    # print(np.shape(train_seqs))
    # train_seqs = one_hot_encoding(train_seqs)
    # print(np.shape(train_seqs))
    train_genes = np.asarray(train_genes)


    # Prepare Eval inputs in similar way
    eval_inputs = []
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data = cell_data[:,:,1:6]
        eval_inputs.append(hm_data)

    eval_inputs = np.concatenate(eval_inputs, axis=0)

    return train_inputs, train_genes, gene2seq, train_outputs, eval_inputs, eval_data


# Define Models
class HMmodel(tf.keras.layers.Layer):
    def __init__(self):
        super(HMmodel, self).__init__()

        self.layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
        self.max_pool_1 = tf.keras.layers.MaxPool1D(5)

        self.layer_2 = tf.keras.layers.Conv1D(50,5,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=3)
        self.max_pool_2 = tf.keras.layers.MaxPool1D(3)

        self.layer_3 = tf.keras.layers.Conv1D(50,3,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=2)
        self.max_pool_3 = tf.keras.layers.MaxPool1D(3)

        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, hm_batch):
        conv_1 = self.layer_1(hm_batch)
        mp_1 = self.max_pool_1(conv_1)
        conv_2 = self.layer_2(mp_1)
        mp_2 = self.max_pool_2(conv_2)
        conv_3 = self.layer_3(mp_2)
        mp_3 = self.max_pool_3(conv_3)
        flat = self.flatten(mp_3)
        return flat


class SEQmodel(tf.keras.layers.Layer):
    def __init__(self):
        super(SEQmodel, self).__init__()
        
        self.layer_1 = tf.keras.layers.Conv1D(128,6,activation='relu', kernel_initializer='glorot_normal', padding="SAME")
        self.max_pool_1 = tf.keras.layers.MaxPool1D(30)

        self.layer_2 = tf.keras.layers.Conv1D(32,9,activation='relu', kernel_initializer='glorot_normal', padding="SAME")
        self.max_pool_2 = tf.keras.layers.MaxPool1D(10)

        self.flatten = tf.keras.layers.Flatten()

        
    @tf.function
    def call(self, seq_batch):
        conv_1 = self.layer_1(seq_batch)
        mp_1 = self.max_pool_1(conv_1)
        conv_2 = self.layer_2(mp_1)
        mp_2 = self.max_pool_2(conv_2)
        flat = self.flatten(mp_2)
        return flat


class COMBmodel(tf.keras.Model):
    def __init__(self):
        super(COMBmodel, self).__init__()

        self.hm_model = HMmodel()
        self.seq_model = SEQmodel()

        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense_1 = tf.keras.layers.Dense(100,activation=tf.keras.layers.LeakyReLU(0.05))
        self.dense_2 = tf.keras.layers.Dense(24,activation=tf.keras.layers.LeakyReLU(0.05))
        self.dense_3 = tf.keras.layers.Dense(1,activation=None)

    
    def call(self, hm_batch, seq_batch):
        hm_flat = self.hm_model(hm_batch)
        hm_drop = self.dropout1(hm_flat)
        seq_flat = self.seq_model(seq_batch)
        seq_drop = self.dropout1(seq_flat)
        combined = tf.keras.layers.concatenate([hm_drop, seq_drop])
        fully_con1 = self.dense_1(combined)
        fully_con2 = self.dense_2(fully_con1)
        output = self.dense_3(fully_con2) 
        return output
    
    
    def loss(self, pred, true):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(true, pred)


def train(model, optimizer, batch_size, hm_training, gene_training, seq_dict, expression_vals):
    loss_list = []
    num_examples = np.shape(hm_training)[0]
    range_indicies = range(0, num_examples)
    shuffled_indicies = tf.random.shuffle(range_indicies)
    shuffled_hm_inputs = tf.gather(hm_training, shuffled_indicies)
    shuffled_genes = tf.gather(hm_training, shuffled_indicies)
    shuffled_expression_vals = tf.gather(expression_vals, shuffled_indicies)
    for i in range(0, num_examples, batch_size):
        batch_hm_inputs = shuffled_hm_inputs[i:i+batch_size,:,:]
        batch_genes = shuffled_genes[i:i+batch_size]
        batch_onehot = [seq_dict[x] for x in batch_genes]
        batch_onehot_inputs = np.concatenate(batch_onehot, axis=0)
        batch_exp_vals = shuffled_expression_vals[i:i+batch_size]
        with tf.GradientTape() as tape:
            output = model.call(batch_hm_inputs, batch_onehot_inputs)
            loss = model.loss(output, batch_exp_vals)
            loss_list.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    avg_loss = np.mean(loss_list)
    return avg_loss


# model = COMBmodel()
# model.built = True
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
# batch_size = 250
# num_epochs = 20

# for e in range(num_epochs):
#     loss = train(model, optimizer, batch_size, dataset)
#     print('epoch ' + str(e) + ': loss ' + str(loss))



def k_cross_validate_model(train_x, train_y, k):
    """
    method to run k-cross validation on the model
    @parma train_x - training inputs
    @param train_y - training labels
    @param k - split ratio int, data splits into (1 - 1/k) train, and 1/k test ratios
    """

    val_loss = []
    train_loss = []
    for i in range(k):
        print('Running fold ' + str(i+1))

        #spliting the training and validation data
        validation_x = train_x[int(i*(1/k)*train_x.shape[0]):int((i+1)*(1/k)*train_x.shape[0])]
        validation_y = train_y[int(i*(1/k)*train_y.shape[0]):int((i+1)*(1/k)*train_y.shape[0])]
        training_x = np.concatenate((train_x[0:int(i*(1/k)*train_x.shape[0])],train_x[int((i+1)*(1/k)*train_x.shape[0]):train_x.shape[0]]), axis=0)
        training_y = np.concatenate((train_y[0:int(i*(1/k)*train_y.shape[0])],train_y[int((i+1)*(1/k)*train_y.shape[0]):train_y.shape[0]]), axis=0)
        
        # constructing the validation model
        model = tf.keras.Sequential()
        layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
        max_pool_1 = tf.keras.layers.MaxPool1D(5)

        layer_2 = tf.keras.layers.Conv1D(50,5,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=3)
        max_pool_2 = tf.keras.layers.MaxPool1D(3)

        layer_3 = tf.keras.layers.Conv1D(50,3,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=2)
        max_pool_3 = tf.keras.layers.MaxPool1D(3)

        flatten = tf.keras.layers.Flatten()

        dropout1 = tf.keras.layers.Dropout(0.3)
        Dense_1 = tf.keras.layers.Dense(50,activation=tf.keras.layers.LeakyReLU(0.05))
        Dense_2 = tf.keras.layers.Dense(10,activation=tf.keras.layers.LeakyReLU(0.05))
        Dense_3 = tf.keras.layers.Dense(1,activation=None)

        model.add(layer_1)
        model.add(max_pool_1)

        model.add(layer_2)
        model.add(max_pool_2)

        model.add(layer_3)
        model.add(max_pool_3)

        model.add(flatten)

        model.add(dropout1)

        model.add(Dense_1)

        model.add(Dense_2)
    
        model.add(Dense_3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
        history = model.fit(x=training_x, y=training_y, batch_size=250, epochs=15, validation_data=(validation_x,validation_y), shuffle=True)
        val_loss.append(history.history["val_loss"])
        train_loss.append(history.history["loss"])
    # calling the method to create validation curves
    create_val_plots(train_loss,val_loss)

def train_model(train_x, train_y):
    """
    method that Implements and trains the model using a cross-validation scheme with MSE loss
    @parma train_x - training inputs
    @param train_y - training labels
    @return model - trained model
    """

    # constructing the model
    model = tf.keras.Sequential()
    layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
    max_pool_1 = tf.keras.layers.MaxPool1D(5)

    layer_2 = tf.keras.layers.Conv1D(50,5,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=3)
    max_pool_2 = tf.keras.layers.MaxPool1D(3)

    layer_3 = tf.keras.layers.Conv1D(50,3,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME", dilation_rate=2)
    max_pool_3 = tf.keras.layers.MaxPool1D(3)

    flatten = tf.keras.layers.Flatten()

    dropout1 = tf.keras.layers.Dropout(0.3)
    Dense_1 = tf.keras.layers.Dense(50,activation=tf.keras.layers.LeakyReLU(0.05))
    Dense_2 = tf.keras.layers.Dense(10,activation=tf.keras.layers.LeakyReLU(0.05))
    Dense_3 = tf.keras.layers.Dense(1,activation=None)
    model.add(layer_1)
    model.add(max_pool_1)

    model.add(layer_2)
    model.add(max_pool_2)

    model.add(layer_3)
    model.add(max_pool_3)

    model.add(flatten)

    model.add(dropout1)
    model.add(Dense_1)

    model.add(Dense_2)
 
    model.add(Dense_3)
    # running k-cross validation
    k_cross_validate_model(train_x,train_y,4)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    history = model.fit(x=train_x, y=train_y, batch_size=250, epochs=15,shuffle=True)
    # printing the model summary
    model.summary()
    # creating the training loss plot
    create_train_plots(history.history["loss"])
    return model

def make_prediction(model, input_data):
    """
    method to make predictions from the model based on the input data
    @param model - a tf.keras.Sequential model
    @param input_data - data to generate the prediction for
    @return - returns the model predictions
    """ 

    return model.predict(input_data)
        
def evaluation_metrics(prediction, train_y):
    """
    method to calculate the final average pearsons correlation and average mean squared error
    @param predictions - predictions of the model
    @param train_y - actual labels to compare the prediction against
    """

    r,_ = pearsonr(train_y.flatten(), prediction.flatten())
    loss = mean_squared_error(train_y.flatten(), prediction.flatten())
    return r,loss

def generate_csv(eval_preds,eval_cells,eval_data):
    """
    method to generate the csv with the predictions for the eval set to be submitted to kaggle
    @param eval_preds - predictions of the model
    @param eval_cells - keys to the eval npz file
    @param eval_data - all of the eval data
    """

    cell_list = []
    gene_list = []
    example_eval_preds = eval_preds
    for cell in eval_cells:
        cell_data = eval_data[cell]
        cell_list.extend([cell]*len(cell_data))
        genes = cell_data[:,0,0].astype('int32')
        gene_list.extend(genes)

    id_column = [] # ID is {cell type}_{gene id}
    for idx in range(len(eval_preds)):
        id_column.append(f'{cell_list[idx]}_{gene_list[idx]}')

    df_data = {'id': id_column, 'expression' : example_eval_preds}
    submit_df = pd.DataFrame(data=df_data)

    submit_df.to_csv('../results/sample_submission.csv', header=True, index=False, index_label=False)

def create_train_plots(training_losses):
    """
    method to create a plot for just the training loss per epoch
    @param training_losses - array of training losses (1 entry per epoch)
    """
    x = [i for i in range(len(training_losses))]
    plt.clf()
    plt.plot(x, training_losses)
    plt.title('Training Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, len(x)+1, 1))
    plt.savefig('../results/train_plot.png')

def create_val_plots(training_losses,validation_losses):
    """
    method to create a plot for the training loss per epoch, and validation loss for the cross-validation scheme
    @param training_losses - array of training losses (1 entry per epoch)
    @param validation_losses - array of validation losses (1 entry per epoch)
    """

    z = [i for i in range(len(validation_losses[0]))]
    plt.clf()
    for k in range(len(validation_losses)):
        plt.plot(z, training_losses[k],label="train_loss_fold"+str(k+1))
        plt.plot(z, validation_losses[k],label="val_loss_fold"+str(k+1))
    plt.title('Cross Fold Validation Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, len(validation_losses[0])+1, 1))
    plt.legend() 
    plt.savefig('../results/val_plot.png')

def main():
        # Keys to npzfile of train & eval
    train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
    'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
    'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    # Call get_data() to read in all of the data
    train_hm_inputs, train_genes, seq_dict, train_expression_vals, eval_inputs, eval_data = get_data(train_cells,eval_cells)

    model = COMBmodel()
    model.built = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    batch_size = 250
    num_epochs = 20

    for e in range(num_epochs):
        loss = train(model, optimizer, batch_size, train_hm_inputs, train_genes, seq_dict, train_expression_vals)
        print('epoch ' + str(e) + ': loss ' + str(loss))

    # Call remove_borders() to properly modify the training labels
    # Call train_model() to train the model
    # model = train_model(train_x,train_y)
    # # Call make_prediction to generate predictions for raining and eval sets
    # test_prediction = make_prediction(model, test_x)

    # train_prediction = make_prediction(model,train_x)

    # # Call evaluation_metrics to generate the average pearson's correlation and average final MSE for the training sets
    # pearsons,loss = evaluation_metrics(train_prediction, train_y)
    # print("Pearsons correlation co-efficent: ", pearsons)
    # print("Average final Mean squared error loss: ",loss)

    # # Call generate csv to submit the csv to kaggle
    # generate_csv(test_prediction.flatten(), eval_cells, test_data)


if __name__ == '__main__':
    main()
