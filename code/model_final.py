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
    train_data = np.load('../data/train.npz')
    eval_data = np.load('../data/eval.npz')
    seq_data = pd.read_csv('../data/seq_data.csv', names=['gene_id', 'sequence'])


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
        for gene in cell_data[:,0,0]:
            if num == 0:
                rowgene = seq_data.loc[seq_data['gene_id'] == gene]
                gene_seq = rowgene['sequence'].values[0]
                onehot_gene_seq = gene_one_hot_encoding(gene_seq)
                gene2seq[gene] = onehot_gene_seq
            train_genes.append(gene)
        train_inputs.append(hm_data)
        train_outputs.append(exp_values)

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)
    train_genes = np.asarray(train_genes)


    # Prepare Eval inputs in similar way
    eval_inputs = []
    eval_genes = []
    for num, cell in enumerate(eval_cells):
        print(cell)
        cell_data = eval_data[cell]
        hm_data = cell_data[:,:,1:6]
        eval_inputs.append(hm_data)
        for gene in cell_data[:,0,0]:
            if num == 0:
                rowgene = seq_data.loc[seq_data['gene_id'] == gene]
                gene_seq = rowgene['sequence'].values[0]
                onehot_gene_seq = gene_one_hot_encoding(gene_seq)
                gene2seq[gene] = onehot_gene_seq
            eval_genes.append(gene)

    eval_inputs = np.concatenate(eval_inputs, axis=0)

    return train_inputs, train_genes, gene2seq, train_outputs, eval_inputs, eval_genes, eval_data


# Define Models
class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        # tf.keras.layers.GaussianNoise(0.25),
        tf.keras.layers.Dense(212, activation='relu'),
        tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(212, activation='relu'),
        tf.keras.layers.Dense(500, activation='sigmoid'),
        tf.keras.layers.Reshape((100, 5))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class HMmodel(tf.keras.layers.Layer):
    def __init__(self):
        super(HMmodel, self).__init__()

        self.layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
        self.max_pool_1 = tf.keras.layers.MaxPool1D(5)

        self.layer_2 = tf.keras.layers.Conv1D(50,5,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
        self.max_pool_2 = tf.keras.layers.MaxPool1D(3)

        self.flatten = tf.keras.layers.Flatten()

        self.dropout_1 = tf.keras.layers.Dropout(0.5)

        self.dense_1 = tf.keras.layers.Dense(82)

    @tf.function
    def call(self, hm_batch):
        conv_1 = self.layer_1(hm_batch)
        mp_1 = self.max_pool_1(conv_1)
        conv_2 = self.layer_2(mp_1)
        mp_2 = self.max_pool_2(conv_2)
        flat = self.flatten(mp_2)
        drop = self.dropout_1(flat)
        dense = self.dense_1(drop)
        return dense


class SEQmodel(tf.keras.layers.Layer):
    def __init__(self):
        super(SEQmodel, self).__init__()
        
        self.layer_1 = tf.keras.layers.Conv1D(128,6,activation=tf.keras.layers.LeakyReLU(0.05), kernel_initializer='glorot_normal', padding="SAME")
        self.max_pool_1 = tf.keras.layers.MaxPool1D(32)

        self.flatten = tf.keras.layers.Flatten()

        self.dropout_1 = tf.keras.layers.Dropout(0.5)

        self.dense_1 = tf.keras.layers.Dense(82)

        
    @tf.function
    def call(self, seq_batch):
        conv_1 = self.layer_1(seq_batch)
        mp_1 = self.max_pool_1(conv_1)
        flat = self.flatten(mp_1)
        drop = self.dropout_1(flat)
        dense = self.dense_1(drop)
        return dense


class COMBmodel(tf.keras.Model):
    def __init__(self):
        super(COMBmodel, self).__init__()

        self.hm_model = HMmodel()
        self.seq_model = SEQmodel()

        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.Dense(36,activation=tf.keras.layers.LeakyReLU(0.05))
        # self.dense_2 = tf.keras.layers.Dense(24,activation=tf.keras.layers.LeakyReLU(0.05))
        self.dense_3 = tf.keras.layers.Dense(1,activation=None)
        self.dropout3 = tf.keras.layers.Dropout(0.1)

    
    def call(self, inputs):
        hm_batch, seq_batch = inputs
        hm_dense = self.hm_model(hm_batch)
        seq_dense = self.seq_model(seq_batch)
        combined = tf.keras.layers.concatenate([hm_dense, seq_dense])
        fully_con1 = self.dense_1(combined)
        con_drop = self.dropout3(fully_con1)
        output = self.dense_3(con_drop) 
        return output
    
    
    def loss(self, pred, true):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(true, pred)



def k_cross_validate_model(train_hm_inputs, train_genes, seq_dict, train_expression_vals, k):
    """
    method to run k-cross validation on the model
    @parma train_x - training inputs
    @param train_y - training labels
    @param k - split ratio int, data splits into (1 - 1/k) train, and 1/k test ratios
    """

    # calling the method to create validation curves
    model = COMBmodel()
    model.built = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    batch_size = 100
    num_epochs = 10
    for i in range(k):
        print('Running fold ' + str(i+1))
        #spliting the training and validation data
        eval_hm_inputs = train_hm_inputs[int(i*(1/k)*train_hm_inputs.shape[0]):int((i+1)*(1/k)*train_hm_inputs.shape[0])]
        eval_genes = train_genes[int(i*(1/k)*train_genes.shape[0]):int((i+1)*(1/k)*train_genes.shape[0])]
        eval_expression_vals = train_expression_vals[int(i*(1/k)*train_expression_vals.shape[0]):int((i+1)*(1/k)*train_expression_vals.shape[0])]
        train_hm_inputs = np.concatenate((train_hm_inputs[0:int(i*(1/k)*train_hm_inputs.shape[0])],train_hm_inputs[int((i+1)*(1/k)*train_hm_inputs.shape[0]):train_hm_inputs.shape[0]]), axis=0)
        train_genes = np.concatenate((train_genes[0:int(i*(1/k)*train_genes.shape[0])],train_genes[int((i+1)*(1/k)*train_genes.shape[0]):train_genes.shape[0]]), axis=0)
        train_expression_vals = np.concatenate((train_expression_vals[0:int(i*(1/k)*train_expression_vals.shape[0])],train_expression_vals[int((i+1)*(1/k)*train_expression_vals.shape[0]):train_expression_vals.shape[0]]), axis=0)

        for e in range(num_epochs):
            loss_list = []
            num_examples = np.shape(train_hm_inputs)[0]
            range_indicies = range(0, num_examples)
            shuffled_indicies = tf.random.shuffle(range_indicies)
            train_hm_inputs = tf.gather(train_hm_inputs, shuffled_indicies)
            train_genes = tf.gather(train_genes, shuffled_indicies).numpy().tolist()
            train_expression_vals = tf.gather(train_expression_vals, shuffled_indicies)
            for i in range(0, num_examples, batch_size):
                batch_hm_inputs = train_hm_inputs[i:i+batch_size,:,:]
                batch_genes = train_genes[i:i+batch_size]
                batch_onehot = [seq_dict[x] for x in batch_genes]
                batch_onehot_inputs = np.concatenate(batch_onehot, axis=0)
                batch_exp_vals = train_expression_vals[i:i+batch_size]
                with tf.GradientTape() as tape:
                    output = model.call((batch_hm_inputs, batch_onehot_inputs))
                    loss = model.loss(output, batch_exp_vals)
                    loss_list.append(loss)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss = np.mean(loss_list)
            print('epoch ' + str(e+1) + ': loss ' + str(loss))

        num_examples = len(train_genes)
        train_predictions = []
        for i in range(0, num_examples, batch_size):
            train_genes_batch = train_genes[i:i+batch_size]
            train_onehot = [seq_dict[x] for x in train_genes_batch]
            train_onehot_batch = np.concatenate(train_onehot, axis=0)
            train_hm_batch = train_hm_inputs[i:i+batch_size]
            preds = make_prediction(model, train_hm_batch, train_onehot_batch)
            train_predictions.append(preds)
        train_prediction = np.asarray([item for sublist in train_predictions for item in sublist])

        num_examples = len(eval_genes)
        test_predictions = []
        for i in range(0, num_examples, batch_size):
            eval_genes_batch = eval_genes[i:i+batch_size]
            eval_onehot = [seq_dict[x] for x in eval_genes_batch]
            eval_onehot_batch = np.concatenate(eval_onehot, axis=0)
            eval_hm_batch = eval_hm_inputs[i:i+batch_size]
            preds = make_prediction(model, eval_hm_batch, eval_onehot_batch)
            test_predictions.append(preds)
        test_prediction = np.asarray([item for sublist in test_predictions for item in sublist])

    create_val_plots(train_loss,val_loss)

def make_prediction(model, eval_inputs, eval_genes):
    """
    method to make predictions from the model based on the input data
    @param model - a tf.keras.Sequential model
    @param input_data - data to generate the prediction for
    @return - returns the model predictions
    """ 

    return model.predict((eval_inputs, eval_genes))
        
def evaluation_metrics(prediction, train_y):
    """
    method to calculate the final average pearsons correlation and average mean squared error
    @param predictions - predictions of the model
    @param train_y - actual labels to compare the prediction against
    """
    print(train_y)
    print(prediction)
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

def res_saliency_map(hm_input, dna_input, model):
    train_hm = tf.Variable(tf.expand_dims(hm_input, 0))
    train_hm = tf.cast(train_hm, tf.float32)
    train_seq = tf.Variable(dna_input)
    train_seq = tf.cast(train_seq, tf.float32)
    with tf.GradientTape() as hm_tape, tf.GradientTape() as seq_tape:
        seq_tape.watch(train_seq)
        pred = model.call((train_hm, train_seq))
        loss_wrt_hm = model.loss(pred, hm_input)
        loss_wrt_seq = model.loss(pred, dna_input)
    grads_hm = hm_tape.gradient(loss_wrt_hm, train_hm)
    grads_seq = seq_tape.gradient(loss_wrt_seq, train_seq)

    print(hm_input)
    
    fig, axes = plt.subplots(1,2,figsize=(28,14))
    i = axes[0].imshow(hm_input, cmap="jet", alpha=0.8)
    plt.colorbar(i)
    j = axes[1].imshow(np.squeeze(grads_hm) * hm_input, cmap="jet", alpha=0.8)
    plt.colorbar(j)
    fig.savefig('hm_saliency.png')

    print(grads_seq)
    print(dna_input)
    print(np.squeeze(grads_seq))

    plt.clf()
    fig, axes = plt.subplots(1,2,figsize=(28,45))
    i = axes[0].imshow(np.transpose(dna_input), cmap="jet", alpha=0.8, aspect='auto')
    plt.colorbar(i)
    j = axes[1].imshow(np.transpose(np.squeeze(grads_seq) * dna_input), cmap="jet", alpha=0.8, aspect='auto')
    plt.colorbar(j)
    fig.savefig('seq_saliency.png')

    plt.clf()
    plt.figure(figsize=(100,15))
    plt.plot(range(0,10000),np.squeeze(grads_seq[:,:,0],axis=0), label="A")
    plt.plot(range(0,10000),np.squeeze(grads_seq[:,:,1],axis=0), label="C")
    plt.plot(range(0,10000),np.squeeze(grads_seq[:,:,2],axis=0), label="T")
    plt.plot(range(0,10000),np.squeeze(grads_seq[:,:,3],axis=0), label="G")
    plt.title('Gradient Across DNA Seq')
    plt.xlabel('DNA Bases')
    plt.ylabel('Gradient')
    plt.legend(loc='upper right')
    plt.savefig('dna_grad.png')


def main():
    # Keys to npzfile of train & eval
    train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
    'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
    'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    # Call get_data() to read in all of the data
    train_hm_inputs, train_genes, seq_dict, train_expression_vals, eval_hm_inputs, eval_genes, eval_data = get_data(train_cells,eval_cells)

    # autoencoder = Autoencoder(64)
    # autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    # autoencoder.fit(train_hm_inputs, train_hm_inputs, batch_size=250, epochs=10, shuffle=True)

    # train_hm_inputs = autoencoder.predict(train_hm_inputs)

    model = COMBmodel()
    model.built = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    batch_size = 100
    num_epochs = 10

    for e in range(num_epochs):
        loss_list = []
        num_examples = np.shape(train_hm_inputs)[0]
        range_indicies = range(0, num_examples)
        shuffled_indicies = tf.random.shuffle(range_indicies)
        train_hm_inputs = tf.gather(train_hm_inputs, shuffled_indicies)
        train_genes = tf.gather(train_genes, shuffled_indicies).numpy().tolist()
        train_expression_vals = tf.gather(train_expression_vals, shuffled_indicies)
        for i in range(0, num_examples, batch_size):
            batch_hm_inputs = train_hm_inputs[i:i+batch_size,:,:]
            batch_genes = train_genes[i:i+batch_size]
            batch_onehot = [seq_dict[x] for x in batch_genes]
            batch_onehot_inputs = np.concatenate(batch_onehot, axis=0)
            batch_exp_vals = train_expression_vals[i:i+batch_size]
            with tf.GradientTape() as tape:
                output = model.call((batch_hm_inputs, batch_onehot_inputs))
                loss = model.loss(output, batch_exp_vals)
                loss_list.append(loss)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = np.mean(loss_list)
        print('epoch ' + str(e+1) + ': loss ' + str(loss))

    res_saliency_map(train_hm_inputs[0,:,:], seq_dict[train_genes[0]], model)

    num_examples = len(train_genes)
    train_predictions = []
    for i in range(0, num_examples, batch_size):
        train_genes_batch = train_genes[i:i+batch_size]
        train_onehot = [seq_dict[x] for x in train_genes_batch]
        train_onehot_batch = np.concatenate(train_onehot, axis=0)
        train_hm_batch = train_hm_inputs[i:i+batch_size]
        preds = make_prediction(model, train_hm_batch, train_onehot_batch)
        train_predictions.append(preds)
    train_prediction = np.asarray([item for sublist in train_predictions for item in sublist])

    num_examples = len(eval_genes)
    test_predictions = []
    for i in range(0, num_examples, batch_size):
        eval_genes_batch = eval_genes[i:i+batch_size]
        eval_onehot = [seq_dict[x] for x in eval_genes_batch]
        eval_onehot_batch = np.concatenate(eval_onehot, axis=0)
        eval_hm_batch = eval_hm_inputs[i:i+batch_size]
        preds = make_prediction(model, eval_hm_batch, eval_onehot_batch)
        test_predictions.append(preds)
    test_prediction = np.asarray([item for sublist in test_predictions for item in sublist])

    # # Call evaluation_metrics to generate the average pearson's correlation and average final MSE for the training sets
    pearsons,loss = evaluation_metrics(train_prediction, train_expression_vals.numpy())
    print("Pearsons correlation co-efficent: ", pearsons)
    print("Average final Mean squared error loss: ", loss)

    plt.clf()
    plt.figure(figsize=(10,8))
    plt.hist(train_expression_vals.numpy().flatten(), bins=100, alpha=0.5, label='true vals', color='blue')
    plt.hist(train_prediction, bins=100, alpha=0.5, label='predictions', color='green')
    plt.legend(loc='upper right')
    plt.ylabel('Counts')
    plt.xlabel('Gene Expression Values')
    plt.savefig("expression_counts.png")


    # Call generate csv to submit the csv to kaggle
    generate_csv(test_prediction.flatten(), eval_cells, eval_data)


if __name__ == '__main__':
    main()
