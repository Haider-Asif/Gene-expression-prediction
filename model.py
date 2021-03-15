import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pandas as pd

def get_data(train_cells,eval_cells):
    # Load data
    train_data = np.load('../train.npz')
    eval_data = np.load('../eval.npz')

    # Combine Train Data to use information from all cells
    train_inputs = [] # Input histone mark data
    train_outputs = [] # Correct expression value
    for cell in train_cells:
        cell_data = train_data[cell]
        hm_data = cell_data[:,:,1:6]
        exp_values = cell_data[:,0,6]
        train_inputs.append(hm_data)
        train_outputs.append(exp_values)

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_outputs = np.concatenate(train_outputs, axis=0)

    # Prepare Eval inputs in similar way
    eval_inputs = []
    for cell in eval_cells:
        cell_data = eval_data[cell]
        hm_data = cell_data[:,:,1:6]
        eval_inputs.append(hm_data)

    eval_inputs = np.concatenate(eval_inputs, axis=0)

    return train_inputs, train_outputs, eval_inputs,eval_data

def k_cross_validate_model(train_x, train_y, k):
    for i in range(k):
        validation_x = train_x[int(i*(1/k)*train_x.shape[0]):int((i+1)*(1/k)*train_x.shape[0])]
        validation_y = train_y[int(i*(1/k)*train_y.shape[0]):int((i+1)*(1/k)*train_y.shape[0])]
        training_x = np.concatenate((train_x[0:int(i*(1/k)*train_x.shape[0])],train_x[int((i+1)*(1/k)*train_x.shape[0]):train_x.shape[0]]), axis=0)
        training_y = np.concatenate((train_y[0:int(i*(1/k)*train_y.shape[0])],train_y[int((i+1)*(1/k)*train_y.shape[0]):train_y.shape[0]]), axis=0)
        model = tf.keras.Sequential()
        layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
        batch_norm_1 = tf.keras.layers.BatchNormalization()
        max_pool_1 = tf.keras.layers.MaxPool1D(5)

        flatten = tf.keras.layers.Flatten()
        dropout1 = tf.keras.layers.Dropout(0.5)
        dropout2 = tf.keras.layers.Dropout(0.3)
        Dense_1 = tf.keras.layers.Dense(625,activation=tf.keras.layers.LeakyReLU(0.05))
        Dense_2 = tf.keras.layers.Dense(125,activation=tf.keras.layers.LeakyReLU(0.05))
        Dense_3 = tf.keras.layers.Dense(25,activation=tf.keras.layers.LeakyReLU(0.03))
        Dense_4 = tf.keras.layers.Dense(1,activation=None)
        model.add(layer_1)
        model.add(batch_norm_1)
        model.add(max_pool_1)
        # model.add(layer_2)
        # model.add(batch_norm_2)
        # model.add(max_pool_2)
        model.add(flatten)
        # model.add(dropout)
        model.add(Dense_1)
        model.add(dropout1)
        model.add(Dense_2)
        # model.add(dropout)
        # model.add(Dense_3)
        model.add(dropout2)
        model.add(Dense_4)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
        model.fit(x=training_x, y=training_y, batch_size=100, epochs=0, validation_data=(validation_x,validation_y), shuffle=True)

def train_model(train_x, train_y):
    """
    Implements and trains the model using a cross-validation scheme with MSE loss
    param train_x: the training inputs
    param train_y: the training labels
    return: a trained model
    """
    model = tf.keras.Sequential()
    layer_1 = tf.keras.layers.Conv1D(50,10,activation=tf.keras.layers.LeakyReLU(0.05), padding="SAME")
    batch_norm_1 = tf.keras.layers.BatchNormalization()
    max_pool_1 = tf.keras.layers.MaxPool1D(5)

    lstm = tf.keras.layers.LSTM(200)

    flatten = tf.keras.layers.Flatten()
    dropout = tf.keras.layers.Dropout(0.5)
    dropout2 = tf.keras.layers.Dropout(0.3)
    Dense_1 = tf.keras.layers.Dense(625,activation=tf.keras.layers.LeakyReLU(0.05))
    Dense_2 = tf.keras.layers.Dense(125,activation=tf.keras.layers.LeakyReLU(0.05))
    Dense_3 = tf.keras.layers.Dense(25,activation=tf.keras.layers.LeakyReLU(0.03))
    Dense_4 = tf.keras.layers.Dense(1,activation=None)
    model.add(layer_1)
    model.add(batch_norm_1)
    model.add(max_pool_1)
    # model.add(layer_2)
    # model.add(batch_norm_2)
    # model.add(max_pool_2)
    model.add(flatten)
    model.add(lstm)
    # model.add(dropout)
    model.add(Dense_1)
    model.add(dropout)
    model.add(Dense_2)
    # model.add(dropout)
    # model.add(Dense_3)
    model.add(dropout)
    model.add(Dense_4)
    k_cross_validate_model(train_x,train_y,4)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
    model.fit(x=train_x, y=train_y, batch_size=100, epochs=20,shuffle=True)
    return model

def make_prediction(model, input_data):
    """
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    """
    return model.predict(input_data)
        
def evaluation_metrics(prediction, train_y):
    r,_ = pearsonr(train_y.flatten(), prediction.flatten())
    loss = mean_squared_error(train_y.flatten(), prediction.flatten())
    return r,loss

def generate_csv(eval_preds,eval_cells,eval_data):
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

    submit_df.to_csv('../sample_submission.csv', header=True, index=False, index_label=False)

def main():
        # Keys to npzfile of train & eval
    train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058', 
    'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097', 
    'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']
    # Call get_data() to read in all of the data
    train_x, train_y, test_x, test_data = get_data(train_cells,eval_cells)
    # Call remove_borders() to properly modify the training labels
    # Call train_model() to train the model
    model = train_model(train_x,train_y)
    # Visualize several of the training and test matrix patches
    test_prediction = make_prediction(model, test_x)

    train_prediction = make_prediction(model,train_x)

    pearsons,loss = evaluation_metrics(train_prediction, train_y)
    print("Pearsons correlation co-efficent: ", pearsons)
    print("Average final Mean squared error loss: ",loss)
    generate_csv(test_prediction.flatten(),eval_cells, test_data)


if __name__ == '__main__':
    main()