import numpy as np
import deepchem as dc
from deepchem.splits import ScaffoldSplitter
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Softmax, Lambda
from keras import backend as K
from keras.initializers import RandomNormal
import utils
import loss_function


def partition_dataset(smiles):
    #can we use s
    # print(f"smiles is {smiles}")
    Xs = np.zeros(len(smiles))
    Ys = np.ones(len(smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(smiles)),ids=smiles)
    partitioner = ScaffoldSplitter()
    train_partition, val_partition, test_partition = partitioner.split(dataset)
    return train_partition, val_partition, test_partition

def get_EBpin(model, vect):
    pLamda4=utils.ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array(vect)))
    pdense3=utils.ebGAP(model.layers[-5].output,pLamda4)
    pLambda3=utils.ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
    pdense2=utils.ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
    pLambda2=utils.ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
    pdense1=utils.ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
    pLambda1=utils.ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
    pin=utils.ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
    
def build_gcn(explanation_method="GCAM"):
    """
    Keras GCN for graph classification
    """
    first_input = Input(shape=(1,None,None), batch_shape=(1,None,None))
    second_input = Input(shape=(1, None, None), batch_shape=(1, None, None))
    third_input = Input(shape=(1, None, None), batch_shape=(1, None, None))
    last_input = Input(shape=(1,None,75), batch_shape=(1,None,75))
    main_matrix = Input(shape=(1, None, 1), batch_shape=(1, None, 1))
    edge_matrix = Input(shape=(1, None, None), batch_shape=(1, None, None))
    adj_matrix = Input(shape=(1, None, None), batch_shape=(1, None, None))

    initial_weight_first = RandomNormal(mean=0.0, stddev=0.1)
    initial_bias_first = RandomNormal(mean=0.0, stddev=0.1)
    initial_weight_second = RandomNormal(mean=0.0, stddev=0.1)
    initial_bias_second = RandomNormal(mean=0.0, stddev=0.1)
    initial_weight_third = RandomNormal(mean=0.0, stddev=0.1)
    initial_bias_third = RandomNormal(mean=0.0, stddev=0.1)
    initial_weight_logit = RandomNormal(mean=0.0, stddev=0.1)
    initial_bias_logit = RandomNormal(mean=0.0, stddev=0.1)
        
    first_output = Dense(256, activation='relu', kernel_initializer=initial_weight_first, bias_initializer=initial_bias_first, use_bias=True)(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([first_input, last_input]))
    second_output = Dense(128, activation='relu', kernel_initializer=initial_weight_second, bias_initializer=initial_bias_second, use_bias=True)(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([second_input, first_output]))
    third_output = Dense(64, activation='relu', kernel_initializer=initial_weight_third, bias_initializer=initial_bias_third, use_bias=True)(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([third_input, second_output]))
    logits = Dense(2, activation='linear', kernel_initializer=initial_weight_logit, bias_initializer=initial_bias_logit, use_bias=True)(Lambda(lambda y: K.squeeze(y, 1))(Lambda(lambda x: K.tf.reduce_mean(x, axis=1, keepdims=True))(third_output)))
    fina_output = Softmax()(logits)
    model = Model(inputs=[main_matrix, edge_matrix, adj_matrix, first_input, second_input, third_input, last_input], outputs=fina_output)

    if explanation_method =='GCAM':
        # node mask
        maskh0 = utils.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1 = utils.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        node_mask = K.stack([maskh0, maskh1], axis=0)
        
        # edge mask
        maskh0_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)
        edge_mask = K.stack([maskh0_edge, maskh1_edge], axis=0)

    elif explanation_method =='EB':
        # node mask
        mask0 = K.squeeze(K.sum(get_EBpin(model, [1, 0]), axis=2),0)
        edge_mask0 = utils.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)
        node_mask = K.stack([mask0, mask1], axis=0)

        # edge mask
        mask1 = K.squeeze(K.sum(get_EBpin(model, [0, 1]), axis=2),0)
        edge_mask1 = utils.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)
        edge_mask = K.stack([edge_mask0, edge_mask1], axis=0)

    print("------------- model.compile ------------------")
    model.compile(optimizer=Adam(lr=0.001),
                  loss=loss_function.call_loss_function_of_GNES(K, ['sparsity', 'consistency'], logits, adj_matrix, node_mask, edge_mask, main_matrix, edge_matrix))
    return model
