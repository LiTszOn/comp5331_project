import numpy as np
import deepchem as dc
from deepchem.splits import ScaffoldSplitter
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Softmax, Lambda
from keras import backend as K
import utils
def partition_dataset(smiles):
    #can we use s
    # print(f"smiles is {smiles}")
    Xs = np.zeros(len(smiles))
    Ys = np.ones(len(smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(smiles)),ids=smiles)
    partitioner = ScaffoldSplitter()
    train_partition, val_partition, test_partition = partitioner.split(dataset)
    return train_partition, val_partition, test_partition

def build_gcn(exp_method_name="GCAM"):
    """
    Keras GCN for graph classification
    """
    # d = config['d']
    # L1 = config['L1']
    # L2 = config['L2']
    # L3 = config['L3']
    # N = config['N']
    # print(f"N is {N}")
    # num_classes = config['num_classes']
    # print(f"batch_size is {batch_size}")
    # exp_method_name = config['exp_method']

    # currently the implementation of the pipeline only support batch_size = 1
    # assert batch_size == 1, "Batch size != 1 Not Implemented!"

    first_input = Input(shape=(1,None,None), batch_shape=(1,None,None))
    second_input = Input(shape=(1, None, None), batch_shape=(1, None, None))
    third_input = Input(shape=(1, None, None), batch_shape=(1, None, None))
    last_input = Input(shape=(1,None,75), batch_shape=(1,None,75))
    main_matrix = Input(shape=(1, None, 1), batch_shape=(1, None, 1))
    edge_matrix = Input(shape=(1, None, None), batch_shape=(1, None, None))
    adj_matrix = Input(shape=(1, None, None), batch_shape=(1, None, None))

    # h1 = dense(L1)(K.tf.matmul(A_batch1, X_batch))
    first_output = Dense(256, activation='relu')(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([first_input, last_input]))
    second_output = Dense(128, activation='relu')(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([second_input, first_output]))
    third_output = Dense(64, activation='relu')(Lambda(lambda x: K.tf.matmul(x[0],x[1]))([third_input, second_output]))
    logits = Dense(2, activation='relu')(Lambda(lambda y: K.squeeze(y, 1))(Lambda(lambda x: K.tf.reduce_mean(x, axis=1, keepdims=True))(third_output)))
    fina_output = Softmax()(logits)
    model = Model(inputs=[main_matrix, edge_matrix, adj_matrix, first_input, second_input, third_input, last_input], outputs=fina_output)

    if exp_method_name =='GCAM':
        # node mask
        maskh0 = utils.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1 = utils.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        node_mask = K.stack([maskh0, maskh1], axis=0)
        # node_mask = [maskh0,maskh1]

        # edge mask
        maskh0_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)
        edge_mask = K.stack([maskh0_edge, maskh1_edge], axis=0)

    elif exp_method_name =='EB':
        pLamda4=utils.ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([1,0])))
        pdense3=utils.ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=utils.ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=utils.ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=utils.ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=utils.ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=utils.ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=utils.ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask0=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask0 = utils.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        pLamda4=utils.ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([0,1])))
        pdense3=utils.ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=utils.ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=utils.ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=utils.ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=utils.ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=utils.ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask1=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask1 = utils.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        # node mask
        node_mask = K.stack([mask0, mask1], axis=0)
        # edge mask
        edge_mask = K.stack([edge_mask0, edge_mask1], axis=0)
    else:
        print('Unknown exp method name. use GCAM as default')
        # node mask
        maskh0 = utils.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1 = utils.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)
        node_mask = K.stack([maskh0, maskh1], axis=0)
        # node_mask = [maskh0,maskh1]

        # edge mask
        maskh0_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge = utils.getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)
        edge_mask = K.stack([maskh0_edge, maskh1_edge], axis=0)
    model.compile(optimizer=Adam(lr=0.001),
                  loss=custom_loss(['sparsity', 'consistency'], logits, adj_matrix, node_mask, edge_mask, main_matrix, edge_matrix))
    return model
    
def custom_loss(reg_list, logits, A, node_mask, edge_mask, M, E):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):

        loss=K.mean(K.binary_crossentropy(y_true, logits, from_logits=True), axis=-1)
        node_s = K.abs(K.gather(node_mask, K.argmax(y_true, axis=-1)))
        edge_s = K.abs(K.gather(edge_mask, K.argmax(y_true, axis=-1)))
        ones = K.ones(K.shape(node_s))
        edge_s = K.squeeze(edge_s * A, axis=0)

        if 'sparsity' in reg_list:
            print('adding sparsity regularization')
            loss += 0.001 * (K.mean(node_s) + K.mean(edge_s))
        if 'consistency' in reg_list:
            print('adding consistency regularization')
            pair_diff = K.tf.matrix_band_part(node_s[..., None] - K.tf.transpose(node_s[..., None]), 0, -1)
            # we use below term to avoid trivial solution when minimizing consistency loss (i.e. edge_s = 0)
            conn = - K.mean(K.log(K.tf.matmul(edge_s, K.transpose(ones)) + 1e-6))
            loss += 0.1 * (K.mean(edge_s * K.square(pair_diff)) + conn)

        # human node importance supervision
        loss += 1 * K.max(M) * K.mean(K.abs(node_s - M))
        # human edge importance supervision
        loss += 1 * K.max(E) * K.mean(K.abs(edge_s - E))
        return loss
    # Return a function
    return loss