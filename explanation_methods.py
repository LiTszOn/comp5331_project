import numpy as np
import keras.models
from keras.models import Model
import matplotlib.pyplot as plt
import keras.backend as K


class GradCAM:
    def __init__(self, model):
        # Node
        self.getMasks = K.function([model.inputs[0], model.inputs[1], model.inputs[2], 
                                    model.layers[0].input, model.layers[3].input, model.layers[6].input, model.layers[1].input],
                                    [self.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output), self.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)])
        # Edge
        self.getMasks_edge = K.function([model.inputs[0], model.inputs[1], model.inputs[2], 
                                        model.layers[0].input, model.layers[3].input, model.layers[6].input, model.layers[1].input], 
                                        [self.getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input), self.getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)])
      
    def getGradCamMask(self, output, act):
        '''
        Calculate the importance weight reported in GradCam
        '''
        alpha = K.squeeze(K.mean(K.gradients(output, act)[0], axis=1), 0)
        return K.squeeze(K.relu(K.sum(act*alpha, axis=2)), 0)

    def getGradCamMask_edge(self,output,act):
        '''
        Calculates the edge importance via gradient w.r.t. adj
        '''
        mask_adj = K.squeeze(K.relu(K.gradients(output,act)[0]), 0)
        return K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))  # set diagonal to be all 0

class EB:
    def __init__(self, model):
        mask0 = K.squeeze(K.sum(get_Pin(self, model, [1, 0]), axis=2), 0)
        mask0_edge = self.EB_MoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)
        
        mask1 = K.squeeze(K.sum(get_Pin(self, model, [0, 1]), axis=2), 0)
        mask1_edge = self.EB_MoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        self.getMasks = K.function([model.inputs[0],model.inputs[1],model.inputs[2], 
                        model.layers[0].input , model.layers[3].input, model.layers[6].input, model.layers[1].input],
                        [mask0,mask1])
        self.getMasks_edge = K.function([model.inputs[0],model.inputs[1],model.inputs[2], 
                            model.layers[0].input, model.layers[3].input, model.layers[6].input, model.layers[1].input],
                            [mask0_edge,mask1_edge])
        
    def get_Pin(self, model, vect):
        pLamda4 = self.EB_DenseLayer(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array(vect)))
        pdense3 = self.EB_GAPLayer(model.layers[-5].output,pLamda4)
        pLambda3 = self.EB_MoleculeDenseLayer(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2 = self.EB_MoleculeAdjLayer(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2 = self.EB_MoleculeDenseLayer(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1 = self.EB_MoleculeAdjLayer(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1 = self.EB_MoleculeDenseLayer(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin = self.EB_MoleculeAdjLayer(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        return pin

    def EB_DenseLayer(self, act_v, W, btmP_v):
        '''
        Calculate EB for a dense layer    
        '''
        pcond = K.tf.matmul(K.tf.diag(act_v), K.relu(W))
        pcond = pcond / (K.sum(pcond,axis=0) + 1e-5)
        return K.transpose(K.tf.matmul(pcond,K.expand_dims(btmP_v, 1)))

    def EB_MoleculeDenseLayer(self, act_v, W, btmP_v):
        '''
        Calculate EB for a molecule dense layer
        '''
        k,l = W.shape.as_list()
        pcond = K.tile(K.expand_dims(act_v,3),(1,1,1,l)) * K.relu(W)
        return K.mean(K.tile(K.expand_dims(btmP_v,2),(1,1,k,1)) * pcond, 3)

    def EB_GAPLayer(self, act_v, btmP_v):
        '''
        Calculate EB for a GAP layer
        '''
        pcond = act_v/(1e-5 + K.sum(act_v, axis=1))
        p = pcond * K.squeeze(btmP_v, 0)
        return p / (K.sum(p, axis=1) + 1e-5)

    def EB_MoleculeAdjLayer(self, act_v, A, btmP_v):
        '''
        Calculate EB for a Adj conv layer
        '''
        pcond = K.expand_dims(K.tf.matmul(A, K.squeeze(act_v, 0)),0)
        return pcond * btmP_v

    def EB_MoleculeEdge(self, act_v, A, btmP_v):
        '''
        Calculate EB for a Edge
        '''
        mask_adj = A*(K.squeeze(K.sum(btmP_v, axis=2), 0)) + A*K.tf.transpose(P)
        # set diagonal to be all 0
        return K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))
