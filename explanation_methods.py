import numpy as np
import keras.models
from keras.models import Model
import matplotlib.pyplot as plt
import keras.backend as K


class GradCAM:
    def __init__(self, model):

        maskh0=self.getGradCamMask(model.layers[-2].output[0,0],model.layers[-5].output)
        maskh1=self.getGradCamMask(model.layers[-2].output[0,1],model.layers[-5].output)

        maskh0_edge=self.getGradCamMask_edge(model.layers[-2].output[0,0],model.layers[6].input)
        maskh1_edge=self.getGradCamMask_edge(model.layers[-2].output[0,1],model.layers[6].input)

        getMasks=K.function([model.inputs[0],model.inputs[1],model.inputs[2], model.layers[0].input , model.layers[3].input, model.layers[6].input, model.layers[1].input],[maskh0,maskh1])
        getMasks_edge = K.function([model.inputs[0],model.inputs[1],model.inputs[2], model.layers[0].input , model.layers[3].input, model.layers[6].input, model.layers[1].input], [maskh0_edge, maskh1_edge])
        self.getMasks = getMasks
        self.getMasks_edge = getMasks_edge
      
    def getGradCamMask(self,output,activation):
        '''
        This function calculates the importance weight reported in GradCam
        Input:
            ouput: The class output 
            activation: activation that we will take gradient with respect to
        '''
        temp_grad= K.gradients(output,activation)
        grad=K.gradients(output,activation)[0]
        alpha=K.squeeze(K.mean(grad,axis=1),0)
        mask=K.squeeze(K.relu(K.sum(activation*alpha,axis=2)),0)
        return mask

    def getGradCamMask_edge(self,output,activation):
        '''
        This function calculates the edge importance via gradient w.r.t. adj
        Input:
            ouput: The class output
            activation: activation that we will take gradient with respect to
        '''
        grad_adj=K.gradients(output,activation)[0]
        mask_adj=K.squeeze(K.relu(grad_adj),0)
        # set diagonal to be all 0
        mask_adj = K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))
        return mask_adj

class EB:
    def __init__(self, model):
        pLamda4=self.ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([1,0])))
        pdense3=self.ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=self.ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=self.ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=self.ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=self.ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=self.ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=self.ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask0=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask0 = self.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        pLamda4=self.ebDense(K.squeeze(model.layers[-3].output,0),model.layers[-2].weights[0],K.variable(np.array([0,1])))
        pdense3=self.ebGAP(model.layers[-5].output,pLamda4)
        pLambda3=self.ebMoleculeDense(model.layers[-6].output,model.layers[-5].weights[0],pdense3)
        pdense2=self.ebMoleculeAdj(model.layers[-7].output,K.squeeze(model.layers[6].input,0),pLambda3)
        pLambda2=self.ebMoleculeDense(model.layers[-9].output,model.layers[-7].weights[0],pdense2)
        pdense1=self.ebMoleculeAdj(model.layers[-10].output,K.squeeze(model.layers[3].input,0),pLambda2)
        pLambda1=self.ebMoleculeDense(model.layers[-12].output,model.layers[-10].weights[0],pdense1)
        pin=self.ebMoleculeAdj(model.layers[-13].output,K.squeeze(model.layers[0].input,0),pLambda1)
        mask1=K.squeeze(K.sum(pin,axis=2),0)
        edge_mask1 = self.ebMoleculeEdge(model.layers[-13].output, K.squeeze(model.layers[0].input, 0), pLambda1)

        getMasks=K.function([model.inputs[0],model.inputs[1],model.inputs[2], model.layers[0].input , model.layers[3].input, model.layers[6].input, model.layers[1].input],[mask0,mask1])
        getMasks_edge=K.function([model.inputs[0],model.inputs[1],model.inputs[2], model.layers[0].input, model.layers[3].input, model.layers[6].input, model.layers[1].input],[edge_mask0,edge_mask1])

        self.getMasks = getMasks
        self.getMasks_edge = getMasks_edge
        
    def ebDense(self, activations, W, bottomP):
        '''
        This function calculates eb for a dense layer
        Input: 
            activations: d-dimensional vector
            W: Weights dxk-dimensional matrix
            bottomP: k-dimensional probability vector
        Output:
            p: the probability of activation d-dimensional vector    
        '''
        Wrelu=K.relu(W)
        pcond=K.tf.matmul(K.tf.diag(activations),Wrelu)
        pcond=pcond/(K.sum(pcond,axis=0) + 1e-5)
        return K.transpose(K.tf.matmul(pcond,K.expand_dims(bottomP,1)))

    def ebMoleculeDense(self, activations, W, bottomP):
        '''
        This function calculates eb for a dense layer
        Input: 
            activations: 1x?xK
            W: Weights dxk-dimensional matrix KxL
            bottomP: probability matrix 1x?xL
        Output:
            p: probability matrix 1x?xK
        '''
        k,l=W.shape.as_list()
        Wrelu=K.relu(W)
        pcond=K.tile(K.expand_dims(activations,3),(1,1,1,l))*Wrelu
        p=K.mean(K.tile(K.expand_dims(bottomP,2),(1,1,k,1))*pcond,3)
        return p

    def ebGAP(self, activations,bottomP):
        '''
        This function calculates eb for GAP layer
        Input: 
            activations: 1x?xK
            bottomP: probability matrix 1xK
        Output:
            p: probability matrix 1x?xK
        '''
        epsilon=1e-5
        pcond=activations/(epsilon+K.sum(activations,axis=1))
        p=pcond*K.squeeze(bottomP,0)
        p=p/(K.sum(p,axis=1)+epsilon)
        return p

    def ebMoleculeAdj(self,activations,A,bottomP):
        '''
        This function calculates eb for a Adj conv layer
        Input: 
            activations: 1x?xK
            A: Adjacency ?x?
            bottomP: probability matrix 1x?xK
        Output:
            p: probability matrix 1x?xK
        '''
        pcond=K.expand_dims(K.tf.matmul(A,K.squeeze(activations,0)),0)
        p=pcond*bottomP
        return p

    def ebMoleculeEdge(self,activations,A,bottomP):
        # P: 1x?
        P = K.squeeze(K.sum(bottomP, axis=2), 0)

        mask_adj = A*P + A*K.tf.transpose(P)

        # set diagonal to be all 0
        mask_adj = K.tf.linalg.set_diag(mask_adj, K.tf.zeros([K.tf.shape(mask_adj)[0], ]))
        return mask_adj
