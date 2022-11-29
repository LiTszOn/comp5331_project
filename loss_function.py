#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Congying
@Created: 2022/10/20
------------------------------------------
@Modify: 2022/10/20
------------------------------------------
@Description:

the loss function of GNES
"""

def call_loss_function_of_GNES(K, reg_list, logits, A, node_mask, edge_mask, M, E):
    def loss_function_of_GNES(y_true, y_pred):
        node_s = K.abs(K.gather(node_mask, K.argmax(y_true, axis=-1)))
        edge_s = K.abs(K.gather(edge_mask, K.argmax(y_true, axis=-1)))
        ones = K.ones(K.shape(node_s))
        edge_s = K.squeeze(edge_s * A, axis=0)

        """ inference accuracy loss  """
        loss_of_inference_accuracy =K.mean(K.binary_crossentropy(y_true, logits, from_logits=True), axis=-1)
        
        """ explainability loss  """
        loss_of_explainability_node = 1 * K.max(M) * K.mean(K.abs(node_s - M))
        loss_of_explainability_edge = 1 * K.max(E) * K.mean(K.abs(edge_s - E))

        """ regularization loss  """
        if 'sparsity' in reg_list:
            print('adding sparsity regularization')
            loss_of_sparsity_regualrization= 0.001 * (K.mean(node_s) + K.mean(edge_s))
        if 'consistency' in reg_list:
            print('adding consistency regularization')
            pair_diff = K.tf.matrix_band_part(node_s[..., None] - K.tf.transpose(node_s[..., None]), 0, -1)
            # we use below term to avoid trivial solution when minimizing consistency loss (i.e. edge_s = 0)
            conn = - K.mean(K.log(K.tf.matmul(edge_s, K.transpose(ones)) + 1e-6))
            loss_of_explanation_regualrization = 0.1 * (K.mean(edge_s * K.square(pair_diff)) + conn)

        all_loss = loss_of_inference_accuracy+loss_of_explainability_node+loss_of_explainability_edge+loss_of_sparsity_regualrization+loss_of_explanation_regualrization

        return all_loss
    return loss_function_of_GNES


