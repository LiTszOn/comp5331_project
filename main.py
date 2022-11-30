import load_data as ld
import preprocessing
import training
import utils
import config
import os
import pandas as pd
import random
import tensorflow as tf
import numpy as np


seed = 10
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
raw_data_file_path="data/BBBP.csv"
# human_data_file_path=""
nodes, edges, label, smiles = ld.load_and_preprocess_chemical_data(raw_data_file_path)
dict4train,dict4test,dict4val = ld.load_manual_annotation()#analogue to train_inds,test_inds and val_inds in 740-742

labels_one_hot,molecules_features,edges,normalise_edges = preprocessing.preprocess_main(nodes, edges, label, smiles)

# inds = partition_train_val_test(smiles, dataset)
train_partition, val_partition, test_partition = training.partition_dataset(smiles)
model = training.build_gcn()

config = config.load_config("BBBP")
human_data = {"train": dict4train,
            "val": dict4val,
            "test":dict4test}
data = {'labels_one_hot': labels_one_hot,
            'node_features': molecules_features,
            'adjs': edges,
            # 'norm_adjs': data_original['norm_adjs']}
            'norm_adjs': normalise_edges}
model_out_fn = "GNES_{}.h5".format("BBBP".lower())
save_path = os.path.join(config["saved_models_dir"], model_out_fn)
loss = utils.gcn_train(model, data, config['num_epochs'], train_partition, val_partition, save_path, human_data)

train_eval = utils.evaluate(model, data, train_partition, human_data['train'], config['exp_method'], human_eval=True)
test_eval = utils.evaluate(model, data, test_partition, human_data['test'], config['exp_method'], human_eval=True)
val_eval = utils.evaluate(model, data, val_partition, human_data['val'], config['exp_method'], human_eval=True)

eval_metric = {0:{"train": train_eval,"test": test_eval,"val": val_eval}}

trainList = []
valList = []
testList = []
index = ["roc_auc", "avg_precision", "node_mse", "node_mae", "edge_mse", "edge_mae"]
evaldf = pd.DataFrame(index = index)
for metric in index:
    trainList.append(utils.print_eval_avg(eval_metric, 'train', metric))
    valList.append(utils.print_eval_avg(eval_metric, 'val', metric))
    testList.append(utils.print_eval_avg(eval_metric, 'test', metric))
evaldf['train'] = trainList
evaldf['val'] = valList
evaldf['test'] = testList
evaldf.to_csv('saved_models/evaluation.csv')

for split in ["test"]:
    utils.plot_roc_curve(split, eval_metric)