import load_data as ld
import preprocessing
import training
import utils
import config
import os
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
# config_original = config.load_config("BBBP")
model = training.build_gcn()

####original

# raw_data_original = utils.load_data(raw_data_file_path)
# data_original = utils.preprocess(raw_data_original)
# human_data_original = utils.load_human_data(config_original, "BBBP")
# print(f"original train_dict: {human_data_original['train']}")
# print(f"original test_dict: {human_data_original['test']}")
# print(f"original val_dict: {human_data_original['val']}")
# assert dict4train == human_data_original['train'], f"dict not equal {dict4train} vs {human_data_original['train']}"
# assert dict4test == human_data_original['test'], f"dict not equal {dict4test} vs {human_data_original['test']}"
# assert dict4val == human_data_original['val'], f"dict not equal {dict4val} vs {human_data_original['val']}"
# inds_original = utils.partition_train_val_test(raw_data_original["smiles"], "BBBP")
# train_inds_original = inds_original["train_inds"]
# val_inds_original = inds_original["val_inds"]
# test_inds_original = inds_original["test_inds"]
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
# train_inds = train_partition
# val_inds = val_partition

# model_original = utils.keras_gcn(config)
# model = utils.build_gcn(config)
loss = utils.gcn_train(model, data, config['num_epochs'], train_partition, val_partition, save_path, human_data)

# model = keras_gcn(training.build_gcn(config_original))
# model.load_weights(save_path)

train_eval = utils.evaluate(model, data, train_partition, human_data['train'], config['exp_method'], human_eval=True)
test_eval = utils.evaluate(model, data, test_partition, human_data['test'], config['exp_method'], human_eval=True)
val_eval = utils.evaluate(model, data, val_partition, human_data['val'], config['exp_method'], human_eval=True)

eval_metric = {0:{"train": train_eval,"test": test_eval,"val": val_eval}}

for metric in ["roc_auc", "avg_precision", "node_mse", "node_mae", "edge_mse", "edge_mae"]:
    print(metric)
    for split in ["train", "val  ", "test "]:
        res = utils.print_eval_avg(eval_metric, split.strip(), metric)
        print(split + " " + res)
    # print()