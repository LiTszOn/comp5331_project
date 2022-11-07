import load_data as ld
import preprocessing
import training
import utils
import config
import os
raw_data_file_path="data/BBBP.csv"
# human_data_file_path=""
nodes, edges, label, smiles = ld.load_and_preprocess_chemical_data(raw_data_file_path)
dict4train,dict4test,dict4val = ld.load_manual_annotation()#analogue to train_inds,test_inds and val_inds in 740-742

labels_one_hot,molecules_features,edges,normalise_edges = preprocessing.preprocess_main(nodes, edges, label, smiles)

# inds = partition_train_val_test(smiles, dataset)
train_partition, val_partition, test_partition = training.partition_dataset(smiles)
config_original = config.load_config("BBBP")
model = training.build_gcn(config_original)

####original

raw_data_original = utils.load_data(raw_data_file_path)
data_original = utils.preprocess(raw_data_original)
human_data_original = utils.load_human_data(config_original, "BBBP")
inds_original = utils.partition_train_val_test(raw_data_original["smiles"], "BBBP")
train_inds_original = inds_original["train_inds"]
val_inds_original = inds_original["val_inds"]
test_inds_original = inds_original["test_inds"]
config = config.load_config("BBBP")
human_data = {"train": dict4train,
            "val": dict4test,
            "test":dict4val}
data = {'labels_one_hot': labels_one_hot,
            'node_features': molecules_features,
            'adjs': edges,
            'norm_adjs': data_original['norm_adjs']}
model_out_fn = "GNES_{}.h5".format("BBBP".lower())
save_path = os.path.join(config["saved_models_dir"], model_out_fn)
# train_inds = train_partition
# val_inds = val_partition

model_original = utils.keras_gcn(config)
loss, accuracy = utils.gcn_train(model, data_original, config['num_epochs'], train_partition, val_partition, save_path, human_data)

# model = keras_gcn(config)
# model.load_weights(save_path)

# train_eval = evaluate(model, data, train_inds, human_data['train'], config['exp_method'], human_eval=True)
# test_eval = evaluate(model, data, test_inds, human_data['test'], config['exp_method'], human_eval=True)
# val_eval = evaluate(model, data, val_inds, human_data['val'], config['exp_method'], human_eval=True)