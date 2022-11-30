import os
import numpy as np
import json
import sys
sys.path.append("../")
from config import load_config
from utils import (load_data, preprocess, partition_train_val_test, keras_gcn, load_human_data)
from plot_utils import (draw_chem_activations, plot_image_grid,
                        create_figs, create_im_arrs)
from explanation_methods import (GradCAM, EB)

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

save_masks = True
config = load_config("BBBP")
path = os.path.join(config['data_dir'], config['data_fn'])
csv_data = load_data(path)
dataProcess = preprocess(csv_data)
chemicals = csv_data["smiles"]

saved_model = "GNES_bbbp.h5"
saved_path =  os.path.join(config["saved_models_dir"], saved_model)
model = keras_gcn(config)
model.load_weights(saved_path)
num_classes = 2

labels_dict = {0: 'Not BBBP', 1:'BBBP'}
#Can choose some data to visualize
data_points = [628, 593, 492]
chemicals_list = chemicals[data_points].tolist()
index_chem_dict = {}
for data in data_points:
    data_int = int(data)
    index_chem_dict[data_int] = chemicals[data]
chem_index_dict = get_swap_dict(index_chem_dict)

num_data = len(data_points)

# Gather data for viz
selected_data = {}

for x, data in dataProcess.items():
    feature_data = [data[i] for i in data_points]
    selected_data[x] = feature_data

gcam = GradCAM(model)
eb = EB(model)

methods = [gcam, eb]
method_names = ["GCAM", "EB"]

N = len(selected_data['norm_adjs'])
results = []
text = []

mask_dict = {}  # {method: smile: {ground_truth, predicted, node_importance}}
for i in range(N):
    Adjs = selected_data['adjs'][i][np.newaxis, :, :]
    A_arr = selected_data['norm_adjs'][i][np.newaxis, :, :]
    X_arr = selected_data['node_features'][i][np.newaxis, :, :]
    Y_arr = selected_data['labels_one_hot'][i]
    chemical = chemicals_list[i]
    num_nodes = A_arr.shape[1]

    # human masks are not used so just create for place-holder
    M = np.zeros((1, num_nodes, 1))
    E = np.zeros((1, num_nodes, num_nodes))

    prob = model.predict_on_batch(x=[M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
    y_hat = prob.argmax()
    y = Y_arr.argmax()

    # Save prediction info:
    text.append(("%s" % labels_dict[y],  # ground truth label
                 "%.2f" % prob.max(),  # probabilistic softmax output
                 "%s" % labels_dict[y_hat]  # predicted label
                 ))
    print(text)
    results_ = []
    for name, method in zip(method_names, methods):
        if name not in mask_dict:
            mask_dict[name]={}

        # node importance
        mask = method.getMasks([M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
        # Normalize
        mask[0] /= (mask[0].max() + 1e-6)
        mask[1] /= (mask[1].max() + 1e-6)
        masks_c0, masks_c1 = mask

        # edge importance
        edge_mask = method.getMasks_edge([M, E, Adjs, A_arr, A_arr, A_arr, X_arr])
        # Normalize
        edge_mask[0] *= Adjs[0]
        edge_mask[1] *= Adjs[0]
        edge_mask[0] /= (edge_mask[0].max() + 1e-6)
        edge_mask[1] /= (edge_mask[1].max() + 1e-6)
        masks_edge_c0, masks_edge_c1 = edge_mask

        if y == 0:
            results_.append({'weights': masks_c0,
                             'edge_weights': masks_edge_c0,
                             'smile': chemical,
                             'index': chem_index_dict[chemical],
                             'method': name,
                             'class': 0})
        elif y == 1:
            results_.append({'weights': masks_c1,
                             'edge_weights': masks_edge_c1,
                             'smile': chemical,
                             'index': chem_index_dict[chemical],
                             'method': name,
                             'class': 1})

        if chemical not in mask_dict[name]:
            mask_dict[name][chemical]={}
            mask_dict[name][chemical]['index'] = chem_index_dict[chemical]
            mask_dict[name][chemical]['ground_truth']=labels_dict[y]
            mask_dict[name][chemical]['predicted'] = labels_dict[y_hat]
            if y == 0:
                mask_dict[name][chemical]['node_importance'] = masks_c0.tolist()
                mask_dict[name][chemical]['edge_importance'] = masks_edge_c0.tolist()
            elif y == 1:
                mask_dict[name][chemical]['node_importance'] = masks_c1.tolist()
                mask_dict[name][chemical]['edge_importance'] = masks_edge_c1.tolist()
        else:
            print('something wrong, duplication:', chemical)

    results.append(results_)

if save_masks:
    print('Saving explanation masks...')
    for name, info in mask_dict.items():
        results_dir = os.path.join(config["results_dir"], "masks")
        out_fn = "mask_bbbp_%s.json"%name
        out_fp = os.path.join(results_dir, out_fn)
        with open(out_fp, 'w') as f:
            json.dump(info, f)


print('Saving explanation visualizations...')
figs = create_figs(results)