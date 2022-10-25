import numpy as np
from deepchem.feat import ConvMolFeaturizer
from rdkit.Chem import MolFromSmiles
import scipy.sparse as sp
def form_one_shot_feature(sorted_id): 
    # num_label = len(list(set(label_list)))
    # print(f"num_label: {num_label}")
    one_shot_label_list = []
    for each_id in sorted_id: #label_list is a multiD list (e.g., [[1],[1],[0]])
        raw_label_list = [0 for i in range(len(sorted_id))]
        raw_label_list[each_id] = 1
        one_shot_label_list.append(raw_label_list)
    # print(f"np.array(one_shot_label_list): {np.array(one_shot_label_list)}")
    sorted_matrix = np.array(one_shot_label_list)
    print(f"form_one_shot_feature's sorted_matrix: {sorted_matrix}")
    return sorted_matrix

def process_features(molecules,vertices):
    features_dict = {}#we are only concerned the values
    molecules_index = 0
    for each_molecule in molecules:
        sortind = np.argsort(vertices[molecules_index].sum(axis = 1) - 1)
        row_index = 0
        row_to_row_sum_dict = {}
        for each_row in vertices[molecules_index]:
            num_of_neighbours = sum(each_row)
            # print(f"each_row: {each_row}, sum: {num_of_neighbours}")
            row_to_row_sum_dict[row_index] = num_of_neighbours
            row_index += 1
        sorted_dict = {k: v for k, v in sorted(row_to_row_sum_dict.items(), key=lambda item: item[1])}
        sorted_row_id = np.array(list(sorted_dict.keys()))
        print(f"sorted_row_id is {sorted_row_id}")
        print(f"sortind is {sortind}")
        # assert sorted_row_id.all() == sortind.all(), f"sorted_row_id ({sorted_row_id}) and sortind ({sortind}) not equal" #John assertion
        
        # number_of_ele_in_row = sorted_row_id.size
        sorted_matrix = form_one_shot_feature(sorted_row_id)
        # assert sorted_matrix.all() == np.eye(20)[sortind,:].all(), f"sorted_matrix incorrect, expected {np.eye(N)[sortind,:]} but got {sorted_matrix}"
        # node_features.append(np.matmul(sortMatrix.T, feat.get_atom_features()))
        features_dict[molecules_index] = sorted_matrix.T.dot(each_molecule.get_atom_features())#correctness checked
        # print(f"features[molecules_index]: {features[molecules_index]}")
        molecules_index += 1
    features = []
    for each_key in features_dict:
        features.append(features_dict[each_key])
    return features
        # break

def preprocess_adj(adj):
    adj = sp.csr_matrix(adj)
    adj_numpy = adj.toarray()

    adj = adj + sp.eye(adj.shape[0])

    identity_matrix = []
    for i in range(adj_numpy[0].size):
        inner_matrix = [0 for j in range(adj_numpy[0].size)]
        inner_matrix[i] = 1
        identity_matrix.append(inner_matrix)
    
    adj_numpy_after_adding_identity = adj_numpy + np.array(identity_matrix)

    sum_of_adj = []
    for each_row in adj.toarray():
        sum = 0
        for entry_index, each_entry in enumerate(each_row):
            sum += each_entry
        sum_of_adj.append(sum**(-0.5))

    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
    multi_diagonal_matrix = []
    for i in range(len(sum_of_adj)):
        zero_array = [0 for j in range(len(sum_of_adj))]
        zero_array[i] = sum_of_adj[i]
        multi_diagonal_matrix.append(zero_array)

    adj = adj.dot(d).transpose().dot(d).tocsr()
    normalised_adj = np.multiply(np.multiply(adj,np.array(multi_diagonal_matrix)).T,multi_diagonal_matrix)

    return normalised_adj

def form_one_shot(label_list): 
    # num_label = len(list(set(label_list)))
    # print(f"num_label: {num_label}")
    one_shot_label_list = []
    for each_label in label_list: #label_list is a multiD list (e.g., [[1],[1],[0]])
        raw_label_list = [0,0]
        raw_label_list[each_label[0]] = 1
        one_shot_label_list.append(raw_label_list)
    # print(f"np.array(one_shot_label_list): {np.array(one_shot_label_list)}")
    return np.array(one_shot_label_list)

def preprocess_main(node, edges, label, smiles):
    labels_one_hot = form_one_shot(label.tolist())
    molecules = ConvMolFeaturizer().featurize([MolFromSmiles(smile) for smile in smiles])
    molecules_features = process_features(molecules,edges)
    # features = process_features(molecules,edges)
    normalise_edges = [preprocess_adj(each_edge) for each_edge in edges] #this function is quite slow
    return labels_one_hot,molecules_features,edges,normalise_edges
'''
analogu to 
    return {'labels_one_hot': labels_one_hot,
            'node_features': node_features,
            'adjs': adjs,
            'norm_adjs': norm_adjs}
'''