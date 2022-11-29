from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.nfp_preprocessor import NFPPreprocessor
import ast


def convert_human_data_to_dict(file_path):
    f = open(file_path, "r") 
    data_dict = {}
    all_lines = f.readlines()
    # print(f"all_lines: {all_lines}")
    for each_line in all_lines[1:]:
        # print(f"each line: {each_line}")
        if "skipped" not in each_line:
            value = "\"{\"\"" + each_line.split(",\"{\"\"")[1]
            row_id,img_id,state = each_line.split(",\"{\"\"")[0].split(",")
            # print(f"value is {value}")
            if state == "labeled":
                if img_id not in data_dict:
                    data_dict[int(img_id)] = {}
                # mask = json.loads(value[1:-1])
                data_dict[int(img_id)]['node_importance'] = ast.literal_eval(value.split(', ""edge_importance"": ')[0].replace('"{""node_importance"": ',""))
                if ",,,,,," in value:#val.csv has ,,,,, attach along some of the records
                    value = value.split(",,,,,,")[0]
<<<<<<< HEAD
                data_dict[int(img_id)]['edge_importance'] = ast.literal_eval(value.split(', ""edge_importance"": ')[1].replace('}"',""))
                # print(f"john's node_importance: {data_dict[img_id]['node_importance']}")
                # print(f"john's edge_importance: {data_dict[img_id]['edge_importance']}")
                # break
=======
                data_dict[img_id]['edge_importance'] = ast.literal_eval(value.split(', ""edge_importance"": ')[1].replace('}"',""))
                #print(f"john's node_importance: {data_dict[img_id]['node_importance']}")
                #print(f"john's edge_importance: {data_dict[img_id]['edge_importance']}")
                #break
>>>>>>> cc007c61f5a95d0a2512e6e313f227aa82171aff
    f.close()
    return data_dict

def load_manual_annotation():
    dict4train = convert_human_data_to_dict("mask_data/BBBP_train.csv")
    dict4test = convert_human_data_to_dict("mask_data/BBBP_test.csv")
    dict4val = convert_human_data_to_dict("mask_data/BBBP_val.csv")
    print(f"john's dict4train: {dict4train}")
    print(f"john's dict4test: {dict4test}")
    print(f"john's dict4val: {dict4val}")
    return dict4train,dict4test,dict4val

def load_and_preprocess_chemical_data(raw_data_file_path):
    """
    Load BBBP data
    """
    csvprocessor = CSVFileParser(NFPPreprocessor(), labels="p_np", smiles_col="smiles")
    csv_data = csvprocessor.parse(raw_data_file_path,return_smiles = True)
    smiles = csv_data['smiles']
    nodes, vertices, label = csv_data['dataset'].get_datasets()
    # return nodes,vertices,labels,smiles
    return nodes, vertices, label, smiles
    # if dataset == "BBBP"
    #     label_one_shot = form_one_shot(label) #correctness examined, please see join assertion 1, one shot
    #     molecules = ConvMolFeaturizer().featurize([MolFromSmiles(smile) for smile in smiles]) #Default is convmol