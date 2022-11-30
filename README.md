Welcome! This is the COMP5331 project "Learning to Explain Graph Neural Networks" conducted by Group13.  

* Group Members: CHEN Xiao, LI Tsz On, LI Xiaolei, WAN Ho Yin, XU Congying
* Project Type: Implementation-oriented
* Paper to Implement:
"Gao, Yuyang, Tong Sun, Rishab Bhatt, Dazhou Yu, Sungsoo Hong, and Liang Zhao. "Gnes: Learning to explain graph neural networks." In 2021 IEEE International Conference on Data Mining (ICDM), pp. 131-140. IEEE, 2021."



# Environment 
* Python 3.7.9
    * Install related packages by `pip install requirements.txt`
* OS: Red Hat 8.5.0-7 （recommened, not necessary）
# Run
```
python main.py
```
It will run the training. The evaluation and ROC curve will be saved in "saved_models/"
# Visualization
After Running the main.py, the trained file will be saved in "saved_models/" as a .h5 file. 

Next if you want to test the model by running some visualization examples, you could run
```
python visualization.py
```
By default, we selected [628, 593, 492] for visualization. You can also change to other chemicals by editing data_points in line 30.  
```
data_points = [628, 593, 492]
```
The visualization will be saved in "figs/" as .jpeg files
# Description of each source file
| Source file | Description |
| --- | ----------- |
| main.py | Script to execute GNES |
| load_data.py | Load BBBP dataset and human annotated dataset |
| preprocessing.py | Preprocess the loaded dataset |
| training.py | construct and train a GNN |
| explanation_methods.py | Optional explanation methods: GradCAM & EB |
| loss_function.py | Loss function of GNES, including inference accuracy loss, explanability loss, and regularization loss |
| utils.py | Common functions which are frequently reused|
# Credits
We adopted some of the codes from the original author. In particular, we reused the following files:
| Used files | Description |
| --- | ----------- |
| utils.py | Some utility function are used |
| plot_utils.py | Some configuration of the plots |
| config.py | The configuration of the dataset is reused |