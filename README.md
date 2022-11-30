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

# Example for evaluation and explanantion @harry
```
```

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
