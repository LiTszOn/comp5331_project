# How to run

1. Install related packages by `pip install requirements.txt`
2. `python main.py`

# Description of each source file
| Source file | Description |
| --- | ----------- |
| main.py | Script to execute GNES |
| load_data.py | Load BBBP dataset and human annotated dataset |
| preprocessing.py | Preprocess the loaded dataset |
| training.py | construct and train a GNN |
| explanation_methods.py | Optional explanation methods: GradCAM & EB |
| loss_function.py | Loss function of GNES, including inference accuracy loss, explanability loss, and regularization loss |
