import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "chemorigins-data", "conditions")

# GNN Feature Dimensions
MOL_FEATURE_DIM = 5  # [MW, Atoms, Rings, Valence e-, LogP]
RXN_FEATURE_DIM = 2  # [Temperature_Kelvin, pH]