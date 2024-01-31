import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import crop

INPUT_CHANNELS = 3




class CustomDataset(Dataset):
    def __init__(self, data_path):
        #For unlabelled samples, make sure that the ground truth is None
        super().__init__()
        
class CustomDataLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__()
        
