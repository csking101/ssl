from tqdm import tqdm
import torch


class Trainer:
    def __init__(self,train_data_loader, val_data_loader,test_data_loader, architecture, optimizer, alpha, adaptive_weighing_coefficient, num_epochs, framework=None,device='cpu'):
        self.train_loader = train_data_loader
        self.val_loader = val_data_loader
        self.test_loader = test_data_loader
        self.device = device
        self.num_epochs = num_epochs
        
        image_sample, msk_sample = next(iter(val_data_loader))
        shape = image_sample.shape
        batch_size = image_sample.size(dim=0)
        input_channels = 32
        input_height = image_sample.size(dim=2)
        input_width = image_sample.size(dim=3)
        
        if framework is not None:
            self.ssl_framework = framework(architecture, alpha, optimizer, adaptive_weighing_coefficient,input_height,input_width,input_channels)
            
        
    def _train_one_epoch(self):
        epoch_loss = 0
        
        for input,label in tqdm(self.train_loader):
            input = input.to(device=self.device,dtype=torch.float)
            label = label.to(device=self.device,dtype=torch.float)
            loss = self.ssl_framework.train(x=input, ground_truth=None)
            
            epoch_loss += loss.item()
            
        return epoch_loss

    def train(self):
        for epoch_idx in tqdm(range(self.num_epochs)):
            print(f"Epoch {epoch_idx}")
            epoch_loss = self._train_one_epoch()
            print(f"Train Loss: {epoch_loss}")
    
    def val_test(self):
        for input,label in self.val_loader:
            input = input.to(device=self.device,dtype=torch.float)
            label = label.to(device=self.device,dtype=torch.float)
            loss = self.ssl_framework.test_models(x=input, ground_truth=label)
            print(f"Validation Loss: {loss}")    
    
    def test(self):
        for input,label in self.test_loader:
            input = input.to(device=self.device,dtype=torch.float)
            label = label.to(device=self.device,dtype=torch.float)
            loss = self.ssl_framework.test_models(x=input, ground_truth=label)
            print(f"Test Loss: {loss}") 