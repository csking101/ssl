import torch.nn as nn
from torchvision.transforms.functional import crop

def cross_entropy_loss(predicted, label): #This function may cause issues during backpropagation
#     print(f"The shape of predicted is {predicted.size()}")
#     print(f"The shape of label is {label.size()}")
    predicted_dim = predicted.size(dim=2)#Cropping is needed here
    label_dim = label.size(dim=2)
    top = int((label_dim - predicted_dim)/2)
    left = top
    cropped_label = crop(label,top=top,left=left,height=predicted_dim,width=predicted_dim)
    return nn.CrossEntropyLoss()(predicted,cropped_label)