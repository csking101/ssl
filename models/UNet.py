import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import crop

class UNetEncoder(nn.Module):
    def __init__(self, input_height, input_width, input_channels, kernel_size=3, is_first_encoder=False, initial_input_channels=3):
        super().__init__()
        
        if is_first_encoder:
            self.first_conv = nn.Conv2d(initial_input_channels,input_channels*2,kernel_size)
        else:
            self.first_conv = nn.Conv2d(input_channels,input_channels*2,kernel_size)
            
        self.second_conv = nn.Conv2d(input_channels*2,input_channels*2,kernel_size)
        
        self.max_pool = nn.MaxPool2d(2,2)#The kernel size and the stride are two respectively, since the dimension gets halved
        

    def forward(self, x):
#         print(x.shape)
        x = self.first_conv(x)
#         print(x.shape)
        x = F.relu(x)
        x = self.second_conv(x)
#         print(x.shape)
        x = F.relu(x)
        x = self.max_pool(x)
#         print(x.shape)
#         print()
        
        return x

class UNetDecoder(nn.Module):
    def __init__(self, input_height, input_width, input_channels, kernel_size=3, is_last_decoder=False, is_bottom_decoder=False):
        super().__init__()
        #This decoder is assuming that the input that is passed, is already concatenated with the residual connection, input channels exlude that
        
        self.first_conv = nn.Conv2d(2*input_channels,input_channels,kernel_size) #Here the number of features get halved
        self.second_conv = nn.Conv2d(input_channels,input_channels,kernel_size)
        #self.up_conv = nn.MaxUnpool2d(2,2)#This does unpooling directly, so the nail in bed of nails is kept at the maximum value index !!!NEEDS TESTING
        if is_last_decoder:
            self.up_conv = nn.Conv2d(input_channels,3,1,1)#This is 1x1 convolution for the last layer
        else:
            self.up_conv = nn.ConvTranspose2d(input_channels,int(input_channels/2),2,2)
            
        if is_bottom_decoder:
            self.first_conv = nn.Conv2d(input_channels,input_channels*2,kernel_size)
            self.second_conv = nn.Conv2d(input_channels*2,input_channels*2,kernel_size)
            self.up_conv = nn.ConvTranspose2d(input_channels*2,input_channels,2,2)

    def forward(self, x):
#         print(x.shape)
        x = self.first_conv(x)
#         print(x.shape)
        x = F.relu(x)
        x = self.second_conv(x)
#         print(x.shape)
        x = F.relu(x)
        x = self.up_conv(x)
#         print(x.shape)
#         print()
        
        return x

class UNet(nn.Module):
    def __init__(self, input_height, input_width, input_channels, kernel_size=3, network_depth=3):
        super().__init__()
        #Here we are assuming that the kernel size is 3

        self.network_depth = network_depth

        self.encoders = []
        self.decoders = []
        
        curr_input_height = input_height
        curr_input_width = input_width
        curr_input_channels = input_channels
        
        for idx in range(network_depth):
            print(f"This is the {idx}th encoder with dimension as {curr_input_height} and channels as {curr_input_channels}")
            is_first_encoder = idx == 0
            
            curr_enc = UNetEncoder(curr_input_height, curr_input_width, curr_input_channels, kernel_size=kernel_size, is_first_encoder=is_first_encoder)
            
            curr_input_height = int((curr_input_height - 4)/2)
            curr_input_width = int((curr_input_width - 4)/2)
            curr_input_channels = int(curr_input_channels*2)
            
            self.encoders.append(curr_enc)
            
        for idx in range(network_depth+1):
            print(f"This is the {idx}th decoder with dimension as {curr_input_height} and channels as {curr_input_channels}")
            is_last_decoder = idx == (network_depth)
            is_bottom_decoder = idx == 0
            
            curr_dec = UNetDecoder(curr_input_height, curr_input_width, curr_input_channels, kernel_size=kernel_size, is_last_decoder=is_last_decoder, is_bottom_decoder=is_bottom_decoder)
            
            self.decoders.append(curr_dec)
            
            curr_input_height = int((curr_input_height-4)*2)
            curr_input_width = int((curr_input_width-4)*2)
            if idx != 0:
                curr_input_channels = int(curr_input_channels/2)
            
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
            
    def forward(self,x):
        to_be_concat = []

        for idx in range(self.network_depth):
#             print(f"Encoder {idx}")
            x = self.encoders[idx](x)
            to_be_concat.append(x)

#         print(f"Base Decoder 0")
        x = self.decoders[0](x)
        
        for idx in range(1,self.network_depth+1): #This is to iterate through the concatenated parts
            img_dim = x.size(dim=2)
            prev_dim = to_be_concat[-idx].size(dim=2)
            
            #We have to crop to concatenate
            top = int((prev_dim - img_dim)/2)            
            left = top
            height = img_dim
            width = img_dim
            
#             print(f"Decoder {idx}")
#             print(f"Before concat x shape is {x.shape}")
            x = torch.cat([crop(to_be_concat[-idx],top=top,left=left,height=height,width=width),x],dim=1)
#             print(f"After concat x shape is {x.shape}")
            x = self.decoders[idx](x)

        return x
