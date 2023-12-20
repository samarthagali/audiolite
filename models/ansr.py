from torch import nn
import torch

class MultiKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_features= 1):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= n_features, stride= 1, kernel_size= 1, padding= 0, bias= False),
            nn.Tanh()
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= n_features, stride= 1, kernel_size= 3, padding= 1, bias= False),
            nn.Tanh()
        )
        self.c5 = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= n_features, stride= 1, kernel_size= 5, padding= 2, bias= False),
            nn.Tanh()
        )
        self.c35 = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= n_features, stride= 1, kernel_size= 3, padding= 1, bias= False),
            nn.Tanh(),
            nn.Conv1d(in_channels= n_features, out_channels= n_features, stride= 1, kernel_size= 5, padding= 2, bias= False),
        )
        self.concat_c1 = nn.Sequential(
            nn.Conv1d(in_channels= n_features*4, out_channels= out_channels, stride= 1, kernel_size= 1, padding= 0, bias= False),
            nn.Tanh()
        )
    
    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c3(x)
        x3 = self.c5(x)
        x4 = self.c35(x)
        x = torch.concat((x1, x2, x3, x4), 1)
        x = self.concat_c1(x)
        return x
    

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_replace = nn.Conv1d(in_channels= in_channels, out_channels= out_channels, stride= 1, kernel_size= 3, padding= 1, bias= False)
        self.max_replace = nn.Conv1d(in_channels= in_channels, out_channels= out_channels, stride= 1, kernel_size= 5, padding= 2, bias= False)
        self.c7 = nn.Sequential(
            nn.Conv1d(in_channels= in_channels*2, out_channels= out_channels, stride= 1, kernel_size= 7, padding= 3, bias= False),
        )
        
    def forward(self, x):
        avg_channel = self.avg_replace(x)
        max_channel = self.max_replace(x)
        concat_channel = torch.cat((avg_channel, max_channel), 1)
        
        concat_c7 = self.c7(concat_channel)
        
        scores = torch.sigmoid(concat_c7)
        outputs = torch.tanh(concat_c7)
        
        x = (scores * outputs) + x
        
        return x


class SuperRes(nn.Module):
    def __init__(self,n_features,scale):
        super().__init__()
        self.scale=scale
        self.decoder=Decoder(n_features)
    
    def forward(self,x):
        decoder_outputs = []
        prev_output =x 
        for i in range(self.scale)[::-1]:
            prev_output = self.decoder(prev_output)
            decoder_outputs.append(prev_output)
        final_out = decoder_outputs[-1]
        return final_out
    


class SRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mkc_sa = nn.Sequential(
            MultiKernelConv(in_channels, out_channels),
            SpatialAttention(in_channels, out_channels)
        )
    
    def forward(self, x):
        return x + self.mkc_sa(x)
    

class ASNR(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor= 4.0)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(SRBlock(in_channels, out_channels))
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels= num_blocks, out_channels= out_channels, stride= 1, kernel_size= 3, padding= 1),
            nn.Tanh(),
            nn.Conv1d(in_channels= in_channels, out_channels= out_channels, stride= 1, kernel_size= 3, padding= 1)
        )
            
    def forward(self,x):
        x = self.upsample(x)
        prev_output = [x, ]
        for block in self.blocks:
            prev_output.append(block(prev_output[-1]))
        
        concat_blocks = torch.cat(prev_output[1:], 1)
        output = self.c3(concat_blocks)
        
        return output + x