import torch 
import numpy as np 


class SliderProjector(torch.nn.Module):
    def __init__(
        self,
        out_dim, # Dimension of the output token that the projector will generate 
        pe_dim, # The dimension of positional embedding that will be applied 
        n_layers = 4,
        is_clip_input = True, # This function will check whether the clip embeddings are the input of the projector net or not 
    ):
        super().__init__()
        self.out_dim = out_dim 
        self.pe_dim = pe_dim 
        self.is_clip_input = is_clip_input 

        # Add the layers here in defining, assume n_layers is another parameter
        layers = []
        pe_extender_dim = 768

        # if the clip embeddings are to be passed along with the input of the slider scalar value, we will increase the dimensions of the input of the projector net
        if is_clip_input:
            in_dim = pe_extender_dim + 768
        else:
            in_dim = pe_extender_dim
        
        # iterating over the layers and accumulating the layers in a list for defining the model 
        for i in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.ReLU())
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))

        # a simple linear layer to extend the pe into a higher dimensional space 
        self.pe_extender = torch.nn.Linear(pe_dim, 768) 
        # then we will pass it through a projector network  
        self.projector = torch.nn.Sequential(*layers)

    # A simple encoding function for the scalar input for a pe embedding 
    def posEnc(self, s):
        pe = torch.stack([torch.sin(torch.pi * s), torch.cos(torch.pi * s)], dim=-1)  
        return pe

    # A forward function that will take the input x and then projects it to a token embedding to condition the diffusion model. 
    def forward(self, s, clip_embeddings = None):
        # Apply the positional embedding to the input scalar 
        x_pe = self.posEnc(s) 
        x_scale_embedding = self.pe_extender(x_pe) # (1, 768)

        if clip_embeddings is not None: # if the clip input is passed, we will concatenated it with the scalar embeddings for processing 
            # print("clip embeddings shape: {}".format(clip_embeddings.shape))
            x_combined_embedding = torch.cat([x_scale_embedding, clip_embeddings], dim=-1) # (1, 768 + 768)

        x_proj = self.projector(x_combined_embedding)
        # print("x proj shape: {}".format(x_proj.shape))
        return x_proj


class SliderProjector_wo_clip(torch.nn.Module):
    def __init__(
        self,
        out_dim, # Dimension of the output token that the projector will generate 
        pe_dim, # The dimension of positional embedding 
        n_layers = 4,
        is_clip_input = False, # This function will check whether the clip embeddings are the input of the projector net or not 
    ):
        super().__init__()
        self.out_dim = out_dim 
        self.pe_dim = pe_dim 

        # Add the layers here in defining, assume n_layers is another parameter
        layers = []
        pe_extender_dim = 768

        # extending the input dimenstion to the 768 with a linear layer to keep the dimensions consistent with other clip based model. 
        in_dim = pe_extender_dim 

        # iterating over the layers and accumulating the layers in a list for defining the model 
        for i in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.ReLU())
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))

        # adding a pe extender to have the same dimension as clip embeddings 
        self.pe_extender = torch.nn.Linear(pe_dim, 768) 
        # then we will pass it through a projector network  
        self.projector = torch.nn.Sequential(*layers)

    def posEnc(self, s):
        pe = torch.stack([torch.sin(torch.pi * s), torch.cos(torch.pi * s)], dim=-1)  
        return pe

    # A forward function that will take the input x and then projects it to a token embedding to condition the diffusion model. 
    def forward(self, s):
        x_pe = self.posEnc(s) 
        x_scale_embedding = self.pe_extender(x_pe)
 
        x_proj = self.projector(x_scale_embedding) 
        return x_proj

