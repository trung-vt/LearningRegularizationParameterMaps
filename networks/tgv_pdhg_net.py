import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.tgv_pdhg import TgvPdhgTorch

class TgvPdhgNet(nn.Module):
    def __init__(
        self,
        cnn=None,
        device="cpu",
        phase="training",
        constraint_activation="softplus",
        upper_bound=0.1,
    ):
        """
        TgvPdhgNet
        
        Parameters
        ----------
        cnn : nn.Module
            CNN block to estimate the lambda regularization map.
            
        device : str
            Device to run the model on.
            Default is "...".
            
        phase : str    
            Phase, either "training" or "testing".
            Default is "training".

        constraint_activation : str
            Activation function to constrain the regularization parameter-maps to be strictly positive. Either "softplus" or "sigmoid".
            Default is "softplus".
            
        upper_bound : int
            Upper bound of the values of lambda regularization map.
            Default is ...
        """

        super(TgvPdhgNet, self).__init__()
        
        self.device = device
        
        # the CNN-block to estimate the lambda regularization map
        # must be a CNN yielding a two-channeld output, i.e.
        # one map for lambda_cnn_xy and one map for lambda_cnn_t
        self.cnn = cnn.to(device)    # NOTE: This is actually the UNET!!! (At least in this project)
        
        # distinguish between training and test phase;
        # during training, the input is padded using "reflect" padding, because
        # patches are used by reducing the number of temporal points;
        # while testing, "reflect" padding is used in x,y- direction, while
        # circular padding is used in t-direction
        self.phase = phase
        
        # See page 10 in paper "Learning Regularization Parameter-Maps for Variational Image Reconstruction using Deep Neural Networks and Algorithm Unrolling"
        # constrain the regularization parameter-maps to be strictly positive
        # apply softplus and multiply by t > 0
        # Empirically, we have experienced that the network’s training beneﬁts in terms of faster convergence if the order of the scale of the output is properly set depending on the application. 
        # This can be achieved either by accordingly initializing the weights of the network, or in a simpler way, as we do here by scaling the output of the CNN. 
        if constraint_activation == "softplus":
            self.constraint_activation = F.softplus
        elif constraint_activation == "sigmoid":
            self.constraint_activation = torch.sigmoid
        else:
            raise ValueError(f"Unknown constraint activation function: {constraint}")
        self.upper_bound = torch.tensor(upper_bound, requires_grad=False, device=self.device)

        self.pdhg = TgvPdhgTorch(device=self.device)
        

    def get_regularisation_param_maps(self, u):
        regularisation_param_maps = self.cnn(u)

        # See page 10 in paper "Learning Regularization Parameter-Maps for Variational Image Reconstruction using Deep Neural Networks and Algorithm Unrolling"
        # constrain the regularization parameter-maps to be strictly positive
        # apply softplus and multiply by t > 0
        # Empirically, we have experienced that the network’s training beneﬁts in terms of faster convergence if the order of the scale of the output is properly set depending on the application. 
        # This can be achieved either by accordingly initializing the weights of the network, or in a simpler way, as we do here by scaling the output of the CNN. 
        # constrain map to be striclty positive and bounded above
        regularisation_param_maps = self.upper_bound * self.constraint_activation(regularisation_param_maps)

        return regularisation_param_maps


    def forward(
            self, u_4d, regularisation_params=None,
            T=128,  # number of iterations for the PDHG algorithm
            # lambda_reg_container=None,
    ):
        assert len(u_4d.shape) == 4, f"Input tensor should be 4D with shape (batch_size, channels, height, width), not {u_4d.shape}" # 4D for cnn input
        
        if regularisation_params is None:
            # estimate lambda reg from the image
            regularisation_params = self.get_regularisation_param_maps(u_4d)
        assert len(regularisation_params.shape) == 2, f"Should have 2 regularisation parameters or 2 parameter maps, not {regularisation_params.shape}"

        u = u_4d.squeeze(0).squeeze(0) # Remove batch and channel dimensions to make u 2D
        u = u.to(self.device)

        alpha0 = regularisation_params[0]
        alpha1 = regularisation_params[1]
        assert alpha0.shape == alpha1.shape, f"alpha0 and alpha1 should have the same shape, not {alpha0.shape} and {alpha1.shape}"
        assert alpha0.shape == u.shape, f"Shape {alpha0} of alpha0 and alpha1 are not matching shape {u.shape} of input"

        # if lambda_reg_container is not None:
        #     assert type(lambda_reg_container) == list, f"lambda_reg_container should be a list, not {type(lambda_reg_container)}"
        #     lambda_reg_container.append(lambda_reg) # For comparison

        p = torch.zeros((u.shape[0], u.shape[1], 2), device=self.device) # Assume u is 2D now
        u_T = self.pdhg.solve(u=u, p=p, alpha1=alpha1, alpha0=alpha0, num_iters=T)
        return u_T
