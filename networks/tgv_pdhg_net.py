import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.tgv_pdhg import TgvPdhgTorch

class TgvPdhgNet(nn.Module):
    def __init__(
        self,
        cnn=None,
        device="cpu",
        constraint_activation="softplus",
        scale_factor=0.1,
        T=128,
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
            
        constraint_activation : str
            Activation function to constrain the regularization parameter-maps to be strictly positive. Either "softplus" or "sigmoid".
            Default is "softplus".
            
        scale_factor : int
            Scale the output of the activation to control the values in the lambda regularization map. If using sigmoid then this is the upper bound.
            Default is 0.1.
        """

        super(TgvPdhgNet, self).__init__()
        
        self.device = device
        
        # the CNN-block to estimate the lambda regularization map
        # must be a CNN yielding a two-channeld output, i.e.
        # one map for lambda_cnn_xy and one map for lambda_cnn_t
        self.cnn = cnn
        if cnn is not None:
            self.cnn = cnn.to(device) # This is the U-Net
        
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
            raise ValueError(f"Unknown constraint activation function: {constraint_activation}")
        self.scale_factor = torch.tensor(scale_factor, requires_grad=False, device=self.device)

        self.pdhg = TgvPdhgTorch(device=self.device)
        self.T = T
        

    def get_regularisation_param_maps(self, u):
        regularisation_param_maps = self.cnn(u)
        regularisation_param_maps = regularisation_param_maps.squeeze(0) # Remove batch dimension

        # See page 10 in paper "Learning Regularization Parameter-Maps for Variational Image Reconstruction using Deep Neural Networks and Algorithm Unrolling"
        # constrain the regularization parameter-maps to be strictly positive
        # apply softplus and multiply by t > 0
        # Empirically, we have experienced that the network’s training beneﬁts in terms of faster convergence if the order of the scale of the output is properly set depending on the application. 
        # This can be achieved either by accordingly initializing the weights of the network, or in a simpler way, as we do here by scaling the output of the CNN. 
        # constrain map to be striclty positive and bounded above
        regularisation_param_maps = self.scale_factor * self.constraint_activation(regularisation_param_maps)

        return regularisation_param_maps


    def forward(
            self, u, regularisation_params=None,
            T=None,  # number of iterations for the PDHG algorithm
    ):
        """
        
        >>> net = TgvPdhgNet()
        >>> u = torch.randn(1, 1, 3, 3) # 4D, normal use case when training
        >>> print(u.shape)
        torch.Size([1, 1, 3, 3])
        >>> scalar_params = torch.tensor([0.1, 0.1]) # scalar alpha1 and alpha0
        >>> u_T = net(u, scalar_params, T=16)
        >>> print(u_T.shape)
        torch.Size([1, 1, 3, 3])
        >>> u = torch.randn(3, 3) # 2D, for demo and easy testing by hand
        >>> print(u.shape)
        torch.Size([3, 3])
        >>> u_T = net(u, scalar_params, T=16)
        >>> print(u_T.shape)
        torch.Size([3, 3])
        >>> reg_map_params = torch.randn(2, 3, 3) # regularisation parameter maps
        >>> print(reg_map_params.shape)
        torch.Size([2, 3, 3])
        >>> u_T = net(u, reg_map_params, T=16)
        >>> print(u_T.shape)
        torch.Size([3, 3])
        
        """
        u = u.to(self.device)
        if regularisation_params is None:
            # estimate lambda regularisation parameter maps from the image
            regularisation_params = self.get_regularisation_param_maps(u)
        assert len(regularisation_params) == 2, f"Should have 2 regularisation parameters or 2 parameter maps, not {len(regularisation_params)}"

        alpha0 = regularisation_params[0]
        alpha1 = regularisation_params[1]
        if type(alpha0) != torch.Tensor:
            alpha0 = torch.tensor(alpha0, requires_grad=False)
        if type(alpha1) != torch.Tensor:
            alpha1 = torch.tensor(alpha1, requires_grad=False)
        alpha0 = alpha0.to(self.device)
        alpha1 = alpha1.to(self.device)
        
        p = torch.zeros((*u.shape, 2), device=self.device) # Assume u is 2D now
        if T is None:
            T = self.T
        u_T = self.pdhg.solve(u=u, p=p, alpha1=alpha1, alpha0=alpha0, num_iters=T)
        return u_T
