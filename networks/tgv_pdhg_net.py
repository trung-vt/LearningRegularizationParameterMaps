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
        
        # TODO: Remove this
        regularisation_param_maps = torch.tensor([0.1, 0.1], device=self.device)

        return regularisation_param_maps


    def forward(
            self, u, regularisation_params=None,
            T=128,  # number of iterations for the PDHG algorithm
    ):
        u = u.to(self.device)
        if regularisation_params is None:
            # estimate lambda reg from the image
            regularisation_params = self.get_regularisation_param_maps(u)
        assert len(regularisation_params) == 2, f"Should have 2 regularisation parameters or 2 parameter maps, not {len(regularisation_params)}"

        alpha0 = regularisation_params[0]
        alpha1 = regularisation_params[1]
        # assert alpha0.shape == alpha1.shape, f"alpha0 and alpha1 should have the same shape, not {alpha0.shape} and {alpha1.shape}"
        
        # # TODO: Do we need to impose the same shape for u and alpha0?
        # assert alpha0.shape == u.shape, f"Shape {alpha0} of alpha0 and alpha1 are not matching shape {u.shape} of input"

        p = torch.zeros((*u.shape, 2), device=self.device) # Assume u is 2D now
        u_T = self.pdhg.solve(u=u, p=p, alpha1=alpha1, alpha0=alpha0, num_iters=T)
        return u_T
