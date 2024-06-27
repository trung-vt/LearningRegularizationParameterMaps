import torch
import torch.nn as nn

from networks.grad_ops import GradOperators
from networks.prox_ops import ClipAct

# Code taken from https://www.github.com/koflera/LearningRegularizationParameterMaps

class PDHG(nn.Module):
    def __init__(
            self, dim=3,
            op_norm_AHA=torch.sqrt(torch.tensor(1.0)),
            op_norm_GHG=torch.sqrt(torch.tensor(12.0)), # TODO: Why sqrt(12.0)???
        ):
        super(PDHG, self).__init__()
        self.GradOps = GradOperators(dim, mode="forward", padmode="circular")

        # operator norms
        self.op_norm_AHA = op_norm_AHA
        self.op_norm_GHG = op_norm_GHG
        # operator norm of K = [A, \nabla]
        # https://iopscience.iop.org/article/10.1088/0031-9155/57/10/3065/pdf,
        # see page 3083. NOTE: This does not explain the choice of 12.0 for the operator norm of GHG
        self.L = torch.sqrt(self.op_norm_AHA**2 + self.op_norm_GHG**2) # TODO: Why sqrt(13.0)???

        # function for projecting
        self.ClipAct = ClipAct()

        # constants depending on the operators
        self.tau = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1/L
        self.sigma = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1/L

        # theta should be in \in [0,1]
        self.theta = nn.Parameter(
            torch.tensor(10.0), requires_grad=True
        )  # starting value approximately  1

        


    def forward(
            self, x_5D, lambda_reg, T=128, 
            # lambda_reg_container=None,
    ):
        """
        Reconstructs the image using the PDHG algorithm.

        Parameters:
            dynamic_image_tensor_5D: The (noisy) (dynamic) image tensor.
            Size of the tensor: (`patches`, `channels`, `Nx`, `Ny`, `Nt`) where
            
            - `patches`: number of patches
            - `channels`: number of (colour) channels
            - `Nx`: number of pixels in x
            - `Ny`: number of pixels in y
            - `Nt`: number of time steps (frames)

            lambda_reg: The regularization parameter. Can be a scalar or a tensor of suitable size.
            T: Number of iterations.

        Returns:
            The reconstructed image tensor.
        """

        dim = 3
        patches, channels, Nx, Ny, Nt = x_5D.shape
        
        assert channels == 1, "Only grayscale images are supported."

        device = x_5D.device


        # starting values
        xbar = x_5D.clone()
        x0 = x_5D.clone()
        xnoisy = x_5D.clone()

        # dual variable
        p = x_5D.clone()
        q = torch.zeros(patches, dim, Nx, Ny, Nt, dtype=x_5D.dtype).to(device)

        # sigma, tau, theta
        sigma = (1 / self.L) * torch.sigmoid(self.sigma)  # \in (0,1/L)
        tau = (1 / self.L) * torch.sigmoid(self.tau)  # \in (0,1/L)
        theta = torch.sigmoid(self.theta)  # \in (0,1)

        # Algorithm 2 - Unrolled PDHG algorithm (page 18)
        # TODO: In the paper, L is one of the inputs but not used anywhere in the pseudo code???
        for kT in range(T):
            # update p
            p =  (p + sigma * (xbar - xnoisy) ) / (1. + sigma)
            # update q
            q = self.ClipAct(q + sigma * self.GradOps.apply_G(xbar), lambda_reg)

            x1 = x0 - tau * p - tau * self.GradOps.apply_GH(q)

            if kT != T - 1:
                # update xbar
                xbar = x1 + theta * (x1 - x0)
                x0 = x1
            with torch.no_grad():
                torch.cuda.empty_cache()

        # Explicitly free up (GPU) memory to be safe
        del x_5D
        del xbar, x0, xnoisy
        del p, q
        del tau, sigma, theta

        with torch.no_grad():
            torch.cuda.empty_cache()

        # if lambda_reg_container is not None:
        #     assert isinstance(lambda_reg_container, list), f"lambda_reg_container should be a list, not {type(lambda_reg_container)}"
        #     lambda_reg_container.append(lambda_reg) # For comparison

        return x1
