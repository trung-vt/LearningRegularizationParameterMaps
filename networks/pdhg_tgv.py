import torch
import torch.nn as nn
import numpy as np

from networks.gradops.gradops_2 import dx_forward, dy_forward, dx_backward, dy_backward
# from networks.gradops.gradops_1 import dx_forward, dy_forward, dx_backward, dy_backward

class PdhgTgvSolver:
    """
    See page 17 in "Recovering piecewise smooth multichannel images..."
    https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370
    """
    # Data-type-independent. User must ensure consistency.
    def __init__(
        self, 
        sigma, tau, nabla_h, e_h, div_h_v, div_h_w, P_alpha1, P_alpha0,
        convergence_limit
    ):
        print(f"convergence_limit: {convergence_limit}")
        excluded = ["self", "convergence_limit"]
        # Set all the attributes without repeating self.x = x
        for name, value in vars().items():
            if name not in excluded:
                setattr(self, name, value)
                
        # while self.sigma * self.tau < convergence_limit:
        #     self.sigma /= 2
        #     self.tau /= 2
        # print(f"sigma: {self.sigma}, tau: {self.tau}")
        # Tried:
        [
            (0.29, 0.29), #31.130604748354536
        ]
        self.sigma = 0.29
        self.tau = 0.29
        # self.sigma = 0.1
        # self.tau = 0.87
        
    def solve(self, f, u, p, u_bar, p_bar, v, w, alpha1, alpha0, num_iters=100):
        # print(f"Using data-type-independent solver")
        for i in range(num_iters):
            # print("Iteration: ", i)
            v_next = self.P_alpha1(v + self.sigma * (self.nabla_h(u_bar) - p_bar), alpha1)
            
            w_next = self.P_alpha0(w + self.sigma * self.e_h(p_bar), alpha0)
            
            # u_next = self.id_tauFh_inverse(u + self.tau * self.div_h_v(v_next), u0)
            
            # Resolvent operator (id + τ∂Fh)^(-1)
            # See page 19 in "Recovering piecewise smooth multichannel..." for case q = 2
            u_next = (u + self.tau * (self.div_h_v(v_next) + f)) / (self.tau + 1.0)
            # f is the noisy image
            
            
            p_next = p + self.tau * (v_next + self.div_h_w(w_next))
            
            u_bar = u_next * 2.0 - u
            p_bar = p_next * 2.0 - p
            
            u, p = u_next, p_next
            v, w = v_next, w_next

        # del v_next, w_next # Explicitly free up memory
        # Don't free u_next and p_next, u_next is actually reference to u
            
        return u
        
    # def id_tauFh_inverse(self, x, x0):
    #     """
    #     Resolvent operator (id + τ∂Fh)^(-1)
    #     See page 15 in "Recovering piecewise smooth multichannel...", 3.2 - A Numerical Algorithm 
    #     """
    #     # Placeholder for (id + τ∂Fh)^(-1)
    #     # return x
    #     # return x / self.tau
    #     # See page 19 in "Recovering piecewise smooth multichannel..."
    #     return (x + self.tau * x0) / (self.tau + 1)



    
class GradOpsTorch:
    def nabla_h(u):
        # $\Nabla_h$ and $\mathcal{E}_h$ ??? See page 13 in 'Recovering piecewise smooth multichannel...'
        # https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370
        
        # Parameters
        # ----------
        # u : torch.Tensor
        #     Assume 2D tensor of shape [n, n] (for now).
            
        #     scalar field?
            
        # Returns
        # -------
        # grad_u : torch.Tensor
        #     Assume 3D tensor of shape [n, n, 2] (for now).
        #     The point is that, the shape of the output is one more dimension added to the end of input,
        #     and that extra last dimension is of size 2.
            
        #     Gradient of the scalar field u?
        # assert u.dim() == 2, f"u must be a 2D tensor, but got {u.dim()}D tensor"
        # Compute the gradient in both x and y directions
        
        """
        
        Example
        -------
        >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> GradOpsTorch.nabla_h(u)
        tensor([[[3, 1],
                 [3, 1],
                 [3, 0]],
        <BLANKLINE>
                [[3, 1],
                 [3, 1],
                 [3, 0]],
        <BLANKLINE>
                [[0, 1],
                 [0, 1],
                 [0, 0]]])
        """
        
        # tensor([[[3, 1],
        #          [3, 1],
        #          [3, -3]],
        # <BLANKLINE>
        #         [[3, 1],
        #          [3, 1],
        #          [3, -6]],
        # <BLANKLINE>
        #         [[-7, 1],
        #          [-8, 1],
        #          [-9, -9]]])
        
        dx_f = dx_forward(u)
        dy_f = dy_forward(u)
        nabla_h_u = torch.stack([dx_f, dy_f], dim=-1)
        return nabla_h_u
    

    def e_h(v):
        """
        
        Parameters
        ----------
        v : torch.Tensor
            Assume 3D tensor of shape [n, n, 2] (for now).
            
        Returns
        -------
        w : torch.Tensor
            Assume 4D tensor of shape [n, n, 2, 2] (for now).
            
        Example
        -------
        >>> v = torch.tensor([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]], [[7, 7], [8, 8], [9, 9]]])
        >>> GradOpsTorch.e_h(v)
        tensor([[[[ 1,  2],
                  [ 2,  5]],
        <BLANKLINE>
                  [[ 5,  6],
                   [ 6,  9]]],
        <BLANKLINE>
                  [[[ 9, 10],
        """
        assert len(v.shape) == 3, f"v must be a 3D tensor, but got {len(v.shape)}D tensor"
        assert v.shape[-1] == 2, f"v must have 2 channels in the last dimension, but got {v.shape[-1]} channels"
        dx_b_1 = dx_backward(v[..., 0])
        dy_b_1 = dy_backward(v[..., 0])
        dx_b_2 = dx_backward(v[..., 1])
        dy_b_2 = dy_backward(v[..., 1])
        # half = torch.tensor(0.5)
        # print(f"dx_b_1: {dx_b_1}, dy_b_1: {dy_b_1}, dx_b_2: {dx_b_2}, dy_b_2: {dy_b_2}")
        # w_1 = torch.tensor([dx_b_1, half * (dy_b_1 + dx_b_2)**2])
        # w_2 = torch.tensor([half * (dy_b_1 + dx_b_2)**2, dy_b_2])
        # w = torch.tensor([w_1, w_2])
        
        w = torch.stack(
            [
                torch.stack([dx_b_1, 0.5 * (dy_b_1 + dx_b_2)], dim=-1),
                torch.stack([0.5 * (dy_b_1 + dx_b_2), dy_b_2], dim=-1)
            ],
            # Notes: Any dim is fine because symmetric (b == c) 
            dim=-2   # [a, b] and [c, d]  -->  [[a, b], [c, d]]
            # dim=-1   # [a, b] and [c, d]  -->  [[a, c], [b, d]] 
        )
        return w

    
    def div_h_v(v):
        """
        $\text{div}_h$. See page 14 in 'Recovering piecewise smooth multichannel...'
        https://unipub.uni-graz.at/obvugroa/content/titleinfo/125370
            
        Parameters
        ----------
        v : torch.Tensor
            Assume 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of the input is one more dimension added to the end of the output,
            and that extra last dimension is of size 2.
            
            representing the vector field?
            
        Returns
        -------
        div_v : torch.Tensor
            Assume 2D tensor of shape [n, n] (for now).
            
            Divergence of the vector field v?
            
        Example
        -------
        """
        # assert v.dim() == 3, f"v must be a 3D tensor, but got {v.dim()}D tensor"
        # Compute the divergence from the gradient in both x and y directions
        dx_b_1 = dx_backward(v[..., 0])
        dy_b_2 = dy_backward(v[..., 1])
        div_h_v = dx_b_1 + dy_b_2
        return div_h_v
    
    def div_h_w(w):
        assert len(w.shape) == 4, f"w must be a 4D tensor, but got {len(w.shape)}D tensor"
        assert w.shape[-2] == 2, f"w must have 2 channels in the second last dimension, but got {w.shape[-2]} channels"
        assert w.shape[-1] == 2, f"w must have 2 channels in the last dimension, but got {w.shape[-1]} channels"
        dx_f_11 = dx_forward(w[..., 0, 0])
        dy_f_12 = dy_forward(w[..., 0, 1])
        
        dx_f_12 = dx_forward(w[..., 0, 1])
        dy_f_22 = dy_forward(w[..., 1, 1])
        v_1 = dx_f_11 + dy_f_12
        v_2 = dx_f_12 + dy_f_22
        v = torch.stack(
            [
                v_1, v_2
            ],
            dim=-1
        )
        return v
    

class PdhgTgvTorch(nn.Module):
    # See page 17 in "Recovering piecewise smooth multichannel..." for the algorithm
    def __init__(self, device):
        super(PdhgTgvTorch, self).__init__() # Ensure proper initialisation
        self.device = device
        self.sigma = torch.tensor(1.0, device=device)
        self.tau = torch.tensor(1.0, device=device)
        self.theta = torch.tensor(1.0, device=device)

        # Norm of K. See page 16 in "Recovering piecewise smooth multichannel..."
        convergence_limit = 1.0 / (0.5 * (17 + torch.sqrt(torch.tensor(33.0, device=device))))
        # TODO: Change this for the 2D static greyscale image denoising problem
        
        self.pdhg_tgv_solver = PdhgTgvSolver(
            sigma=self.sigma,
            tau=self.tau,
            nabla_h=GradOpsTorch.nabla_h,
            e_h=GradOpsTorch.e_h,
            div_h_v=GradOpsTorch.div_h_v,
            div_h_w=GradOpsTorch.div_h_w,
            P_alpha1=self.P_alpha1,
            P_alpha0=self.P_alpha0,
            convergence_limit=convergence_limit
        )

    def solve(self, u, p, alpha1, alpha0, num_iters=100):
        """
        Wrapper for the forward method.
        Only adding the device management.
        """
        old_device = u.device
        u = self(u.to(self.device), p.to(self.device), alpha1, alpha0, num_iters)
        return u.to(old_device)

    def forward(self, u, p, alpha1, alpha0, num_iters=100):
        """
        
        Parameters
        ----------
        u0 : torch.Tensor
            Initial guess for the solution u (denoised image). 
            Expect 2D tensor of shape [n, n] (for now).
            
        p0 : torch.Tensor
            Initial guess for the solution p?
            Expect 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of p0 is one more dimension added to the end of u0,
            and that extra last dimension is of size 2.
            
        alpha1 : float or torch.Tensor
            The regularization scalar parameter or parameters map.
            It is the threshold for the projection operator P_alpha1 (in this case)?
            
        alpha0 : float or torch.Tensor
            The regularization scalar parameter or parameters map.
            It is the threshold for the projection operator P_alpha0 (in this case)?
        """
        # # Get number of non-zeros in p
        # cnt_non_zeros = torch.sum(p != 0)
        # print(f"cnt_non_zeros = {cnt_non_zeros}")
        # # p can not be zeros
        # assert torch.sum(p) != 0, "p can not be zeros"
        
        device = self.device
        f = u.clone() # 2D shape = [n, n]
        
        # p = p.to(device)
        # # Change shape from [n, n] to [n, n, 2] for grad_h(u0) and grad_h(p0)
        # p = p.unsqueeze(-1).expand(-1, -1, 2) # shape = [n, n, 2]
        v = torch.zeros(u.shape + (2,), device=device) # Adjusted to match gradient size [n, n, 2]
        w = torch.zeros(p.shape + (2,), device=device) # Adjusted to match gradient size [n, n, 2, 2]
        
        u_bar = u.clone() # 2D shape = [n, n]
        
        p_bar = p.clone() # 3D shape = [n, n, 2]

        u = self.pdhg_tgv_solver.solve(f, u, p, u_bar, p_bar, v, w, alpha1, alpha0, num_iters)
                
        del u_bar, p_bar, v, w
        # del v_next, w_next, u_next, p_next

        return u
    
    def pointwise_norm2(self, v, dim):
        norm_2_v = torch.sqrt(torch.sum(v**2, dim=dim))
        return norm_2_v
    
    
    # ChatGPT's understanding: factor is a matrix
    # This gives better results
    def P_alpha(self, x, alpha, dim):
        # Projection
        # See page 16 in "Recovering piecewise smooth multichannel..."
        norm = self.pointwise_norm2(x, dim)
        ones = torch.ones_like(norm)
        factor_matrix = torch.maximum(ones, norm / alpha)
        for _ in range(len(dim)):
            factor_matrix = factor_matrix.unsqueeze(-1)  # Add an extra dimension for broadcasting
        projection = x / factor_matrix
        return projection
    
    
    # # My understanding: factor is a scalar
    # # This gives worse results
    # def P_alpha(self, x, alpha, dim):
    #     # Projection
    #     # See page 16 in "Recovering piecewise smooth multichannel..."
    #     norm = self.pointwise_norm2(x, dim)
    #     norm_max_element = torch.max(norm)
    #     one = torch.tensor(1.0, device=self.device)
    #     factor = torch.max(one, norm_max_element / alpha)
    #     projection = x / factor
    #     return projection
    
    
    def P_alpha1(self, v, alpha1):
        return self.P_alpha(v, alpha1, dim=(-1,))
    
    def P_alpha0(self, w, alpha0):
        return self.P_alpha(w, alpha0, dim=(-1, -2))
    
    
    # def P_alpha1(self, v, alpha1):
    #     """
    #     Scale v by alpha1 based on the provided formula using PyTorch.

    #     Parameters:
    #     v (torch.Tensor): A torch tensor of shape (n, n, 2)
    #     alpha1 (float): The alpha1 value for scaling

    #     Returns:
    #     torch.Tensor: A torch tensor of shape (n, n, 2) representing the scaled v.
    #     """
    #     # Calculate the magnitude of each vector in v
    #     # pointwise_norm = torch.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)
    #     pointwise_norm = self.pointwise_norm2(v, dim=-1)
        
    #     # Calculate the scaling factor for each element
    #     factor_matrix = torch.maximum(torch.ones_like(pointwise_norm), pointwise_norm / alpha1)
        
    #     # Scale each vector in v by the corresponding factor
    #     factor_matrix = factor_matrix.unsqueeze(-1)  # Add an extra dimension for broadcasting
    #     projection = v / factor_matrix
        
    #     return projection
    
    
    # def P_alpha0(self, w, alpha0):
    #     """
    #     Scale w by alpha0 based on the provided formula using PyTorch.

    #     Parameters:
    #     w (torch.Tensor): A torch tensor of shape (n, n, 2, 2)
    #     alpha0 (float): The alpha0 value for scaling

    #     Returns:
    #     torch.Tensor: A torch tensor of shape (n, n, 2, 2) representing the scaled w.
    #     """
    #     # Calculate the magnitude of each submatrix in w
    #     # pointwise_norm = torch.sqrt(w[:, :, 0, 0]**2 + w[:, :, 1, 1]**2 + 2 * w[:, :, 0, 1]**2)
    #     pointwise_norm = self.pointwise_norm2(w, dim=(-2, -1))
        
    #     # Calculate the scaling factor for each element
    #     factor_matrix = torch.maximum(torch.ones_like(pointwise_norm), pointwise_norm / alpha0)
        
    #     # Scale each submatrix in w by the corresponding factor
    #     factor_matrix = factor_matrix.unsqueeze(-1).unsqueeze(-1)  # Add extra dimensions for broadcasting
    #     projection = w / factor_matrix
        
    #     return projection
    
    
class PdhgTgvNumpy():
    def __init__(self):
        self.sigma = 1.0
        self.tau = 1.0
        self.theta = 1.0
        
        # Norm of K. See page 16 in "Recovering piecewise smooth multichannel..."
        convergence_limit = 0.5 * (17 + np.sqrt(33))
            
        self.pdhg_tgv_solver = PdhgTgvSolver(
            sigma=self.sigma,
            tau=self.tau,
            grad_h=self.grad_h,
            div_h=self.div_h,
            P_alpha1=self.P_alpha1,
            P_alpha0=self.P_alpha0,
            convergence_limit=convergence_limit
        )
            
    def solve(self, u, p, alpha1, alpha0, num_iters=100):
        u0 = u.copy()
        u_bar = u.copy()
        p_bar = p.copy()
        
        v = np.zeros(u.shape + (2,))
        w = np.zeros(p.shape + (2,))
        
        u = self.pdhg_tgv_solver.solve(u0, u, p, u_bar, p_bar, v, w, alpha1, alpha0, num_iters)
        
        return u
    
    def grad_h(self, u):
        """
        
        Parameters
        ----------
        u : np.ndarray
            Assume 2D tensor of shape [n, n] (for now).
            
            scalar field?
            
        Returns
        -------
        grad_u : np.ndarray
            Assume 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of the output is one more dimension added to the end of input,
            and that extra last dimension is of size 2.
            
            Gradient of the scalar field u?
        """
        # assert u.ndim == 2, f"u must be a 2D, but got {u.ndim}D"
        # Compute the gradient in both x and y directions
        grad_u = np.zeros(u.shape + (2,))
        grad_u[..., 0] = np.diff(u, axis=0, append=u[-1:, ...])
        grad_u[..., 1] = np.diff(u, axis=1, append=u[:, -1:])
        return grad_u
    
    def div_h(self, v):
        """
        
        Parameters
        ----------
        v : np.ndarray
            Assume 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of the input is one more dimension added to the end of the output,
            and that extra last dimension is of size 2.
            
            representing the vector field?
            
        Returns
        -------
        div_v : np.ndarray
            Assume 2D tensor of shape [n, n] (for now).
            
            Divergence of the vector field v?
        """
        # assert v.ndim == 3, f"v must be a 3D, but got {v.ndim}D"
        # Compute the divergence from the gradient in both x and y directions
        div_v = np.zeros(v.shape[:-1])
        div_v += np.diff(v[..., 0], axis=0, prepend=v[0:1, ..., 0])
        div_v += np.diff(v[..., 1], axis=1, prepend=v[:, 0:1, ..., 1])
        return div_v
    
    def pointwise_norm2(self, v, axis):
        norm_2_v = np.sqrt(np.sum(v**2, axis=axis))
        return norm_2_v
    
    def P_alpha(self, x, alpha, axis):
        # Projection
        # return np.clip(x, -alpha, alpha) # Thresholding / Clip act
        # See page 16 in "Recovering piecewise smooth multichannel..."
        norm = self.pointwise_norm2(x, axis)
        norm_max_element = np.max(norm)
        return x / np.max([1.0, norm_max_element / alpha])
    
    def P_alpha1(self, v, alpha1):
        return self.P_alpha(v, alpha1, axis=-1)
    
    def P_alpha0(self, w, alpha0):
        return self.P_alpha(w, alpha0, axis=(-1, -2))