import torch

def dx_forward(u, dx=1.0):
    """
    Compute the forward finite difference in the x direction.
    
    Args:
    u (torch.Tensor): Input 2D tensor.
    dx (float): Grid spacing in the x direction.

    Returns:
    torch.Tensor: Forward finite difference in the x direction with the same dimensions as the input.
    
    Example:
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dx = 1.0
    >>> dx_forward(u, dx)
    tensor([[ 3.,  3.,  3.],
            [ 3.,  3.,  3.],
            [-7., -8., -9.]])
    """
    diff_x = torch.diff(u, dim=0, append=torch.zeros_like(u[0:1, :])) / dx
    diff_x[-1, :] = -u[-1, :] / dx
    return diff_x

def dy_forward(u, dy=1.0):
    """
    Compute the forward finite difference in the y direction.
    
    Args:
    u (torch.Tensor): Input 2D tensor.
    dy (float): Grid spacing in the y direction.

    Returns:
    torch.Tensor: Forward finite difference in the y direction with the same dimensions as the input.
    
    Example:
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dy = 1.0
    >>> dy_forward(u, dy)
    tensor([[ 1.,  1., -3.],
            [ 1.,  1., -6.],
            [ 1.,  1., -9.]])
    """
    diff_y = torch.diff(u, dim=1, append=torch.zeros_like(u[:, 0:1])) / dy
    diff_y[:, -1] = -u[:, -1] / dy
    return diff_y

def dx_backward(u, dx=1.0):
    """
    Compute the backward finite difference in the x direction.
    
    Args:
    u (torch.Tensor): Input 2D tensor.
    dx (float): Grid spacing in the x direction.

    Returns:
    torch.Tensor: Backward finite difference in the x direction with the same dimensions as the input.
    
    Example:
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dx = 1.0
    >>> dx_backward(u, dx)
    tensor([[1., 2., 3.],
            [3., 3., 3.],
            [3., 3., 3.]])
    """
    diff_x = torch.diff(u, dim=0, prepend=torch.zeros_like(u[0:1, :])) / dx
    diff_x[0, :] = u[0, :] / dx
    return diff_x

def dy_backward(u, dy=1.0):
    """
    Compute the backward finite difference in the y direction.
    
    Args:
    u (torch.Tensor): Input 2D tensor.
    dy (float): Grid spacing in the y direction.

    Returns:
    torch.Tensor: Backward finite difference in the y direction with the same dimensions as the input.
    
    Example:
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dy = 1.0
    >>> dy_backward(u, dy)
    tensor([[1., 1., 1.],
            [4., 1., 1.],
            [7., 1., 1.]])
    """
    diff_y = torch.diff(u, dim=1, prepend=torch.zeros_like(u[:, 0:1])) / dy
    diff_y[:, 0] = u[:, 0] / dy
    return diff_y

def test_adjoint_property():
    """
    Test the adjoint property of forward and backward difference operators.
    """
    dx = 1.0
    dy = 1.0

    # Define two random 2D tensors
    u = torch.rand((5, 5), dtype=torch.float32)
    v = torch.rand((5, 5), dtype=torch.float32)

    # Compute forward differences
    Df_x_u = dx_forward(u, dx)
    Df_y_u = dy_forward(u, dy)

    # Compute backward differences
    Db_x_v = dx_backward(v, dx)
    Db_y_v = dy_backward(v, dy)

    # Compute inner products
    inner_product_x = torch.sum(Df_x_u * v)
    inner_product_y = torch.sum(Df_y_u * v)

    adjoint_product_x = -torch.sum(u * Db_x_v)
    adjoint_product_y = -torch.sum(u * Db_y_v)

    # Check if they are approximately equal
    assert torch.allclose(inner_product_x, adjoint_product_x, atol=1e-6), f"X-direction: {inner_product_x} != {adjoint_product_x}"
    assert torch.allclose(inner_product_y, adjoint_product_y, atol=1e-6), f"Y-direction: {inner_product_y} != {adjoint_product_y}"
    
    print(f"Test adjoint property successful!")

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()

#     # Test with the given example
#     u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
#     dx = 1.0
#     dy = 1.0

#     print("X forward difference:")
#     print(dx_forward(u, dx))

#     print("X backward difference:")
#     print(dx_backward(u, dx))

#     print("Y forward difference:")
#     print(dy_forward(u, dy))

#     print("Y backward difference:")
#     print(dy_backward(u, dy))


    def forward_diff(u):
        """
        Combine forward difference in x and y directions.
        
        Parameter
        ---------
        u : torch.Tensor
            Assume 2D tensor of shape [n, n] (for now).
            scalar field?
            
        Returns
        -------
        forward_grad_u : torch.Tensor
            Assume 3D tensor of shape [n, n, 2] (for now).
            The point is that, the shape of the output is one more dimension added to the end of input,
            and that extra last dimension is of size 2.
            Gradient of the scalar field u?
            
        Example
        -------
        >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> GradOpsTorch.forward_diff(u)
        tensor([[[3, 3, 3],
                 [3, 3, 3],
                 [0, 0, 0]],
        <BLANKLINE>
                [[1, 1, 0],
                 [1, 1, 0],
                 [1, 1, 0]]])
        """
        # assert u.dim() == 2, f"u must be a 2D tensor, but got {u.dim()}D tensor"
        grad_x = torch.diff(u, n=1, dim=0, append=u[-1:, ...])
        grad_y = torch.diff(u, n=1, dim=1, append=u[:, -1:])
        forward_grad_u = torch.stack((grad_x, grad_y))
        return forward_grad_u
    
    def backward_diff(u):
        """
        Combine backward difference in x and y directions.
        
        Parameter
        ---------
        u : torch.Tensor
            Assume 3D tensor of shape [2, n, n] (for now).
            The point is that, the shape of the input is one more dimension added to the end of the output,
            and that extra last dimension is of size 2.
            representing the vector field?
            
        Returns
        -------
        backward_div_u : torch.Tensor
            Assume 2D tensor of shape [n, n] (for now).
            Divergence of the vector field u?
            
        Example
        -------
        >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> GradOpsTorch.backward_diff(u)
        tensor([[[1, 2, 3],
                 [3, 3, 3],
                 [1, 1, 1]],
        <BLANKLINE>
                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]])
        """
        raise NotImplementedError("Not implemented yet")