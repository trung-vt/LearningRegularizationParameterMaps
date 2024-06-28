import torch

def dx_forward(u):
    """
    Computes the forward difference in the x direction.
    
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dx_forward(u)
    tensor([[3., 3., 3.],
            [3., 3., 3.],
            [0., 0., 0.]])
    """
    N1, N2 = u.shape
    diff_x = torch.zeros_like(u)
    diff_x[:-1, :] = u[1:, :] - u[:-1, :]
    return diff_x

def dy_forward(u):
    """
    Computes the forward difference in the y direction.
    
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dy_forward(u)
    tensor([[1., 1., 0.],
            [1., 1., 0.],
            [1., 1., 0.]])
    """
    N1, N2 = u.shape
    diff_y = torch.zeros_like(u)
    diff_y[:, :-1] = u[:, 1:] - u[:, :-1]
    return diff_y

def dx_backward(u):
    """
    Computes the backward difference in the x direction.
    
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dx_backward(u)
    tensor([[ 1.,  2.,  3.],
            [ 3.,  3.,  3.],
            [-4., -5., -6.]])
    """
    N1, N2 = u.shape
    diff_x = torch.zeros_like(u)
    diff_x[0, :] = u[0, :]
    diff_x[1:-1, :] = u[1:-1, :] - u[:-2, :]
    diff_x[-1, :] = -u[-2, :]
    return diff_x

def dy_backward(u):
    """
    Computes the backward difference in the y direction.
    
    >>> u = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    >>> dy_backward(u)
    tensor([[ 1.,  1., -2.],
            [ 4.,  1., -5.],
            [ 7.,  1., -8.]])
    """
    N1, N2 = u.shape
    diff_y = torch.zeros_like(u)
    diff_y[:, 0] = u[:, 0]
    diff_y[:, 1:-1] = u[:, 1:-1] - u[:, :-2]
    diff_y[:, -1] = -u[:, -2]
    return diff_y

def test_adjoint_property():
    """
    Test the adjoint property for forward and backward difference operators.
    
    >>> adjoint_x, adjoint_y = test_adjoint_property()
    >>> assert abs(adjoint_x) < 1e-6, f"Adjoint property failed for x: {adjoint_x}"
    >>> assert abs(adjoint_y) < 1e-6, f"Adjoint property failed for y: {adjoint_y}"
    """
    N1, N2 = 4, 4  # Example dimensions
    u = torch.rand((N1, N2))
    v = torch.rand((N1, N2))

    fwd_x_u = dx_forward(u)
    bwd_x_v = dx_backward(v)
    adjoint_x = torch.sum(fwd_x_u * v) + torch.sum(u * bwd_x_v)
    
    fwd_y_u = dy_forward(u)
    bwd_y_v = dy_backward(v)
    adjoint_y = torch.sum(fwd_y_u * v) + torch.sum(u * bwd_y_v)
    
    return adjoint_x.item(), adjoint_y.item()

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
#     print("Adjoint property test results:", test_adjoint_property())
