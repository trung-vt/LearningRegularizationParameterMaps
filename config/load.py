

def get_sigmas(sigmas_str:str):
    # Convert string "[0.1, 0.15, 0.2, 0.25, 0.3]" to list [0.1, 0.15, 0.2, 0.25, 0.3]
    sigmas = [float(sigma) for sigma in sigmas_str[1:-1].split(", ")]
    return sigmas