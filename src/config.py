import torch

# TODO: Put in a config .yaml file


DISABLING_TESTS = False
# DISABLING_TESTS = True   # Disable tests for less output



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # print(f"Using {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    # print(f"Using {torch.backends.mps.get_device_name(0)} with MPS")
else:
    DEVICE = torch.device("cpu")
    # print("Using CPU")

torch.set_default_device(DEVICE)



BASE_DIR = "data/chest_xray"


BATCH_SIZE = 1