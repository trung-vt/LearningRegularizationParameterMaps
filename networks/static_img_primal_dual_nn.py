import torch
import torch.nn as nn
import torch.nn.functional as F

from .pdhg import PDHG

class StaticImagePrimalDualNN(nn.Module):
    def __init__(
        self,
        T=128,
        cnn_block=None,
        mode="lambda_cnn",
        up_bound=0,
        phase="training",
        device="cuda",
    ):
        """
        StaticImagePrimalDualNN
        
        Parameters
        ----------
        T : int
            Number of iterations.

        cnn_block : nn.Module
            CNN block to estimate the lambda regularization map.

        mode : str
            Mode. Default is "lambda_cnn".

        up_bound : int
            Upper bound. Default is 0.

        phase : str    
            Phase. "training" or "testing". Default is "training".

        device : str
            Device. Default is "cuda".
        """

        super(StaticImagePrimalDualNN, self).__init__()
        self.device = device
        self.pdhg = PDHG()

        # gradient operators and clipping function
        dim = 3  # TODO: What does this 3 mean?? Does this include the time dimension?
        # dim = 2 # TODO: Will this limit to 2 spatial dimensions?



        if mode == "lambda_cnn":
            # the CNN-block to estimate the lambda regularization map
            # must be a CNN yielding a two-channeld output, i.e.
            # one map for lambda_cnn_xy and one map for lambda_cnn_t
            self.cnn = cnn_block    # NOTE: This is actually the UNET!!! (At least in this project)
            self.up_bound = torch.tensor(up_bound)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode

        # distinguish between training and test phase;
        # during training, the input is padded using "reflect" padding, because
        # patches are used by reducing the number of temporal points;
        # while testing, "reflect" padding is used in x,y- direction, while
        # circular padding is used in t-direction
        self.phase = phase

    def get_lambda_cnn(self, x):
        # padding
        # arbitrarily chosen, maybe better to choose it depending on the
        # receptive field of the CNN or so;
        # seems to be important in order not to create "holes" in the
        # lambda_maps in t-direction
        npad_xy = 4
        # npad_t = 8
        npad_t = 0 # TODO: Time dimension should not be necessary for single image input.
        # I changed the npad_t to 0 so that I can run on single image input without change the 3D type config. It seems that the number of frames must be greater than npad_t?

        pad = (npad_t, npad_t, npad_xy, npad_xy, npad_xy, npad_xy)

        if self.phase == "training":
            x = F.pad(x, pad, mode="reflect")

        elif self.phase == "testing":
            pad_refl = (0, 0, npad_xy, npad_xy, npad_xy, npad_xy)
            pad_circ = (npad_t, npad_t, 0, 0, 0, 0)

            x = F.pad(x, pad_refl, mode="reflect")
            x = F.pad(x, pad_circ, mode="circular")

        # estimate parameter map
        lambda_cnn = self.cnn(x) # NOTE: The cnn is actually the UNET block!!! (At least in this project)

        # crop
        neg_pad = tuple([-pad[k] for k in range(len(pad))])
        lambda_cnn = F.pad(lambda_cnn, neg_pad)

        # double spatial map and stack
        lambda_cnn = torch.cat((lambda_cnn[:, 0, ...].unsqueeze(1), lambda_cnn), dim=1)

        # constrain map to be striclty positive; further, bound it from below
        if self.up_bound > 0:
            # constrain map to be striclty positive; further, bound it from below
            lambda_cnn = self.up_bound * self.op_norm_AHA * torch.sigmoid(lambda_cnn)
        else:
            lambda_cnn = 0.1 * self.op_norm_AHA * F.softplus(lambda_cnn)

        del x, npad_xy, npad_t, pad, neg_pad # Explicitly free up (GPU) memory to be safe

        return lambda_cnn


    def forward(
            self, x, lambda_map=None, 
            # lambda_reg_container=None,
    ):
        if lambda_map is None:
            # estimate lambda reg from the image
            lambda_reg = self.get_lambda_cnn(x)
        else:
            lambda_reg = lambda_map

        # if lambda_reg_container is not None:
        #     assert type(lambda_reg_container) == list, f"lambda_reg_container should be a list, not {type(lambda_reg_container)}"
        #     lambda_reg_container.append(lambda_reg) # For comparison

        x.to(self.device)
        x1 = self.pdhg(x, lambda_reg, self.T)

        del lambda_reg, x # Explicitly free up (GPU) memory to be safe

        return x1
