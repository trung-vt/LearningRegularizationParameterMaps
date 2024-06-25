Try `up_bound=1` to multiply with a sigmoid.
This will force the regularization params to be between 0 and 1.
I am not worried about the model trying to give very large positive values so that the output is close to 1, since it is almost certain that a high lambda is bad and will increase the loss.
What I am more worried about is the other end of the spectrum.
It is likely that, for some local lambdas, the optimal values are very close to 0.
This will make the model output very small negative values,
which can be unstable?


What if we set T=0 and and see whether the network can denoise by itself?
In this case I don't like the idea of using sigmoid. The image will mostly have very bright pixels (close to 1) or very dark pixels (close to 0). 
In order to make the result 0 or 1,
the unet must output a very negative or very positive value.
This is not stable?


Make the code creates model name before init wandb and adds to config for easier finding later.

Fix unet code to allow for deeper networks. Right now 4 layers has error.