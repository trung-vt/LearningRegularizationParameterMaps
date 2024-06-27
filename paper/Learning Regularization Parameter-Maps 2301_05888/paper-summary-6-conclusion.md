# What we did

simple yet efficient data-driven approach to automatically select data/patient-adaptive spatial/spatio-temporal dependent regularization parameter-maps for the variational regularization approach focusing on TV-minimization. 

simple yet efficient and elegant way to combine variational methods with the versatility of deep learning-based approaches,
- an interpretable reconstruction algorithm which inherits all theoretical properties
of the scheme the network implicitly defines. 

We showed consistency results of the proposed unrolled scheme and we applied the proposed method to 
- a dynamic MRI reconstruction problem, 
- a quantitative MRI reconstruction problem, 
- a dynamic image denoising problem and 
- a low-dose CT reconstruction problem.

# Future research directions 

## Compare

for a fixed problem formulation and choice of regularization method (i.e. the TV-minimization
considered in this work) there exist several different reconstruction algorithms, all with their theo-
retical and practical advantages and limitations, see e.g. [17, 32, 42, 88].

interesting to
investigate whether our approach yields similar regularization maps regardless of the chosen reconstruction method and if not, to what extent they differ in.

## Expand to TGV and other combinations of regularizers

we have considered
the TV-minimization as the regularization method of choice. However, also TV minimization-based
methods are known to have limitations, e.g. in producing staircasing effects. We hypothesize that
the proposed method could as well be expanded to TGV-based methods [11] to overcome these
limitations.

the parameter-map learning can be applied when a combination of reg-
ularizers is considered. For example, similar to the dynamic MRI and denoising case studies, the
proposed method can be used for Hyperspectral X-ray CT, where the spatial and spectral domains
are regularized differently, see e.g., [91, 92]. 

other regularization methods as for example Wavelet-based methods [19, 29] could be considered as well, where instead of employing the finite
differences operator ∇, a Wavelet-operator Ψ would be the sparsity-transform of choice. Thereby,
the multi-scale decomposition of the U-Net which we have used in our work also naturally fits the
problem and could be utilized to estimate different parameter-maps for each different level of the
Wavelet-decomposition. 

## More sophisticated network architectures

Third, although we have used a plain U-Net [75] for the estimation of
the regularization parameter-maps, there exist nowadays more sophisticated network architectures,
e.g. transformers [56, 61], which could be potentially adopted as well. 

## Theory

from the theoretical
prospective, future work can include extension of the consistency results to stationary points instead for minimizers only as well as extension to the non-strongly convex fidelity terms in order to
cover the CT case as well.

interesting to investigate theoretically in what degree
CNN-produced artefacts in the parameter-maps can affect or create artefacts to the corresponding
reconstructions.



# Limitations

common for every unrolled NN:

## GPU memory

large GPU-memory consumption to store intermediate results and their corresponding gradients during training

## Training time

possibly long training times: repeatedly apply the forward and the adjoint operator during training

As we have seen from Figure 15, to be able to learn the regularization parameter-map with a
CNN as proposed, one must use a certain number of iterations T for the unrolled NN to
ensure that the output image of the reconstruction network has sufficiently converged to the solution
of problem (3). 

How large this number needs to be depends on the considered application as well
as the convergence rate of the unrolled algorithm which is used for the reconstruction.