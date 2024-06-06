
- Use more channels in the output layers = Segment the image further
    - Can we make use of the information that these are different segments into the PDHG?

- If we have a denoising function, then ideally it should not make any change to a clean image.
  Theoretically this is what the PDHG algorithm does.
  However, is it the same with the U-Net and the regularisation parameters it finds?
  If we run a forward pass with a clean image, will all the regularisation parameters be zero?
  If we run a forward pass on a denoised image, will all the regularisation parameters be zero? If not, will they decrease?
    - Test this by repeatedly calling U-Net
        - to a noisy image, see if it converges at some point
        - to a clean image, see if it remains the same

- I start