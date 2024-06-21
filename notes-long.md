
% \begin{minted}[frame=single,
%                framesep=3mm,
%                linenos=true,
%                xleftmargin=21pt,
%                fontsize=\small,
%                tabsize=4]{Python}
% -  report
    
%     - experiment with different datasets
    
%     - always started with training on a single image
    
%     - precaution taken
    
%     - performance: time, memory, 
    
%     - data preparation
        
%         - some images are cropped too much
        
%         - variable noise levels
        
%         - 
  
%     - performance as image size, T, params change

% - debug

%     - extract eval out of train, put them all into main

%     - check to see if data loader  should puit tensors into gpu or  not

% - record training times: log into a file for each checkpoint, or just read off wandb somehow? 

% - Implementation:
%     - up-convolution (either linear interpolation combined with normal 1x1 convolution or transposed convolution)
%         - If linear interpolation, either align corners or not
%     - double-convolution block
%         - Batch normalisation or not?
%         - Mid-channels = in-channels // 2 if using linear interpolation?
%             - But in-channels = 2 \times out-channels
%     - Clearly mention all conventions
%         - Each subsequent layer double/halve the number of channels and halve/double the image side

% - Experiments
%     - different datasets
%     - different number of blocks
%     - different initial filters number
%     - different activation (ReLU)


% GPU memory consumption is stable at around 1 GB. (It kept increasing during training before I fixed it by deleting as much temporary variables as possible.)



% I need to make sure that the U-Net does exactly what I think it does.



% I also need to understand the PDHG part.



% To be honest, I want to double-check if the training and validation iteration codes are actually correct as well.



% U-Net implementation was originally for 3D dynamic image denoising. Now I'm trying to understand and change it to work specifically for 2D static images. There are a couple things I still need to understand... But at least I got the max-pooling to work, use linear interpolation (trilinear for now, want to make it bilinear after changing everything to 2D), and most importantly, keep GPU memory consumption from rising during training!



% Next time, I want to:



% Understand and edit PDHG-Net code to work specifically for 2D static images. Currently there are a few thing I am concerned with.




% The original code treated the 5 dimensions as (mb, channels, Nx, Ny, Nt) which is like (N, C, W, H, D) in pytorch documentation notation. However, pytorch documentation actually uses the convention (N, C, D, H, W). This might have implication on the meaning of the code, especially concerning the time dimension. The width and height might be ok being swapped since we use only square input, but still need to check.



% Also, originally in dynamic image denoising code, the output of U-Net are 2 matrices, 1 for spatial dimensions (x, y) and the third for time dimension t. This is probably because that's what they are used in the PDHG algorithm. However, I don't need the time dimension in this problem. Does that mean I only need one?



% If I only need one, should I set the number of out-channels to 1 instead of 2? It seems strange to set it to 1, considering U-Net is for image segmentation?



% If I can use more than one, can I try increasing the number of out-channels to 3 or 4?




% In the end, I can only make the right decision after I understand how PDHG algorithm works.




% U-Net implementation was originally for 3D dynamic image denoising. Now I'm trying to understand and change it to work specifically for 2D static images. There are a couple things I still need to understand... But at least I got the max-pooling to work, use linear interpolation (trilinear for now, want to make it bilinear after changing everything to 2D), and most importantly, keep GPU memory consumption from rising during training!


% - sanity check:
%   - gen very little noise sigma=0.001 and see what a very good psnr and ssim and nrmse looks like

% - implement function for nrmse
% - use pytorch ignite for psnr and ssim faster


% I tried
% - Setting 1: T=32, 100 train, sigma=0.5
%     - After ? epochs, validation loss starts to increase
%     - Each epoch took about 10 seconds. I ran for 620 epochs. The total time recorded by wandb is 1h 40 minutes = 100 minutes = 6000 seconds.
%     - validation PSNR peaked at 32.8 and SSIM only at 0.88 
% - Setting 2: T=16, 200 train, sigma=[0.1, 0.5]
%     - All loss, psnr, ssim are better then Setting 1, at least in the beginning.
%         - Could this be due to a mistake somewhere in the implementation where the sum is not much different but since the number of samples, hence the batches, is higher, the average is lower?
%         - More likely, though, because there the average noise level is lower than before, so ...
%         - Lower T should make the metrics a bit worse. On the other hand, lower T means that the results depend less on the PDHG algorithm and more on the UNet, so the UNet tries harder to fit to the data. In the general the UNet is probably better than PDHG in fitting to the data since it has many times more parameters (honestly PDHG has only one parameter: the number of iterations T).
%     - Validation metrics fluctuate much more wildly compared to Setting 1. Generally training metrics fluctuate as well.
%         - Most likely because we have more training data now so ...
%     - Each epoch took about 10 seconds as well, similar to Setting 1. Makes sense since T is halved but training size is doubled.
%     - After ? minutes, training loss seems to stop decreasing, and even starts increasing
%     - At 1h 10 minutes, average validation PSNR seems around 33.5 and SSIM around 0.92
% - Setting 3: T=256, 1000 train, sigma=[0.1, 0.5]
%     - 7 minutes and the first epoch is still not done.
%         - Since T increases by 16 times and train size increases by 5 times compared to setting 2, I estimate one epoch to take about 80 times longer, meaning 800 seconds = 12 minutes. 1000 epochs take 12,000 minutes = 200 hours = 10 days!!!
%         - In the paper the max epoch time seems to be 5 seconds according to Table 4, and total of 24 hours. If this is correct then the number of epochs is 24 * 60 / 5 = 300 epochs.
%     - 25 minutes and epoch 2 is still not done. Tqdm estimates each epoch to take 11 minutes though.
%     - On the bright side, "only" takes 3 GB of VRAM, which I can afford.
%     - Uses "only" 50% Cuda load on 4090.
% - Setting 4:  T=256, 200 train, sigma=[0.1, 0.5]
%     - Estimate 10 \times 16 = 160 seconds = 3 minutes each epoch.
%     1000 epochs take 3000 minutes = 50 hours = 2 days!!!
%     - Tqdm estimates each epoch takes 140 seconds = 2.2 minutes.
% - Setting 5: T=32, 200 train, sigma=[0.1, 0.5]
%     - Estimate 10 \times 2 = 20 seconds each epoch. 1000 epochs = 20,000 seconds = 300 minutes = 5 hours
% - Later:
%     - Setting 3: T=256, 200 train, sigma=[0.1, 0.5]
%     - Setting 4: T=64, 200 train, sigma=[0.1, 0.5]
%     - Setting 5: T=128, 200 train, sigma=[0.1, 0.5]
%     - Setting 6: T=32, 200 train, sigma=[0.1, 0.5]
%     - Setting 7: T=8, 200 train, sigma=[0.1, 0.5]

% - Note (to rearrange to appropriate section later)
%     - In the paper, part 5.6.1 Choosing T at Training Time:
%         - We want to set T as high as possible.
%             - Maybe I can just use T = 256?
%         - It said that NN is not flexible enough to compensate for low T and showed experimental results.
%             - However, maybe because I used a slightly different architecture with a larger UNet, in Setting 2 with T=16 the validation results seem to be better than Setting 1 with T=32. Moreover the train size in Setting 2 is twice that of Setting 1. I expect larger train data to have even higher regularisation effect, meaning it is harder to overfit.
%                 - My conjecture is that, as the UNet gets larger enough, it will be able to compensate for low T.
%         - The section did end, however, with the sentence:
%             - However, from Figure 15 one cannot infer whether the limited reconstruction accuracy is attributable to the too low number of iterations, a resulting sub-optimal ΘT or combinations thereof.
%         - I don't know if I should point this out though. If I want to check this, I will need to experiment with different UNet architecture as well. Say 4 different configs times 4 different T values. Maybe cut down to 3 different UNet's times 3 differnt T values. T values probably should be minimum of 3, but maybe I can test only 2 UNet's? Or maybe test for a short time, just enough to see that it is clearly better?
%             - In Figure 15, it shows 70,000 backpropagations. If each epoch has 100 batches of size 1, then it is about 700 epochs.

% - Maybe increase batch size can help speed up a bit if we can run for fewer epochs, and PDHG can runs fewer times? I don't think it is the case though, since we probably want to run for the same number of epochs...

% - I think there is a sweet spot for the value of T and the size of the UNet (and also the size of training data, which properly closely relates to the size of the UNet). I'm just wondering if the sweet spot is a very large UNet and T=0 ...

% - In some (most) cases, validation metrics actually seem better than training metrics, especially in the beginning.
%     - This probably just be due to my choice of validation data. Currently I use only 8 images for validation. It is likely that those 8 images are already very close to some images in the training data, and so the average metrics are better.

% - If I want to see whether the choise of data matters, which I think it does matter a lot, I can try to train the same model of 2 different datasets (say one chest xray and the other static objects), and use one to test on the other dataset, and compare the performace to testing on its own dataset.
%     - My guess is that it will be mostly worse.
%         - UNet learns features. Since it has a finite number of filters, it can learn only that many features, and one type of images may have more of some features than the other.
%         - If we train a very very large UNet, it can learn many more features, and we can use many different types of images.

% - Is it normal that the metrics start off really good at the beginning?
%     - I think that this means the PDHG algorithm is too good and stable at reconstructing that even different lambda maps still produce similar reconstructions
%     - If I set T = 0, does that mean the UNet will basically recreate the image itself?

% \textbf{IMPORTANT!} Strategies
% - \textbf{IMPORTANT!} Do I want to fix the code now?
%     - Time is running out, and I'm feeling more confident about the UNet since I can properly to max pooling and upsample with linear interpolation. It just seems clumsy and unnecessary still using 3D instead of 2D library functions.
%     - The more important thing to consider is how the PDHG works. Will it be very different if I make it 2D? Will I output only 1 channel and uses only 1 lambda map?
%     - \textbf{IMPORTANT!} Since I have to write a report, I properly need to explain it well, which means I need to actually understand, which means I should be able to implement myself, which means I should be able to reduce the dimension to 2D in the code?
% - Fix seed = 42
%     - Should I not specify a fixed seed, and accept that it will be hard to replicate exactly?
% - Limiting the search space: I think we should 
%     - fix min T to 4, max T to 512
%         - Figures 15 and 16 show min T of 8 and max T of 256
%     - fix max sigma to 0.5, min sigma can be a bit lower than 0.1?
%         - Figure 10 in the paper only shows sigma values of 0.1, 0.2 and 0.3
%         - Part 5.2 Dynamic MRI uses 3 sigmas 0.15, 0.3, 0.45
%     - fix max training data size to 300. It just takes too long.
%         - I tried 1000, with T=32. One epoch took 1 minute.
%     - try to keep one epoch below 20 seconds.
%     - try to keep total training time below 30 hours
%         - Table 4 in the paper shows training time of 24 hours.
%         It also shows runtime of 5 seconds, not sure if this means time for 1 epoch? It shows 3 models and 4 metrics for a total of 12 combinations, and each model has 2 time values. 
%     - try to keep the number of epochs at least 500?
%         - For setting 1, I stopped the training at around epoch 600, since I saw that validation already started increasing at around epoch 200?
%     - try to keep the total number of combinations, hence experiments, under 100!!! Honestly I want to do only about 10 to 20 experiments. If each experiment runs for 1000 epochs and each epoch takes 1 minute, then one experiment takes ~ 1 day, and 20 experiments take ~ one month. And that's just for the dissertation. Doubling that to account for other tasks like coding and writing the paper, I easily run out of time... 
%         - Figure 17 shows combinations of 6 T values, 3 sigma values and 3 metrics for a total of 6 \times 3 \
%         times 3 = 54 combinations.
%         - Table 1 and 3 each reports 3 different models, 3 sigma values, and 4 metrics for a total of 36 combinations
%         - Table 2 shows 4 models, 3 sigma values and 4 metrics for a total of 48 combinations
%     - 3 metrics: NRMSE, SSIM, PSNR
%         - skimage.metrics
%             - has 10 metrics, including the above 3
%     - 1 measure: and blur metric
%         - Blur metric:
%             - skimage.measure
%                 - has ~30 measures
%             - https://scikit-image.org/docs/stable/auto_examples/filters/plot_blur_effect.html
% - In conclusion, maybe experiment with
%     - 1 sigma range [0.1 to 0.5], but test with 5 sigma values 0.1, 0.2, ..., 0.5 or imitate paper and test with only 3
%     - 6 T values: 8, 16, ..., 256
%     - 1 resolution: 256
%         - I thought about 3 resolutions: 128, 256, 512. I tried 1024 once, it was too large even for 24 GB GPU. That was before I fixed to code to free up memory though.
%         However, I think that 1 is enough for now.
%     - 1 training data size: 200? 300? At most 500, but I want to avoid this
%     - 2 models: single lambda and UNet
%         - For UNet, I could try with 4 blocks instead of 3, and initial number of filters to 16 and 64. However I don't think I have enough time.
%     - 1 Batch size of 1
%         - Maybe batch sizes 2 and 4. But I don't want to try it now either. For other problems I saw batch sizes of 64 and even 128. I don't think the batch size makes much difference to the amount of time until validation loss starts to increase. 
%     - Number of metrics are not too important. This only concerns
%         - Test data
%         - Choice of graphs
% - So in total: 6 \times 3 = 18 experiments for UNet, and 18 corresponding brute-force for PDHG. Very worried about large T and resolution
%     - Estimation: I tried T=16 and resolution=256, for 200 samples dataset. Each epoch took 10 seconds. If increase T by 16 times, resoltion by 2 times, even without increasing training data, each epoch will take 10 * 16 * 2 = 320 seconds = 5 minutes. 1000 epochs take 5000 minutes = 100 hours = 4 days! And I have in total 18 experiments!
%         - Maybe fix the resolution as well to 256 or 512. Then I have only 6 experiments.

% Goals
% - Not necessary need to be very good, but maybe psnr of 32 to 40, ssim from 0.8 to 0.95?
%     - For reference, in the paper, 
%         - Table 4: max PSNR 33.9, SSIM 0.81, NRMSE 0.071, blur effect 0.407
%         - Figure 15 line plots: best MSE 0.053
%         - Figure 17: PSNR 34 and ...
%         - Table 1: SSIM 0.927 and ...
%         - Table 2: PSNR 34.23 and ...
%         - Table 3: PSNR 39.3 for sigma = 0.1
%     - In the figures and tables, also look at the differences, aka. improvements, from the single lambda and other methods
    
% Visualisation
% - Box plots
%     - Use a test dataset of around 100 samples?
%     - For each sample, generate noises of 3 to 5 different sigmas. 
%     - Calculate the metrics and plot them
%     - Do this for each experiment (combination of hyperparameters), and the simple single lambda methods as well. 
% - Bruteforce lambda values: 200 steps within 0 to 1
% - Pick a few examples: at most 10?
%     - Pattern: Square within square, Colorwheel
%     - Animal: Baby turtle
%     - Medical: Chest xray
%     - Real-life scenes and objects: Chessboard
%     - People: Astronaut
%     - ...


% Writing report
% - The paper lists 95 references, so maybe I should also aim for 100?

    
% \end{minted}
