
% \section{Experimental setups}

% I carried out a few experiments on a few different datasets:

% \subsection{Chest Xray data set}

% I used 3-block UNet 32-64-128-256. 

% Common settings:

% \begin{minted}[frame=single,
%                framesep=3mm,
%                linenos=true,
%                xleftmargin=21pt,
%                fontsize=\small,
%                tabsize=4]{Json}
% {
%     "resize_square": 256,
%     "min_sigma": 0.1,
%     "max_sigma": 0.5,
%     "batch_size": 1,
%     "random_seed": 42,

%     "in_channels": 1,
%     "out_channels": 2,
%     "init_filters": 32,
%     "n_blocks": 3,
%     "activation": "LeakyReLU",
%     "downsampling_kernel": (2, 2, 1),
%     "downsampling_mode": "max",
%     "upsampling_kernel": (2, 2, 1),
%     "upsampling_mode": "linear_interpolation",

%     "optimizer": "Adam",
%     "learning_rate": 1e-4,
%     "loss_function": "MSELoss",

%     "up_bound": 0,
    
%     "device": "cuda:0",

%     "wandb_mode": "online",
%     "save_epoch_wandb": 10_000,
%     "save_epoch_local": 10,
%     "save_dir": "tmp_2",
% }
% \end{minted}


% \subsubsection{Experiment 1 (ethereal-tree-30)}

%     \url{https://wandb.ai/wof/chest_xray/runs/ad1vzkck}

%     \begin{minted}[frame=single,
%                    framesep=3mm,
%                    linenos=true,
%                    xleftmargin=21pt,
%                    fontsize=\small,
%                    tabsize=4]{Json}
    
%     {
%         "train_num_samples": 200,
%         "val_num_samples": 8,
    
%         "T": 16,
%     }
    
%     {
%       "_step": 3311,
%       "training SSIM": 0.920365142135862,
%       "validation SSIM": 0.9260838512155668,
%       "validation_loss": 0.0004371008544694632,
%       "_runtime": 5877.103338479996,
%       "_timestamp": 1717753962.2061036,
%       "training PSNR": 33.426429748535156,
%       "training_loss": 0.0004778142458235379,
%       "validation PSNR": 33.67319107055664
%     }
    
%     \end{minted}
