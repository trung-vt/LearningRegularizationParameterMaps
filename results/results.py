import pandas as pd



class Result:
    def __init__(self, is_greater:bool):
        self.is_greater = is_greater
        self.best_metric = 0 if is_greater else float("inf")
        self.best_lambda = None
        
    def update(self, _lambda, metric):
        if self.is_greater:
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_lambda = _lambda
        else:
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_lambda = _lambda



class SampleResult(pd.DataFrame):
    def __init__(self):
        pd.DataFrame.__init__(self, columns=["lambda", "loss", "psnr", "ssim", "denoised"])
        self.best_loss = Result(is_greater=False)
        self.best_psnr = Result(is_greater=True)
        self.best_ssim = Result(is_greater=True)
        
    def update_lambda(self, _lambda, cmp_results, denoised):
        loss, psnr, ssim = cmp_results
        self.best_loss.update(_lambda, loss)
        self.best_psnr.update(_lambda, psnr)
        self.best_ssim.update(_lambda, ssim)
        self.append({
            "lambda": _lambda,
            "loss": loss,
            "psnr": psnr,
            "ssim": ssim,
            "denoised": denoised,
        })
        
        
    
def brute_force_scalar_reg(sample, model, lambdas, cmp_images, process_denoised, sample_result):
    noisy, clean, output_path = sample
    noisy, clean = transform(sample)
    sample_result = ResultDataFrame()
    # Noisy and clean images must be the same size
    assert noisy.shape == clean.shape, f"Noisy and clean images have different sizes!\nnoisy.shape: {noisy.shape}, clean.shape: {clean.shape}"
    # Assert that the dimensions of the noisy image is accepted by the model
    try:
        model(noisy, 0)
    except Exception as e:
        raise ValueError(f"Model does not accept noisy image: {e.shape}")
    
    for _lambda in lambdas:
        denoised = model(noisy, _lambda)
        cmp_results = cmp_images(denoised, clean)
        denoised = process_denoised(denoised, _lambda, output_path)
        sample_result.update_lambda(_lambda, cmp_results, denoised)
    
    
    
def process_samples(sample_collection, model, lambdas, transform, cmp_images, process_denoised, num_threads:int=1):
    def get_sample_result(sample):
        brute_force_scalar_reg(
            sample, model, lambdas, cmp_images, process_denoised, sample_result
        )
        return sample_result
    
    from multiprocessing.dummy import Pool as ThreadPool
    # https://stackoverflow.com/questions/2846653/how-do-i-use-threading-in-python
    pool = ThreadPool(num_threads)
    print(f"Multiprocessing {len(sample_collection)} samples in {num_threads} threads")
    sample_results = pool.map(get_sample_result, sample_collection)
    return sample_results



def process(
    data_loader, model_path, 
    min_lambda, max_lambda, num_lambdas, processed_lambdas,
    transform, compare, 
    output_path,
    num_threads:int=1,
):
    import torch
    model = torch.load(model_path)
    lambdas = torch.linspace(min_lambda, max_lambda, num_lambdas)
    # Remove processed lambdas
    lambdas = [l for l in lambdas if l not in processed_lambdas]
    
    def save_result(sample_result, i):
        sample_result_path = f"{output_path}"
    
    sample_results = process_samples(data_loader, model, lambdas, transform, compare, save_result, num_threads)
    return sample_results