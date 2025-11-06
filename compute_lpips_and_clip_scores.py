import numpy as np 
import cv2
import torch
import clip
import os 
import json
import matplotlib.pyplot as plt
import torchvision.transforms as T
import lpips
import random
from dreamsim import dreamsim
from PIL import Image

# this function will compute the clip directional similarity between the two images. 
from typing import Union
from PIL import Image 


def load_clip_model():
    # Load the CLIP model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess 

# this function will compute the clip image similiarity between the two images. 
def compute_clip_image_similarity(image1, image2, model, preprocess):
    """
    Computes the cosine similarity between two images using the CLIP model.
    Args:
        img1: PIL.Image - first image
        img2: PIL.Image - second image
        model: CLIP model
        preprocess: CLIP preprocess function
    Returns:
        similarity: float - cosine similarity between the two image embeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    image1 = preprocess(_to_pil(image1)).unsqueeze(0).to(device)
    image2 = preprocess(_to_pil(image2)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

        # Normalize the features
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
        image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_features1 @ image_features2.T).item()

    return similarity 

    
def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Converts input (file path, PIL.Image, or cv2/numpy array) to PIL.Image in RGB.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img).convert("RGB")
    return im

@torch.no_grad()
def clip_directional_similarity(
    source_prompt: str,
    target_prompt: str,
    source_image: Union[str, Image.Image, np.ndarray],
    target_image: Union[str, Image.Image, np.ndarray],
    model,
    preprocess,
    device: str = None
) -> float:
    """
    Computes CLIP directional similarity:
        cos( (E_img(target) - E_img(source)),
             (E_text(target) - E_text(source)) )
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Convert all images to PIL
    src_im = preprocess(_to_pil(source_image)).unsqueeze(0).to(device)
    tgt_im = preprocess(_to_pil(target_image)).unsqueeze(0).to(device)

    # Encode images
    e_src_img = model.encode_image(src_im)
    e_tgt_img = model.encode_image(tgt_im)

    e_src_img = e_src_img / e_src_img.norm(dim=-1, keepdim=True)
    e_tgt_img = e_tgt_img / e_tgt_img.norm(dim=-1, keepdim=True)

    d_img = e_tgt_img - e_src_img
    d_img = d_img / (d_img.norm(dim=-1, keepdim=True) + 1e-12)

    # Encode text
    text_tokens = clip.tokenize([source_prompt, target_prompt]).to(device)
    e_text = model.encode_text(text_tokens)
    e_text = e_text / e_text.norm(dim=-1, keepdim=True)

    d_txt = e_text[1:2] - e_text[0:1]
    d_txt = d_txt / (d_txt.norm(dim=-1, keepdim=True) + 1e-12)

    sim = (d_img @ d_txt.T).squeeze().item()
    return float(sim) 

def init_lpips_model():
    lpips_model = lpips.LPIPS(net='vgg')
    lpips_model.eval()
    lpips_model.to('cuda')
    return lpips_model

# this will return the dreamsim score between the two images 
def compute_dreamsim_score(img1, img2, preprocess, model):
    device = "cuda"
    img1 = preprocess(_to_pil(img1)).to(device)
    img2 = preprocess(_to_pil(img2)).to(device) 
    return model(img1, img2)

def get_dreamsim_scores(img_stack, dreamsim_model, preprocess_dreamsim):
    height, width, _ = img_stack.shape
    n_imgs = width // height
    imgs = []
    for idx in range(n_imgs):
        img = img_stack[:, idx*height:(idx+1)*height, :]
        imgs.append(img)
    
    dreamsim_scores = []
    for idx in range(1, n_imgs):
        img1 = imgs[idx-1]
        img2 = imgs[idx]
        dreamsim_scores.append(compute_dreamsim_score(img1, img2, preprocess_dreamsim, dreamsim_model).item())

    return dreamsim_scores

def compute_lpips_score(img1, img2, lpips_model):
    to_tensor = T.ToTensor()
    img1 = to_tensor(img1)
    img2 = to_tensor(img2) 
    img1 = img1.to('cuda')
    img2 = img2.to('cuda')
    return lpips_model(img1, img2)

# this will compute the lpips scores for the image stack 
def get_lpips_scores(img_stack, lpips_model):
    height, width, _ = img_stack.shape
    n_imgs = width // height
    imgs = []
    for idx in range(n_imgs):
        img = img_stack[:, idx*height:(idx+1)*height, :]
        imgs.append(img)
    
    lpips_scores = []
    for idx in range(1, n_imgs):
        img1 = imgs[idx-1]
        img2 = imgs[idx]
        lpips_scores.append(compute_lpips_score(img1, img2, lpips_model).item())

    return lpips_scores

# this will compute the dreamsim score between the adjacent images to measure the local smoothness using the dreamsim metric.
def get_dreamsim_triangle_scores(img_stack, dreamsim_model, preprocess_dreamsim):
    height, width, _ = img_stack.shape
    n_imgs = width // height
    imgs = []
    for idx in range(n_imgs):
        img = img_stack[:, idx*height:(idx+1)*height, :]
        imgs.append(img)
    
    dreamsim_triangle_scores = []
    for idx in range(1, len(imgs) - 1):
        left = imgs[idx-1]
        mid = imgs[idx]
        right = imgs[idx+1]
        
        # computing the dreamsim scores for the three chunks to be used for triangle deficit computation 
        left_distance = compute_dreamsim_score(left, mid, preprocess_dreamsim, dreamsim_model).item()
        right_distance = compute_dreamsim_score(mid, right, preprocess_dreamsim, dreamsim_model).item()
        direct_distance = compute_dreamsim_score(left, right, preprocess_dreamsim, dreamsim_model).item()

        triangle_score = (left_distance + right_distance - direct_distance)
        triangle_score = triangle_score / direct_distance 
        dreamsim_triangle_scores.append(triangle_score)
    
    return dreamsim_triangle_scores

# this will compute the dreamsim metrics. First it will compute the regular metrics in the sequence and then it will compute the triangle metrics to measure the local smoothness. 
def get_dreamsim_metrics(img_stack, dreamsim_model, preprocess_dreamsim):
    dreamsim_scores = get_dreamsim_scores(img_stack, dreamsim_model, preprocess_dreamsim)
    dreamsim_scores = np.array(dreamsim_scores)

    # renormalizing the dreamsim scores to be the change between 0 and 1
    # adding all the dreamsim changes and then dividing by the total sum to adjust for the differences between the final edit 
    sum_dreamsim = np.sum(dreamsim_scores)
    dreamsim_scores = dreamsim_scores / sum_dreamsim

    # computing the dreamsim triangle scores that will measure the local smoothness interms of the dreamsim metrics 
    avg_dreamsim = np.mean(dreamsim_scores)
    max_dreamsim = np.max(dreamsim_scores)
    std_dreamsim = np.std(dreamsim_scores)

    dreamsim_triangle_scores = get_dreamsim_triangle_scores(img_stack, dreamsim_model, preprocess_dreamsim)
    dreamsim_triangle_scores = np.array(dreamsim_triangle_scores)

    avg_dreamsim_triangle = np.mean(dreamsim_triangle_scores)
    max_dreamsim_triangle = np.max(dreamsim_triangle_scores)
    std_dreamsim_triangle = np.std(dreamsim_triangle_scores)

    return [avg_dreamsim, max_dreamsim, std_dreamsim, avg_dreamsim_triangle, max_dreamsim_triangle, std_dreamsim_triangle]


# This function will compute the lpips triangle scores to measure the local smoothness for the lpips metrics. 
def get_lpips_triangle_scores(img_stack, lpips_model):
    height, width, _ = img_stack.shape
    n_imgs = width // height
    imgs = []
    for idx in range(n_imgs):
        img = img_stack[:, idx*height:(idx+1)*height, :]
        imgs.append(img)

    # a triangle equality with computing the scores between the three images in the given triangle. 
    lpips_triangle_scores = []
    for idx in range(1, len(imgs) - 1):
        left = imgs[idx-1]
        mid = imgs[idx]
        right = imgs[idx+1]

        left_distance = compute_lpips_score(left, mid, lpips_model).item()
        right_distance = compute_lpips_score(mid, right, lpips_model).item()
        direct_distance = compute_lpips_score(left, right, lpips_model).item()
        # computing the proxy for area of the triangle and using that to measure the local smoothness 
        triangle_score = (left_distance + right_distance - direct_distance)
        # normalizing the triangle score by the length of the direct distance to keep the scores under control 
        triangle_score = triangle_score / direct_distance 
        lpips_triangle_scores.append(triangle_score)

    return lpips_triangle_scores


# computing the metrics based on the lpips scores 
def get_lpips_metrics(img_stack, lpips_model):
    lpips_scores = get_lpips_scores(img_stack, lpips_model)
    lpips_scores = np.array(lpips_scores)
    
    # renormalizing the lpips scores to be the change between 0 and 1
    # adding all the lpips changes and then dividing by the total sum, so that we capture normalized changes between the images in the stack. ----------------------------- 
    sum_lpips = np.sum(lpips_scores)
    lpips_scores = lpips_scores / sum_lpips

    avg_lpips = np.mean(lpips_scores)
    max_lpips = np.max(lpips_scores)
    std_lpips = np.std(lpips_scores)

    # computing the lpips triangle scores that will measure the local smoothness interms of the lpips metrics  ------------------------------ 
    lpips_triangle_scores = get_lpips_triangle_scores(img_stack, lpips_model)
    lpips_triangle_scores = np.array(lpips_triangle_scores)

    avg_lpips_triangle = np.mean(lpips_triangle_scores)
    max_lpips_triangle = np.max(lpips_triangle_scores)
    std_lpips_triangle = np.std(lpips_triangle_scores)

    return [avg_lpips, max_lpips, std_lpips, avg_lpips_triangle, max_lpips_triangle, std_lpips_triangle]

    

def compute_clip_continuty(img_stack, src_prompt, tgt_prompt, model, preprocess):
    # loading the clip model 
    height, width, _ = img_stack.shape
    n_imgs = width // height
    imgs = []

    for idx in range(n_imgs):
        img = img_stack[:, idx*height:(idx+1)*height, :]
        imgs.append(img) 

    # ---------------------- computing the clip image similarity for the sequence ---------------------- # 
    src_img = imgs[0]
    clip_img_sim_seq = []

    for idx in range(1, n_imgs):
        tgt_img = imgs[idx]
        clip_sim = compute_clip_image_similarity(src_img, tgt_img, model, preprocess)
        clip_img_sim_seq.append(clip_sim) 

    # ---------------------- computing the clip direction similarity fot the edit sequence ------------- # 
    clip_dir_sim_seq = []
    for idx in range(1, n_imgs):
        tgt_img = imgs[idx]
        clip_dir_sim = clip_directional_similarity(src_prompt, tgt_prompt, src_img, tgt_img, model, preprocess)
        clip_dir_sim_seq.append(clip_dir_sim) 

    clip_dir_sim_seq_norm = []
    for idx in range(0, len(clip_dir_sim_seq)): # this will be of the length 6 as we are doing 6 edits in the sequence 
        norm_factor = (idx + 1) / 6
        clip_val_norm = clip_dir_sim_seq[idx] / norm_factor
        clip_dir_sim_seq_norm.append(clip_val_norm)

    clip_dir_sim_avg = np.array(clip_dir_sim_seq_norm).mean()

    return clip_img_sim_seq, clip_dir_sim_seq, clip_dir_sim_avg


def compute_folder_metrics(folder_path, json_path, lpips_model, clip_model, preprocess_clip, dreamsim_model, preprocess_dreamsim):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # lpips_model, clip_model, preprocess are passed here as we are using the same model for all the images in the folder 
    # data = data[:100]
    # print("working with only the first 100 images for now ..")

    # data = random.sample(data, 50)

    # lists to store the metrics computed till now for the folder 
    img_sim_seq, dir_sim_seq, dir_sim_mean = [], [], []
    lpips_metrics = [] 
    dreamsim_metrics = []

    # iterating over each of the example in the dataset and getting the metrics for each of the sequence 
    for item in data:
        # loading the items from the json file to compute the metrics 
        img_nm = item['image_name']
        src_prompt = item['source_prompt']
        tgt_prompt = item['target_prompt']
        category = item['category']
        transform_instruction = item['edit_instruction'] 

        if category == 'roughness' or category == 'transparency':
            continue 
        img_stack = cv2.imread(os.path.join(folder_path, img_nm))
        # print("loaded image stack shape: {}".format(img_stack.shape))
        clip_img_sim_seq, clip_dir_sim_seq, clip_dir_sim_mean = compute_clip_continuty(img_stack, src_prompt, tgt_prompt, clip_model, preprocess_clip)

        lpips_scores = get_lpips_metrics(img_stack, lpips_model)
        dreamsim_scores = get_dreamsim_metrics(img_stack, dreamsim_model, preprocess_dreamsim)
        
        lpips_metrics.append(lpips_scores) 
        img_sim_seq.append(clip_img_sim_seq)
        dir_sim_seq.append(clip_dir_sim_seq) 
        dir_sim_mean.append(clip_dir_sim_mean)
        dreamsim_metrics.append(dreamsim_scores)
        
    # converting the list into numpy arrays
    img_sim_seq = np.array(img_sim_seq)
    dir_sim_seq = np.array(dir_sim_seq) 
    dir_sim_mean = np.array(dir_sim_mean)
    lpips_metrics = np.array(lpips_metrics)
    dreamsim_metrics = np.array(dreamsim_metrics)

    # print("dimension img sim: {}".format(img_sim_seq.shape))
    # print("dimension dir sim: {}".format(dir_sim_seq.shape))
    # print("dimension lpips: {}".format(lpips_metrics.shape))

    avg_img_sim = np.mean(img_sim_seq, axis=0)
    avg_dir_sim = np.mean(dir_sim_seq, axis=0)
    avg_dir_sim_mean = np.mean(dir_sim_mean, axis=0)
    avg_lpips_metrics = np.mean(lpips_metrics, axis=0)
    avg_dreamsim_metrics = np.mean(dreamsim_metrics, axis=0)
    
    # print("avg img sim: {}".format(avg_img_sim))
    # print("avg dir sim: {}".format(avg_dir_sim))
    # print("avg lpips: {}".format(avg_lpips_metrics)) 
    # print("avg dreamsim: {}".format(avg_dreamsim_metrics))

    return avg_img_sim, avg_dir_sim, avg_dir_sim_mean, avg_lpips_metrics, avg_dreamsim_metrics


def compute_overall_metrics():
    # This is for comparison with the marble material editing baseline ------------------------ # 
    # root_path = "/s3-data/rparihar/simple_benchmarking"
    # results_list = ['kontinuous_kontext/marble_evals_extended_loss_reweight_110K'] 
    # json_path = "../benchmark_marble_metadata_semantic_prompts.json"  
    # metrics_save_name = "baseline_marble_comparison_pie_bench_wo_roughness_transparency_loss_reweight_110K.json" 

    # This is for comparison with domain specific conceptSliders baseline ------------------------ # 
    # root_path = "/s3-data/rparihar/simple_benchmarking"
    # results_list = ['kontinuous_kontext/concept_sliders_v2_loss_reweight_110K']
    # json_path = "../cslider_v2_metadata_transformed.json"  
    # metrics_save_name = "baseline_cslider_v2_comparison_loss_reweight_110K.json" 

    # This is for comparisong with the major baslines ---------------------------------
    # root_path = "/s3-data/rparihar/simple_benchmarking/processed_results"
    # results_list = ['wan_edits', 'diffmorpher_edits', 'freemorph_edits', 'kontinuous_kontext/no_filter_110K', 'kontinuous_kontext/simple_filter_110K', 'kontinuous_kontext/simple_kl_15_110K'] 
    # json_path = "../../PIE_Bench_pp/processed_pie_bench/metadata_with_instructions.json" 

    # For the new model trained with loss modification ---------------------------------
    # root_path = "/s3-data/rparihar/simple_benchmarking"
    # results_list = ['kontinuous_kontext/loss_reweighting_kl_15_110K'] 
    # json_path = "../../PIE_Bench_pp/processed_pie_bench/metadata_with_instructions.json" 
    # metrics_save_name = "baseline_kslider_comparison_loss_reweighting_kl_15_110K.json"

    # For the new set of baselines --------------------------------- 
    # root_path = "/s3-data/rparihar/simple_benchmarking/kontinuous_kontext"
    # results_list = ['ablate_clip_orig_110K'] # 'ablate_clip_orig_110K'
    # json_path = "../../PIE_Bench_pp/processed_pie_bench/metadata_with_instructions.json" 
    # metrics_save_name = "ablate_kslider_comparison_clip_ablate_orig.json" 

    # Ablation study with the number of data points in the dataset --------------------------------- 
    root_path = ""
    results_list = ['data_11K', 'data_22K'] # 'data_55K' 
    json_path = "../../PIE_Bench_pp/processed_pie_bench/metadata_with_instructions.json" 
    metrics_save_name = "metrics_output.json"   

    # metrics_save_name = "baseline_continuty_metrics_normalized_lpips_triangle_normalized_dreamsim.json"
    avg_img_sim_list, avg_dir_sim_list, avg_dir_sim_mean_list = [], [], []
    avg_lpips_metrics_list = []
    avg_dreamsim_metrics_list = []

    lpips_model = init_lpips_model()
    clip_model, preprocess_clip = load_clip_model()
    dreamsim_model, preprocess_dreamsim = dreamsim(pretrained=True, device="cuda")
    
    # each folder is one model and we are iterating over all the models and computing their scores 
    for result in results_list:
        folder_path = os.path.join(root_path, result)
        avg_img_sim, avg_dir_sim, avg_dir_sim_mean, avg_lpips_metrics, avg_dreamsim_metrics = compute_folder_metrics(folder_path, json_path, lpips_model, clip_model, preprocess_clip, dreamsim_model, preprocess_dreamsim)
        avg_img_sim_list.append(avg_img_sim)
        avg_dir_sim_list.append(avg_dir_sim)
        avg_dir_sim_mean_list.append(avg_dir_sim_mean)
        avg_lpips_metrics_list.append(avg_lpips_metrics)
        avg_dreamsim_metrics_list.append(avg_dreamsim_metrics)
        print("computed metrics for: {}".format(result)) 

    # print("average image similarity for {} is - {}".format(results_list[0], avg_img_sim_list[0]))
    # print("average image similarity for {} is - {}".format(results_list[1], avg_img_sim_list[1]))
    # print("average image similarity for {} is - {}".format(results_list[2], avg_img_sim_list[2]))
    # print("--------------------------------")
    # print("average direction similarity for {} is - {}".format(results_list[0], avg_dir_sim_list[0]))
    # print("average direction similarity for {} is - {}".format(results_list[1], avg_dir_sim_list[1]))
    # print("average direction similarity for {} is - {}".format(results_list[2], avg_dir_sim_list[2]))
    # print("--------------------------------") 
    
    json_path = os.path.join(root_path, metrics_save_name) 
    with open(json_path, 'w') as f:
        for idx in range(len(results_list)):
            # print("lpips scores sequence: {}".format(avg_lpips_metrics_list[idx]))
            json.dump({
                "method": results_list[idx],
                "img_sim_list": avg_img_sim_list[idx].tolist(),
                "dir_sim_list": avg_dir_sim_list[idx].tolist(),
                "dir_sim_mean": avg_dir_sim_mean_list[idx].tolist(),
                "lpips_metrics_avg": avg_lpips_metrics_list[idx][0].tolist(),
                "lpips_metrics_max": avg_lpips_metrics_list[idx][1].tolist(),
                "lpips_metrics_std": avg_lpips_metrics_list[idx][2].tolist(),
                "lpips_metrics_triangle_avg": avg_lpips_metrics_list[idx][3].tolist(),
                "lpips_metrics_triangle_max": avg_lpips_metrics_list[idx][4].tolist(),
                "lpips_metrics_triangle_std": avg_lpips_metrics_list[idx][5].tolist(),
                "dreamsim_metrics_avg": avg_dreamsim_metrics_list[idx][0].tolist(),
                "dreamsim_metrics_max": avg_dreamsim_metrics_list[idx][1].tolist(),
                "dreamsim_metrics_std": avg_dreamsim_metrics_list[idx][2].tolist(),
                "dreamsim_metrics_triangle_avg": avg_dreamsim_metrics_list[idx][3].tolist(),
                "dreamsim_metrics_triangle_max": avg_dreamsim_metrics_list[idx][4].tolist(),
                "dreamsim_metrics_triangle_std": avg_dreamsim_metrics_list[idx][5].tolist()
                }, f)

    # Plot Average Image Similarity
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(results_list):
        steps = np.arange(len(avg_img_sim_list[i]))
        plt.plot(steps, avg_img_sim_list[i], label=method.replace('_', ' ').title())
        plt.scatter(steps, avg_img_sim_list[i], s=40)  # Add points at each data location
    plt.title('Average Image Similarity Across Methods')
    plt.xlabel('Step')
    plt.ylabel('CLIP Image Similarity')
    plt.legend(title='Method', loc='best')
    plt.tight_layout()
    plt.savefig('average_image_similarity_marble.png')
    plt.clf()

    # Plot Average Direction Similarity
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(results_list):
        steps = np.arange(len(avg_dir_sim_list[i]))
        plt.plot(steps, avg_dir_sim_list[i], label=method.replace('_', ' ').title())
        plt.scatter(steps, avg_dir_sim_list[i], s=40)  # Add points at each data location
    plt.title('Average Direction Similarity Across Methods')
    plt.xlabel('Step')
    plt.ylabel('CLIP Direction Similarity')
    plt.legend(title='Method', loc='best')
    plt.tight_layout()
    plt.savefig('average_direction_similarity_marble.png')
    plt.clf() 


if __name__ == "__main__":
    # computing the clip based metrics for the overall folder and plotting a plot for different methods and reporting metrics.
    compute_overall_metrics()