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
from cleanfid import fid 
import torch_fidelity 


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

# This function will load the source image, splits it into the number of edits and saves the individual edit images in the target folder 
def process_imgs_for_fid(folder_path, target_folder_path):
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)
        print("created target folder: {}".format(target_folder_path))

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        height, width, _ = img.shape
        n_cols = 7
        col_width = width // n_cols

        img_splits = [] 
        for i in range(n_cols):
            x1 = i * col_width
            x2 = (i + 1) * col_width if i < n_cols - 1 else width

            crop = img[:, x1:x2, :] 
            img_splits.append(crop)

        # taking the intermediate splits for the fid computations 
        img_splits_intermediate = img_splits[1:-1] 
        
        for i, img_split in enumerate(img_splits_intermediate):
            img_split_path = os.path.join(target_folder_path, img_name.split(".")[0] + "_" + str(i) + ".png")
            cv2.imwrite(img_split_path, img_split)


def process_fid_images():
    # This is for comparison with the marble material editing baseline ------------------------ # 
    fid_image_path = "path for fid image folder"

    root_path = "root path for the output folder"
    results_list = ['loss_reweighting_kl_15_110K']

    # each folder is one model and we are iterating over all the models and computing their scores 
    for result in results_list:
        folder_path = os.path.join(root_path, result)
        target_folder_path = os.path.join(fid_image_path, result)
        print("processing folder: {}".format(folder_path))

        process_imgs_for_fid(folder_path, target_folder_path)
        print("saving folder: {}".format(target_folder_path)) 

# this will compute the metrics all in one go 
def compute_fast_kid(source_fld):
    reference_fld = "path for the reference folder"

    metrics = torch_fidelity.calculate_metrics(input1=reference_fld, input2=source_fld, kid=True, kid_subset_size=100, cuda=True)
    print(metrics['kernel_inception_distance_mean'], metrics['kernel_inception_distance_std'])
    return [metrics['kernel_inception_distance_mean'], metrics['kernel_inception_distance_std']]

# compute the fid in comparison to the omini-control dataset for now 
def compute_single_kid(source_fld):
    reference_fld = "path for the reference folder"
    score = fid.compute_kid(reference_fld, source_fld, mode="clean")
    return score

# This fucntion will compute the fid scores between the source and the target folder images 
def compute_fid_scores():
    root_path = "root path for the computation of the fid metric scores"
    # result_list = ['benchmark_marble_edits', 'kontinuous_kontext/marble_evals_extended', 'cslider_edits_v2', 'kontinuous_kontext/concept_sliders', 'wan_edits', 'diffmorpher_edits', 'freemorph_edits', 'kontinuous_kontext/no_filter_110K', 'kontinuous_kontext/simple_filter_110K', 'kontinuous_kontext/simple_kl_15_110K']
    result_list = ['kontinuous_kontext/loss_reweighting_kl_15_110K']  

    fid_scores = []
    for result in result_list:
        source_folder_path = os.path.join(root_path, result)
        print("computing fid scores for folder: {}".format(source_folder_path))

        # fid_score = compute_single_kid(source_folder_path)
        kid_score = compute_fast_kid(source_folder_path)
        data_item = {'result': result, 'kid_score': kid_score}
        print("kid score: {}".format(kid_score))
        fid_scores.append(data_item)

    with open("kid_scores.json", "w") as f:
        json.dump(fid_scores, f)

if __name__ == "__main__":
    # computing the clip based metrics for the overall folder and plotting a plot for different methods and reporting metrics.
    process_fid_images()

    # computing the fid scores for all the folders by iterating over the folders and comparing them with the omini-ctrl data.
    # compute_fid_scores()