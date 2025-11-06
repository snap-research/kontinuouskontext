"""
This script will be used for filtering the dataset based on the lpips scores between the elements of the image stack and save them in a json file for later use case in dataloader.
""" 

import torch 
import lpips 
from dreamsim import dreamsim
import json
from PIL import Image
import os 
import torchvision.transforms as T
import sys
from typing import Union

# # A helper function to conver the image into a PIL image for preprocessing for the model 
# def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
#     """
#     Converts input (file path, PIL.Image, or cv2/numpy array) to PIL.Image in RGB.
#     """
#     img = img.type(np.uint8)
#     print("image min max: {}, {}".format(img.min(), img.max()))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     im = Image.fromarray(img).convert("RGB")
#     return im

# loading the vgg model for computing the lpips scores 
def load_lpips_model():
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    loss_fn_vgg.eval()
    return loss_fn_vgg.to("cuda:0")  

# loading the dreamsim model for computing the dreamsim scores 
def load_dreamsim_model():
    dreamsim_model, preprocess_dreamsim = dreamsim(pretrained=True, device="cuda")
    return dreamsim_model, preprocess_dreamsim

# this function will compute the lpips between any two given images and returns the distance between them. 
def compute_lpips(image_1, image_2, lpips_model):
    image1 = T.ToTensor()(image_1).to("cuda:0")
    image2 = T.ToTensor()(image_2).to("cuda:0")
    distance_lpips = lpips_model(image1, image2)
    return distance_lpips.cpu()

# this will return the dreamsim score between the two images | already the input images are pil images, we don't have to conver them 
def compute_dreamsim_score(img1, img2, preprocess, model):
    device = "cuda"
    img1 = preprocess(img1).to(device)
    img2 = preprocess(img2).to(device) 
    return model(img1, img2)

# This function will help us to compute lpips between the images in the stack and that will be used for filtering the data based on the given thresholds.
def evaluate_lpips_scores(list_images, lpips_model):                # Computing the lpips distance between the images from the stack that will be used for filtering 
    score_kontext_edit = compute_lpips(list_images[-1], list_images[0], lpips_model) # taking the first and the last image for checking if the flux has done the correct editing or not 
    score_edit_inversion = compute_lpips(list_images[-1], list_images[-2], lpips_model) # taking the first and the second last image for checking the quality of flux inversion 
    score_inversion_edit = compute_lpips(list_images[-2], list_images[1], lpips_model) # taking the second and the second last from the sequence for checking the edit sequence and its quality 

    return score_kontext_edit.item(), score_edit_inversion.item(), score_inversion_edit.item() 

# This function will disintegrate the image into a list of images, the stack has images horizontally stacked with each other and we want to crop the images 
def disintegrate_image_stack(image_stack):
    # this will decompose a given image stack into the individual images with the original source and target and the intermediate images with the inversions and interpolations 
    width, height = image_stack.size
    n_imgs = width // height
    img_width = width // n_imgs
    img_height = height 

    list_stacked_images = []
    # Iterating over the number of images for the edits and saving cropped images in the list 
    for i in range(n_imgs):
        list_stacked_images.append(image_stack.crop((img_width * i, 0, img_width * (i + 1), img_height)))
    
    return list_stacked_images  

# Computing the lpips triangle score as an additional metric to be used during filtering of data 
def compute_lpips_triangle_score(start, mid, end, lpips_model):
    delta_left = compute_lpips(start, mid, lpips_model)
    delta_right = compute_lpips(mid, end, lpips_model)
    delta_direct = compute_lpips(start, end, lpips_model)
    
    # computing the delta by adding the left and right and subtracting the direct delta 
    triangle_score = (delta_left + delta_right - delta_direct)
    # normalizing the triangle score by the direct delta to have a normalized values for the triangle inequality 
    triangle_score = triangle_score / delta_direct 
    return triangle_score


# This function will compute the dreamsim triangle score between the set of three input images 
def compute_dreamsim_triangle_score(start, mid, end, preprocess_dreamsim, dreamsim_model):
    delta_left = compute_dreamsim_score(start, mid, preprocess_dreamsim, dreamsim_model)
    delta_right = compute_dreamsim_score(mid, end, preprocess_dreamsim, dreamsim_model)
    delta_direct = compute_dreamsim_score(start, end, preprocess_dreamsim, dreamsim_model)
    
    # computing the delta by adding the left and right and subtracting the direct delta 
    triangle_score = (delta_left + delta_right - delta_direct)
    # normalizing the triangle score by the direct delta to have a normalized values for the triangle inequality 
    triangle_score = triangle_score / delta_direct 
    return triangle_score


# iterating over the list of images and adding the lpips scores between the adjacent images into the list 
def evaluate_lpips_sequence(list_images, lpips_model):
    lpips_sequence = [] 
    for i in range(len(list_images) - 1):
        lpips_val = compute_lpips(list_images[i], list_images[i+1], lpips_model)
        lpips_sequence.append(lpips_val.item());

    lpips_sequence_triangle = []
    for i in range(len(list_images) - 2):
        lpips_val = compute_lpips_triangle_score(list_images[i], list_images[i+1], list_images[i+2], lpips_model)
        lpips_sequence_triangle.append(lpips_val.item())

    return lpips_sequence, lpips_sequence_triangle

# This function will compute the dreamsim scores for the first order and the second order over the list of images 
def evaluate_dreamsim_sequence(list_images, dreamsim_model, preprocess_dreamsim):
    dreamsim_sequence_first_order = []
    for i in range(len(list_images) - 1):
        dreamsim_val = compute_dreamsim_score(list_images[i], list_images[i+1], preprocess_dreamsim, dreamsim_model)
        dreamsim_sequence_first_order.append(dreamsim_val.item())

    dreamsim_sequence_triangle = []
    for i in range(len(list_images) - 2):
        start = list_images[i]
        mid = list_images[i+1]
        end = list_images[i+2]

        dreamsim_val = compute_dreamsim_triangle_score(start, mid, end, preprocess_dreamsim, dreamsim_model)
        dreamsim_sequence_triangle.append(dreamsim_val.item())

    return dreamsim_sequence_first_order, dreamsim_sequence_triangle

# This function will iterate over all the images in the metadata json file and compute the lpips scores of the three types for all the images and saves them into a .json file.
def compute_metrics_scores(dataset_meta_data, image_dirs, save_json_path, lpips_model, dreamsim_model, preprocess_dreamsim, start_idx, end_idx):
    # slicing the dataset and will work with only that chunk of the dataset in this particular run 
    print("complete dataset length: {}".format(len(dataset_meta_data))) 

    dataset_meta_data = dataset_meta_data[start_idx:end_idx]
    print(f"working with the dataset from range: {start_idx} to {end_idx}") 
    print("dataset_meta_data length: {}".format(len(dataset_meta_data))) 

    # iterating over the dataset and computing the lpips scores for the images in the stack 
    for idx in range(len(dataset_meta_data)):
        # if (idx > 150): 
        #     break 

        try:
            image_meta_data_sample = dataset_meta_data[idx] # Extracting one row from the meta data, we will use that to get the image name and the edit instruction used for training 
            # print("image_meta_data_sample: {}".format(image_meta_data_sample)) 
            
            # Extacting the image name and the edit instruction from the meta data to perform the processing 
            image_name = image_meta_data_sample['image_name'] # getting the image name 

            # currently we are interpolating from the dataset that have 7 images stacked togather and there is a starting and the ending image that are the real images in the dataset 
            n_edits = 7 

            image_stack_path = os.path.join(image_dirs, image_name[:-4] + '_nsamples_'+str(n_edits)+'.png') # currently our dataset has 7 intermediate images for the interpolation and hence, it is used in the dataloader 
            # print("image_stack_path: {}".format(image_stack_path)) 
            img_stack = Image.open(image_stack_path) 
            # print("image loaded: {}".format(img_stack.size)) 
            list_images_stack = disintegrate_image_stack(img_stack) # full list with n+2 images, n are the intermediate edits and the others are the first and the last edits from the real source images 

            # for testing if the images are formed correctly or not 
            # img_stack.save('../src_data/debug_lpips/'+str(idx)+'.png' )
            # for idx_img in range(len(list_images_stack)):
            #     img = list_images_stack[idx_img]
            #     img_path = '../src_data/debug_lpips/'+str(idx)+'_'+str(idx_img)+'.png' 
            #     img.save(img_path)

            # this function will compute the lpips scores between the adjacent images in the squence and returns a list for all the scores 
            lpips_sequence, lpips_sequence_triangle = evaluate_lpips_sequence(list_images_stack, lpips_model) # just computing the lpips between the intermediate images and throwing away the end images 
            dreamsim_sequence_first_order, dreamsim_sequence_triangle = evaluate_dreamsim_sequence(list_images_stack, dreamsim_model, preprocess_dreamsim) 
            
            print("lpips sequence: {}".format(lpips_sequence))  
            print("dreamsim first order sequence: {}".format(dreamsim_sequence_first_order))
            print("dreamsim triangle sequence: {}".format(dreamsim_sequence_triangle)) 

            # Computing the lpips scores for the the given image stack, we will use these to flag the sample as relevant or not 
            lpips_kontext_edit, lpips_edit_inversion, lpips_inversion_edit = evaluate_lpips_scores(list_images_stack, lpips_model)
            # Add the computed metrics to the current metadata row
            image_meta_data_sample['lpips_kontext_edit'] = lpips_kontext_edit
            image_meta_data_sample['lpips_edit_inversion'] = lpips_edit_inversion
            image_meta_data_sample['lpips_inversion_edit'] = lpips_inversion_edit
            image_meta_data_sample['lpips_sequence'] = lpips_sequence
            image_meta_data_sample['lpips_sequence_triangle'] = lpips_sequence_triangle

            image_meta_data_sample['dreamsim_sequence_first_order'] = dreamsim_sequence_first_order
            image_meta_data_sample['dreamsim_sequence_triangle'] = dreamsim_sequence_triangle 

        except Exception as e:
            print("error in processing image: {}, currently proceeding with image".format(image_name))
            print("error: {}".format(e))
            continue

        if (idx % 100 == 0):
            print("processed the {}th image".format(idx)) 

    # After processing all rows, save the updated metadata as a new .json file
    print("writing the meta data with the computed lpips score to the directory: {}".format(save_json_path))
    
    save_json_name = "datasetv2_metadata_" + str(start_idx) + "_" + str(end_idx) + "_with_lpips.json"
    with open(os.path.join(save_json_path, save_json_name), "w") as f:   
        json.dump(dataset_meta_data, f, indent=4) 

def process_dataset(): 
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    # image_dirs = "/s3-data/rparihar/omini-ctrl-dataset/freemorph_data_20K_interp"  
    # dataset_meta_data = json.load(open("/s3-data/rparihar/omini-ctrl-dataset/0_20000_edit_pairs.json"))

    image_dirs = "/s3-data/rparihar/processed-kontext-dataset-v2/morphing_data/"
    dataset_meta_data = json.load(open("/s3-data/rparihar/processed-kontext-dataset-v2/updated_image_metadata_combined.json"))
    
    # save_json_path = "/s3-data/rparihar/omini-ctrl-dataset/" 
    save_json_path = "../src_data/metrics_jsons" 
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)
        print(f"created the directory for lpips at: {save_json_path}") 

    # loading the lpips model 
    loss_fn_vgg = load_lpips_model().to("cuda:0")
    dreamsim_model, preprocess_dreamsim = load_dreamsim_model()

    # this will compute both the lpips and dreamsim scores, the dreamsim scores will be computed for the first order and the second order triangle metrics over the list 
    compute_metrics_scores(dataset_meta_data, image_dirs, save_json_path, loss_fn_vgg, dreamsim_model, preprocess_dreamsim, start_idx, end_idx)

if __name__ == "__main__":
    process_dataset()

        