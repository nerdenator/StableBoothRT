### script to contain general utility functions that I'll need

import gc
import torch
import json
import gradio as gr
import cv2 as cv
from PIL import Image
import numpy as np
import os
import re
from datetime import datetime
import subprocess
import platform
import sys

# diffusers objects
from diffusers import (
                       AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline, 
                       StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, 
                       LCMScheduler, TCDScheduler )

from safetensors.torch import load_file # for lightning


############
############ CONSTANTS (modify these!)
############


#### IMPORTANT FILE LOCATIONS [MODIFY THESE]
########

# ## SD 1.5 Models
LCM_Dv7_MODEL_LOCATION        = '/Users/rolando/Documents/PROJECTS/YouTube/DIYAI_WebcamDiffusion/tutorial_scripts/models/LCM_Dreamshaper_v7' #
LCM_Dv8_MODEL_LOCATION        = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/dreamshaper-8/"
CONTROLNET_CANNY_LOCATION     = "/Users/rolando/Documents/PROJECTS/YouTube/DIYAI_WebcamDiffusion/tutorial_scripts/models/control_v11p_sd15_canny" 

## SDXL Models
SDXLTURBO_MODEL_LOCATION      = '/Users/rolando/Documents/PROJECTS/YouTube/DIYAI_WebcamDiffusion/tutorial_scripts/models/sdxl-turbo'
SDXLL_MODEL_LOCATION          = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/SDXL-Lightning"
SDXL_BASEMODEL_LOCATION       = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/stable-diffusion-xl-base-1.0"
SDXLL_CKPT_LOCATION           = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/SDXL-Lightning/sdxl_lightning_2step_unet.safetensors" # Use the correct ckpt for your step setting!
SDXL_CANNY_CONTROLNET_LOCATION=  "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/controlnet-canny-sdxl-1.0"

## SDXS 
SDXS_MODEL_LOCATION           = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/sdxs-512-dreamshaper/"
CONTROLNET_SKETCH_LOCATION    = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/sdxs-512-dreamshaper-sketch/"

## Hyper-SD
HYPERSD_MODEL_LOCATION        = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/Hyper-SD"
HYPERSD_UNET_LOCATION         = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/Hyper-SD/Hyper-SDXL-1step-Unet.safetensors"
HYPERSD_LORA_LOCATION         = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/models/Hyper-SD/Hyper-SDXL-1step-lora.safetensors"

# SDXL 
SDXL_LCM_LORA_LOCATION        = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/lcm-lora-sdxl"
SDXL_PAPERCUT_LOCATION        = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/Papercut_SDXL"
SDXL_MIDJOURNEYV16_LOCATION   = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/Midjourney-V6.1"
SDXL_AAM_XL_ANIMEMIX_LOCATION = "/Volumes/980ProGyrus/Projects/BuildStuff2024_RTSD/AAM_XL_AnimeMix"

#### IMPORTANT WINDOWS LOCATIONS [MODIFY THESE]
########

# ## SD 1.5 Models
# LCM_Dv7_MODEL_LOCATION        = 'D:\\rmasiso\\PROJECTS\\AI\\models\\LCM_Dreamshaper_v7' #
# LCM_Dv8_MODEL_LOCATION        = "F:\Projects\BuildStuff2024_RTSD\models\dreamshaper-8"

# CONTROLNET_CANNY_LOCATION     = "D:\\rmasiso\\PROJECTS\\AI\\models\\control_v11p_sd15_canny"

# ## SDXL Models
# SDXLTURBO_MODEL_LOCATION      = 'D:\rmasiso\PROJECTS\AI\models\sdxl-turbo'
# SDXLL_MODEL_LOCATION          = "F:\Projects\BuildStuff2024_RTSD\models\SDXL-Lightning"
# SDXL_BASEMODEL_LOCATION       = "F:\Projects\BuildStuff2024_RTSD\models\stable-diffusion-xl-base-1.0"
# SDXLL_CKPT_LOCATION           = "F:\Projects\BuildStuff2024_RTSD\models\SDXL-Lightning\sdxl_lightning_2step_unet.safetensors" # Use the correct ckpt for your step setting!
# SDXL_CANNY_CONTROLNET_LOCATION= "F:\Projects\BuildStuff2024_RTSD\models\controlnet-canny-sdxl-1.0"

# ## SDXS 
# SDXS_MODEL_LOCATION           = "F:\Projects\BuildStuff2024_RTSD\models\sdxs-512-dreamshaper"
# CONTROLNET_SKETCH_LOCATION    = "F:\Projects\BuildStuff2024_RTSD\models\sdxs-512-dreamshaper-sketch"


# HYPERSD_MODEL_LOCATION        = "F:\Projects\BuildStuff2024_RTSD\models\Hyper-SD"
# HYPERSD_UNET_LOCATION         = "F:\Projects\BuildStuff2024_RTSD\models\Hyper-SD\Hyper-SDXL-1step-Unet.safetensors"
# HYPERSD_LORA_LOCATION         = "F:\Projects\BuildStuff2024_RTSD\models\Hyper-SD\Hyper-SDXL-1step-lora.safetensors"

# SDXL_LCM_LORA_LOCATION        = "F:\Projects\BuildStuff2024_RTSD\models\lcm-lora-sdxl"
# SDXL_PAPERCUT_LOCATION        = "F:\Projects\BuildStuff2024_RTSD\models\Papercut_SDXL"
# SDXL_MIDJOURNEYV16_LOCATION   = "F:\Projects\BuildStuff2024_RTSD\models\Midjourney-V6.1"
# SDXL_AAM_XL_ANIMEMIX_LOCATION = "F:\Projects\BuildStuff2024_RTSD\models\AAM_XL_AnimeMix"


AVAILABLE_MODELS = [
                    "canny",
                    "Dreamshaper_v7_LCM_Canny",
                    "Dreamshaper_v7_LCM_img2img", 
                    "Dreamshaper_v8_LCM_Canny",
                    "Dreamshaper_v8_LCM_img2img",
                    "SDXL_Turbo",
                    "SDXL_Lightning",
                    "Hyper_SD",
                    "SDXS",
                    "Papercut_LCM-LoRA",
                    "MidjourneyV1.6_LCM-LoRA",
                    "AAM_XL_AnimeMix_LCM-LoRA",
                    ]

#### OTHER CONSTANTS
###########

# Global variable to store the processed image
RAW_IMAGE              = None
PROCESSED_IMAGE        = None
SELECTED_IMAGES        = []

SD_MODEL               = None
LOADED_SD_MODEL        = None #cache of most recent loaded model so as not to reload if we go and change canny settings
PIPELINE               = None

JSON_FILE_PATH         = 'parameters.json' #this is where the json files with model parameter info are located
SAVE_FOLDER            = "output" # folder where output images are saved
DPI                    = 300 # dots per inch / determined by device
IMAGE_PREFIX           = "image"
IMAGE_GALLERY_MAIN     = "image_gallery"
IMAGE_GALLERY_LOCATION = "image_gallery/20241019_Session_000"
OUTPUT_PREFIX          = "output"
IMAGE_EXTENSIONS       = ['.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff', '.webp']
IMAGE_GALLERY_HEIGHT   = 100 # resolution of images inside gallery

DYNAMIC_PARAMETERS     = ["RANDOM_SEED", "PROMPT", "WIDTH",'HEIGHT',"SD_MARKDOWN",'GUIDANCE_SCALE','INFERENCE_STEPS','NOISE_STRENGTH',
                            'CONDITIONING_SCALE','GUIDANCE_START',"GUIDANCE_END", "CANNY_MARKDOWN", "CANNY_LOWER", "CANNY_UPPER", "CANNY_APERTURE", 
                            "COLOR_INVERT", "ETA", "ZOOM"]


DEFAULT_PROMPT         = "portrait of a minion, wearing goggles, yellow skin, wearing a beanie, despicable me movie, in the style of pixar movie" #van gogh in the style of van gogh"


mm2inch         = 25.4 # mm/inch
inch2mm         = (1/25.4) # inch/mm


## initial parameters for gradio
PARAMETER_STATE        = {
                            "PROMPT": DEFAULT_PROMPT,
                            "GUIDANCE_SCALE": 3,
                            "INFERENCE_STEPS": 2,
                            "DEFAULT_NOISE_STRENGTH": 0.7,
                            "CONDITIONING_SCALE": 0.8,
                            "GUIDANCE_START": 0.0,
                            "GUIDANCE_END": 1.0,
                            "RANDOM_SEED": 21,
                            "HEIGHT": 512,
                            "WIDTH": 512,
                            "NOISE_STRENGTH" : 0.7,
                            "CANNY_LOWER": 50,
                            "CANNY_HIGHER": 150,
                            "CANNY_APERTURE": 3,
                            "COLOR_INVERT": False,
                            "ETA" : 1.0,
                            "ZOOM" : 1.0,
                          }


############
############ NEURAL NETWORK ENGINE STUFF
############

def choose_device(torch_device = None):
    """
    Determines what engine to use for inference based on current computer.
    """

    print('...Is CUDA available in your computer?',\
          '\n... Yes!' if torch.cuda.is_available() else "\n... No D': ")
    print('...Is MPS available in your computer?',\
          '\n... Yes' if torch.backends.mps.is_available() else "\n... No D':")

    if torch_device is None:
        if torch.cuda.is_available():
            torch_device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available() and not torch.cuda.is_available():
            torch_device = "mps"
            torch_dtype = torch.float16
        else:
            torch_device = "cpu"
            torch_dtype = torch.float32

    print("......using ", torch_device)

    return torch_device, torch_dtype

TORCH_DEVICE, TORCH_DTYPE     = choose_device()  

def flush():
    """
    Help deal with potential memory issues when switching models.

    Run with: "flush()"
    
    """
    gc.collect()
    if torch.cuda.is_available():
        # Clear CUDA cache
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # Clear MPS cache
        torch.mps.empty_cache()
    else:
        print("...Couldn't flush torch cache; neither CUDA nor MPS is available on this system.")


def delete_objects(list_of_objects):
    """
    Delete objects in a list. May help clear memory.
    """

    for obj in list_of_objects:
        del obj

    return print("... deleted objects.")



############
############ MISC FUNCTIONS
############

def convert_BGR2RGB(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(image)

def extract_and_convert_list_to_int(text):
    cleaned_text = text.strip('[]')
    int_list = [int(item.strip()) for item in cleaned_text.split(',')]
    return int_list

def mm2inch(source_system="metric", value=100):
    if source_system=="metric":
        # mm to inches
        converted_val = value / 25.4
    else: 
        # inches to mm
        converted_val = value * 25.4

    return converted_val

############
############ IMAGE GALLERY STUFF
############

def get_image_files(prefix, folder_path):
    files = np.sort(os.listdir(folder_path))
    # Filter files that start with the prefix and end with any of the image extensions
    image_files = [file for file in files if file.startswith(prefix) and any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)]
    return image_files

def refresh_gallery():
    """
    compile images from folder
    """

    folder_path = IMAGE_GALLERY_LOCATION

    images = []
    for filename in get_image_files(IMAGE_PREFIX, folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((int(img.width * (IMAGE_GALLERY_HEIGHT / img.height)), IMAGE_GALLERY_HEIGHT), Image.Resampling.NEAREST)
        images.append(img)

    return images

def StartNewSessionFolder(base_path=IMAGE_GALLERY_MAIN):
    """
    Start new session folder. Creates a new folder for new photos to be taken. 
    """

    
    folder_pattern = re.compile(r'(\d{8})_Session_(\d{3})') # Regular expression to match the folder format "YearMonthDate_Session_001" --simple format
    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))] # List all subfolders in the base path
    valid_folders = [] # Filter and sort subfolders based on the date and session number
    for folder in subfolders:
        match = folder_pattern.match(folder)
        if match:
            valid_folders.append(folder)
    
    sorted_folders = sorted(valid_folders, key=lambda x: (x[:8], x[-3:]), reverse=True)
    if sorted_folders:
        latest_folder = sorted_folders[0]
        latest_date_str, latest_session_str = folder_pattern.match(latest_folder).groups()
        latest_session = int(latest_session_str)
        
        today_str = datetime.now().strftime('%Y%m%d')  # Check if the latest folder is from today
        if latest_date_str == today_str:
            new_session = latest_session + 1
        else:
            new_session = 0
    else:
        today_str = datetime.now().strftime('%Y%m%d')
        new_session = 0
    
    
    new_folder_name = f"{today_str}_Session_{new_session:03d}" # Create the new folder name
    new_folder_path = os.path.join(base_path, new_folder_name)
    
   
    os.makedirs(new_folder_path)  # Create the new folder
    
    print(new_folder_path)

    global IMAGE_GALLERY_LOCATION 
    IMAGE_GALLERY_LOCATION = new_folder_path # set global variable of image gallery location to new.

    return refresh_gallery()


def NavigateGalleryFolders(previous_or_next):
    """
    receives text that is "Previous" or "Next" to navigate photo booth sessions 
    """

    global IMAGE_GALLERY_LOCATION  # need to declare global cus i modify its path inside

    base_path = IMAGE_GALLERY_MAIN

    subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    sorted_subfolders = np.sort(subfolders)

    current_session = IMAGE_GALLERY_LOCATION.split('/')[-1] #name of folder is after last slash

    print("CURRENT STUFF: ", current_session, sorted_subfolders, np.where(sorted_subfolders==current_session))
    index_of_current_session = np.where(sorted_subfolders==current_session)[0][0]

    if previous_or_next == 'Previous':
        if index_of_current_session>0:
            IMAGE_GALLERY_LOCATION = os.path.join(IMAGE_GALLERY_MAIN,sorted_subfolders[index_of_current_session-1])
    elif previous_or_next == "Next":
        if (index_of_current_session>=0) and (index_of_current_session<len(sorted_subfolders)):
            IMAGE_GALLERY_LOCATION = os.path.join(IMAGE_GALLERY_MAIN,sorted_subfolders[index_of_current_session+1])

    return refresh_gallery()


def save_image(image, img_path):
    if isinstance(image,np.ndarray):
        Image.fromarray(image).save(img_path)
    else: # save if not an np.ndarray
        image.save(img_path)

def capture_and_save_images():
    """
    Capture and save 4 images to the gallery directory.
    """
    # global RAW_IMAGE
    # global PROCESSED_IMAGE
    # global SAVED_IMAGES_INDEX

    if PROCESSED_IMAGE is not None:
        image_files = get_image_files(IMAGE_PREFIX, IMAGE_GALLERY_LOCATION)

        # SAVED_IMAGES_INDEX = len(image_files) + 1

        raw_img_path = os.path.join(IMAGE_GALLERY_LOCATION, f"{IMAGE_PREFIX}_{len(image_files)+1:03d}.png")
        processed_img_path = os.path.join(IMAGE_GALLERY_LOCATION, f"{IMAGE_PREFIX}_{len(image_files)+2:03d}.png")

        save_image(RAW_IMAGE, raw_img_path)
        save_image(PROCESSED_IMAGE, processed_img_path)

    return refresh_gallery()

############
############ CREATE PHOTOBOOTH STRIP STUFF
############

def resize_image_to_height(img, target_height):
    """
    Resize the image to the specified height while maintaining the aspect ratio.
    
    Parameters:
    img (PIL.Image.Image): The image to resize.
    target_height (int): The desired height in pixels.
    
    Returns:
    PIL.Image.Image: The resized image.
    """
    # Calculate the new width to maintain the aspect ratio (simple calc)
    new_width = int(img.width * (target_height / img.height))
    
    # Resize the image using LANCZOS resampling (it's the best one for quality at sacrifice of a little bit of speed)
    resized_img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    return resized_img

def create_photo_strip(nStrips, selected_images_idx_list, paper_width_mm, paper_height_mm, dpi=DPI, save_folder=SAVE_FOLDER):
    
    # dpi                     = 300 # dots per inch (different for different devices) -- this is the DPI for the cannon printer
    # nStrips         = 4 # how many photobooth strips
    # nPhotosPerStrip = 3 # how many photos stacked vertically per strip 
    selected_images_idx_list = extract_and_convert_list_to_int(selected_images_idx_list)
    nPhotosPerStrip          = len(selected_images_idx_list) #should be 3

    # paper_width_mm  = 148
    # paper_height_mm = 100

    ## ESTABLISH SIZING INFORMATION

    paper_width_px      = int(paper_width_mm * dpi * inch2mm) # (mm * dots / inch) * (inch / mm) = dots = pixels
    paper_height_px     = int(paper_height_mm * dpi * inch2mm)  # (mm * dots / inch) * (inch / mm) = dots = pixels
    square_image_width  = int(paper_width_px / nStrips) # tells me how wide the pixels I have for each imageb
    square_image_height = int(paper_height_px / nPhotosPerStrip) # tells me how tall the images need to be to fit
    square_image_size   = np.min([square_image_width, square_image_height]) # the minimum is usually gonna be based on how many can be fit vertically (when trying to maximize full sets/vertical strips to print)
    scissor_padding     = ((paper_width_px - (square_image_size*nStrips)) / (nStrips - 1)) # how many square images fit with max image size to fit 3 vertically, then identify how the spacing available for cutting with scissors (nStrips-1)

    ## CREATE NEW IMAGE:

    # 1) resize images
    resized_images = []
    folder_path = IMAGE_GALLERY_LOCATION
    img_filenames = get_image_files(IMAGE_PREFIX, folder_path)
    for fi in selected_images_idx_list:
        img = Image.open(os.path.join(folder_path, img_filenames[fi]))
        img = resize_image_to_height(img, square_image_size)
        resized_images.append(img)

    # 2) stack the images into strip

    # single vertical strip
    strip_width    = square_image_size
    strip_height   = square_image_size * nPhotosPerStrip #(width, height) --> where width==height --> (width, width*3) for 3 photos per strip
    vertical_strip = Image.new('RGB', (strip_width, strip_height))

    # Paste the three images into the vertical strip
    for i,img in enumerate(resized_images):  
        vertical_strip.paste(img, (0, (square_image_size)*i)) #. paste in upper left corner, then paste again at height, then height*2, etc

    # 3) Paste the strip repeatedly 
    # Create a blank image for the paper
    paper = Image.new('RGB', (paper_width_px, paper_height_px), 'white')


    # Paste the vertical strips onto the paper
    for i in range(nStrips):
        padding = scissor_padding if i > 0 else 0
        # print((393 + add)*i)
        paper.paste(vertical_strip, (int(strip_width+padding)*i, 0))
    # paper.paste(vertical_strip, (i * strip_width, 0))

    # Save the final image to a file
    fnum = len(get_image_files(OUTPUT_PREFIX, save_folder)) + 1
    paper.save(os.path.join(SAVE_FOLDER,f'output_{fnum:03}.png'))
    return paper

############
############ PARAMETER UPDATE STUFF
############

def select_images_for_print(selection: gr.SelectData):
    global SELECTED_IMAGES
    SELECTED_IMAGES.append(selection.index)
    return SELECTED_IMAGES

def deselect_images_for_print():
    global SELECTED_IMAGES
    SELECTED_IMAGES = []
    return SELECTED_IMAGES

def update_gallery_markdown():
   global IMAGE_GALLERY_LOCATION

   markdown = f"<center><h3>{IMAGE_GALLERY_LOCATION.split('/')[-1]}</h3></center>"
   return markdown

def load_parameters(file_path, model_name):
    """
    Loads parameters.json file and grabs paremeters based on selected model.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        params = data[model_name] #index into model name info
    return params

def convert_parameters_to_gradio_list(parameter_state, dynamic_parameter_names):
    """
    Creates gradio objects to update parameters in gradio app.py. Follows the parameters.json formatting.
    """
    param_list = []
    for para in dynamic_parameter_names:
            if "markdown" not in para.lower():
                param_list.append(gr.update(value=parameter_state[para][0], 
                                                 visible=parameter_state[para][1], 
                                                 interactive=parameter_state[para][2]))
            else:
                param_list.append(gr.update(value=parameter_state[para][0], 
                                                 visible=parameter_state[para][1], 
                                                 ))

    return param_list

def change_parameters(model):

    PARAMETER_CONFIG       = load_parameters(JSON_FILE_PATH, model)
    NEW_PARAMETER_LIST     = convert_parameters_to_gradio_list(PARAMETER_CONFIG, DYNAMIC_PARAMETERS)

    print("CHANGE PARAMETERS FUNCTION: ", DYNAMIC_PARAMETERS)

    return NEW_PARAMETER_LIST


############
############ STABLE DIFFUSION MODEL STUFF
############

def prepare_seed(RANDOM_SEED):
    generator = torch.manual_seed(RANDOM_SEED)
    return generator


def process_canny(image, lower_threshold = 100, upper_threshold = 100, aperture=3): 
    image = cv.Canny(image, lower_threshold, upper_threshold,apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2) # convert to 3 channel image (for color)
    return image

def process_sdxlturbo(image):
    return image

def resize_frame(frame, zoom_out):
    """
    Resize the frame to half its original size.
    """
    h, w = frame.shape[:2]
    new_h, new_w = h // zoom_out, w // zoom_out #zoom_out tells us how much we want to zoom-out of the captured frame from camera
    #print(h,w, new_h, new_w)
    resized_frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized_frame

def get_result_and_mask(frame, center_x, center_y, width, height):
    """
    Gets the full frame and the mask for cutout
    """
    
    # Ensure the crop dimensions do not exceed the image boundaries
    top_left_y = max(0, center_y - height // 2)
    top_left_x = max(0, center_x - width // 2)
    bottom_right_y = min(frame.shape[0], center_y + height // 2)
    bottom_right_x = min(frame.shape[1], center_x + width // 2)

    mask = np.zeros_like(frame)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :] = 255
    cutout = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]

    return frame, cutout


# def prepare_lcm_pipeline(canny_location, lcm_location, i2i_type = "canny"):
    
#     if i2i_type=="canny":
#         controlnet = ControlNetModel.from_pretrained(canny_location, torch_dtype=TORCH_DTYPE,
#                                                 use_safetensors=True)
        
        

#         pipeline = StableDiffusionControlNetPipeline.from_pretrained(lcm_location,\
#                                                             controlnet=controlnet, 
#                                                             # unet=unet,\
#                                                             torch_dtype=TORCH_DTYPE, safety_checker=None).\
#                                                         to(TORCH_DEVICE)
#     elif i2i_type=="img2img":
#         pipeline = AutoPipelineForImage2Image.from_pretrained(
#                     lcm_location, torch_dtype=TORCH_DTYPE,
#                     safety_checker=None).to(TORCH_DEVICE)
        
#     return pipeline

    
def prepare_pipeline(model):
    global SD_MODEL 
    global PIPELINE
    # global PARAMETER_STATE
    # global PARAMETER_STATE_2

    if model=="Dreamshaper_v7_LCM_Canny":

        # pipeline = prepare_lcm_pipeline(CONTROLNET_CANNY_LOCATION, LCM_Dv7_MODEL_LOCATION, "canny")

        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE, use_safetensors=True)
        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(LCM_Dv7_MODEL_LOCATION, controlnet=controlnet, 
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).to(TORCH_DEVICE)
        
    elif model=="Dreamshaper_v7_LCM_img2img":

        # pipeline = prepare_lcm_pipeline(CONTROLNET_CANNY_LOCATION, LCM_Dv7_MODEL_LOCATION, "img2img")
        
        pipeline = AutoPipelineForImage2Image.from_pretrained(
                    LCM_Dv7_MODEL_LOCATION, torch_dtype=TORCH_DTYPE,
                    safety_checker=None).to(TORCH_DEVICE)
        

    elif model=="Dreamshaper_v8_LCM_Canny":

        # pipeline = prepare_lcm_pipeline(CONTROLNET_CANNY_LOCATION, LCM_Dv7_MODEL_LOCATION, "canny")

        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE,use_safetensors=True)
        
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(LCM_Dv8_MODEL_LOCATION, controlnet=controlnet, 
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).to(TORCH_DEVICE)
        
    elif model=="Dreamshaper_v8_LCM_img2img":

        # pipeline = prepare_lcm_pipeline(CONTROLNET_CANNY_LOCATION, LCM_Dv7_MODEL_LOCATION, "img2img")
        
        pipeline = AutoPipelineForImage2Image.from_pretrained(
                    LCM_Dv8_MODEL_LOCATION, torch_dtype=TORCH_DTYPE,
                    safety_checker=None).to(TORCH_DEVICE)
        
    elif model == "SDXL_Turbo":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
                    SDXLTURBO_MODEL_LOCATION, torch_dtype=TORCH_DTYPE,
                    safety_checker=None).to(TORCH_DEVICE)
        
    elif model == "SDXL_Lightning":

        controlnet = ControlNetModel.from_pretrained(SDXL_CANNY_CONTROLNET_LOCATION, torch_dtype=TORCH_DTYPE)
    
        unet       = UNet2DConditionModel.from_config(SDXL_BASEMODEL_LOCATION, subfolder="unet").to(TORCH_DEVICE, TORCH_DTYPE)
        unet.load_state_dict(load_file(SDXLL_CKPT_LOCATION, device=TORCH_DEVICE))

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(SDXL_BASEMODEL_LOCATION,\
                                                        controlnet=controlnet, 
                                                        unet=unet,\
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).\
                                                    to(TORCH_DEVICE)
        
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

    elif model=="Hyper_SD":
        controlnet = ControlNetModel.from_pretrained(SDXL_CANNY_CONTROLNET_LOCATION, torch_dtype=TORCH_DTYPE
                                                )
        
        ###### UNCOMMENT FOR UNET not lora
        # unet       = UNet2DConditionModel.from_config(SDXLL_BASEMODEL_LOCATION, subfolder="unet").to(TORCH_DEVICE, TORCH_DTYPE)
        # unet.load_state_dict(load_file(HYPERSD_UNET_LOCATION, device=TORCH_DEVICE))

    
        # pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(SDXLL_BASEMODEL_LOCATION,\
        #                                                 controlnet=controlnet, 
        #                                                 unet=unet,\
        #                                                 torch_dtype=TORCH_DTYPE, safety_checker=None).\
        #                                             to(TORCH_DEVICE)
        
        # pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        # FOR LORA
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(SDXL_BASEMODEL_LOCATION,\
                                                        controlnet=controlnet, 
                                                        # unet=unet,\
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).\
                                                    to(TORCH_DEVICE)
        state_dict = load_file(HYPERSD_LORA_LOCATION, device=TORCH_DEVICE)
        pipeline.load_lora_weights(state_dict)
        # pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipeline.fuse_lora()
        # Use TCD scheduler to achieve better image quality
        pipeline.scheduler = TCDScheduler.from_config(pipeline.scheduler.config)
        
        
    elif model=="SDXS":
        
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_SKETCH_LOCATION, torch_dtype=TORCH_DTYPE ) 
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(SDXS_MODEL_LOCATION, controlnet=controlnet, 
                                                                 torch_dtype=TORCH_DTYPE, safety_checker=None).to(TORCH_DEVICE)
        
        
    elif "LCM-LoRA" in model:
        
        if model=="Papercut_LCM-LoRA":
            lora_location = SDXL_PAPERCUT_LOCATION
            adapter_name = "papercut"
            base_model_to_use = SDXL_BASEMODEL_LOCATION
            use_adapter = True
        elif model=="MidjourneyV1.6_LCM-LoRA":
            lora_location = SDXL_MIDJOURNEYV16_LOCATION
            adapter_name = "aidmaMidjourneyV61-v01"
            base_model_to_use = SDXL_BASEMODEL_LOCATION
            use_adapter = True
        elif model == "AAM_XL_AnimeMix_LCM-LoRA":
            base_model_to_use = SDXL_AAM_XL_ANIMEMIX_LOCATION
            use_adapter=False
                
        controlnet = ControlNetModel.from_pretrained(SDXL_CANNY_CONTROLNET_LOCATION, torch_dtype=TORCH_DTYPE)
        

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(base_model_to_use,\
                                                        controlnet=controlnet, 
                                                        # unet=unet,\
                                                        torch_dtype=TORCH_DTYPE, safety_checker=None).\
                                                    to(TORCH_DEVICE)
                                                    
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)# Combine LoRAs

        pipeline.load_lora_weights(SDXL_LCM_LORA_LOCATION, adapter_name="lcm")
        
        if use_adapter: 
            pipeline.load_lora_weights(lora_location, adapter_name=adapter_name)
            pipeline.set_adapters(["lcm", adapter_name], adapter_weights=[1.0, 1.0])# Combine LoRAs
            pipeline.fuse_lora(adapter_names=["lcm", adapter_name], lora_scale=1.0)# fuse LoRAs and unload weights
            # pipeline.unload_lora_weights()
        else:
            print("...not using extra adapter.")
    
    else:
        pipeline=None
        
    ## update global variables
    SD_MODEL = model
    PIPELINE = pipeline

    return gr.update(value=model, visible=False)


def run_lcm_canny(pipeline, ref_image, generator):

    if pipeline is None:
        raise ValueError("Pipeline is not initialized.")

    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image, 
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"], 
                         controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
                         control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
                         control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
                        ).images[0]
    

    return gen_image

def run_lcm_img2img(pipeline, ref_image, generator):
    # global PARAMETER_STATE

    if pipeline is None:
        raise ValueError("Pipeline is not initialized. Using non-SD model.")
        # gen_image = ref_image 
        
    # generator = prepare_seed()

    # gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
    #                      num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
    #                      guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
    #                      width                         = PARAMETER_STATE["WIDTH"], 
    #                      height                        = PARAMETER_STATE["HEIGHT"], 
    #                      generator                     = generator,
    #                      image                         = ref_image, 
    #                      controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
    #                      control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
    #                      control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
    #                     ).images[0]
    else:
        gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image, 
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"],
                        ).images[0]

    return gen_image

def run_sdxlturbo(pipeline,ref_image,generator):

    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image, 
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"],
                        ).images[0]
    
                        
    return gen_image

def run_sdxl_lightning(pipeline, ref_image, generator):

    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image,
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"], 
                         controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
                         control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
                         control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
                        ).images[0]

    return gen_image


def run_hyper_sd(pipeline, ref_image, generator):

    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image,
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"], 
                         controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
                         control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
                         control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
                         eta                           = PARAMETER_STATE["ETA"],
                        ).images[0]

    return gen_image

def run_sdxs(pipeline, ref_image, generator):
    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image, 
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"], 
                         controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
                         control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
                         control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
                        ).images[0]
    
    return gen_image

def run_lcm_lora(pipeline, ref_image, generator):

    gen_image = pipeline(prompt                        = PARAMETER_STATE['PROMPT'],
                         num_inference_steps           = PARAMETER_STATE['INFERENCE_STEPS'], 
                         guidance_scale                = PARAMETER_STATE["GUIDANCE_SCALE"],
                         width                         = PARAMETER_STATE["WIDTH"], 
                         height                        = PARAMETER_STATE["HEIGHT"], 
                         generator                     = generator,
                         image                         = ref_image,
                         strength                      = PARAMETER_STATE["NOISE_STRENGTH"], 
                         controlnet_conditioning_scale = PARAMETER_STATE["CONDITIONING_SCALE"], 
                         control_guidance_start        = PARAMETER_STATE["GUIDANCE_START"], 
                         control_guidance_end          = PARAMETER_STATE["GUIDANCE_END"], 
                        ).images[0]
    
    return gen_image

def process_image(model_selected, input_img, canny_lower_threshold,canny_upper_threshold, canny_aperture, color_invert, 
                  seed, prompt, width, height, guidance_scale, inference_steps, noise_strength, conditioning_scale,
                    guidance_start, guidance_end, eta, zoom_out_amount):
    """
    
    MASTER PROCESS FUNCTION
    
    """
    global RAW_IMAGE 
    global PROCESSED_IMAGE 
    global PARAMETER_STATE


    PARAMETER_STATE['WIDTH'] = width
    PARAMETER_STATE['HEIGHT'] = height
    PARAMETER_STATE['PROMPT'] = prompt
    PARAMETER_STATE['GUIDANCE_SCALE'] = guidance_scale
    PARAMETER_STATE['INFERENCE_STEPS'] = inference_steps
    PARAMETER_STATE['NOISE_STRENGTH'] = float(noise_strength)
    PARAMETER_STATE['CONDITIONING_SCALE'] = float(conditioning_scale)
    PARAMETER_STATE['GUIDANCE_START'] = float(guidance_start)
    PARAMETER_STATE["GUIDANCE_END"] = float(guidance_end)
    PARAMETER_STATE["ETA"] = float(eta)
    PARAMETER_STATE["ZOOM"] = float(zoom_out_amount)

    generator = prepare_seed(seed)

    input_img = resize_frame(input_img, zoom_out_amount)
    center_x = (input_img.shape[1]) // 2
    center_y = (input_img.shape[0]) // 2    

    full_size_image, cropped_image = get_result_and_mask(input_img, center_x, center_y, PARAMETER_STATE['WIDTH'], PARAMETER_STATE['HEIGHT'])

    # This function should process the image and return the output image
    if model_selected == 'canny':
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
    
    elif model_selected == "Dreamshaper_v7_LCM_Canny":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        # image = util.convert_numpy_image_to_pil_image(util.convert_BGR2RGB(image))
        # image = util.convert_numpy_image_to_pil_image(cv.cvtColor(image, cv.COLOR_RGB2BGR))
        image = convert_numpy_image_to_pil_image(image)
        image = run_lcm_canny(PIPELINE, image, generator)

    elif model_selected == "Dreamshaper_v7_LCM_img2img":
        image = convert_numpy_image_to_pil_image(cropped_image)
        image = run_lcm_img2img(PIPELINE, image, generator)

    elif model_selected == "Dreamshaper_v8_LCM_Canny":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        # image = util.convert_numpy_image_to_pil_image(util.convert_BGR2RGB(image))
        # image = util.convert_numpy_image_to_pil_image(cv.cvtColor(image, cv.COLOR_RGB2BGR))
        image = convert_numpy_image_to_pil_image(image)
        image = run_lcm_canny(PIPELINE, image, generator)

    elif model_selected == "Dreamshaper_v8_LCM_img2img":
        image = convert_numpy_image_to_pil_image(cropped_image)
        image = run_lcm_img2img(PIPELINE, image, generator)

    elif model_selected == "SDXL_Turbo":
        # image = process_sdxlturbo(cropped_image)
        image = convert_numpy_image_to_pil_image(cropped_image)
        image = run_sdxlturbo(PIPELINE, image, generator)

    elif model_selected == "SDXL_Lightning":
        # image = process_sdxlturbo(cropped_image)
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = convert_numpy_image_to_pil_image(image)
        image = run_sdxl_lightning(PIPELINE, image, generator)
        
    elif model_selected == "Hyper_SD":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = convert_numpy_image_to_pil_image(image)
        image = run_hyper_sd(PIPELINE, image, generator)
        
    elif model_selected == "SDXS":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = cv.bitwise_not(image) # sdxs sketch -- so needs to look like sketch!
        image = convert_numpy_image_to_pil_image(image)
        image = run_sdxs(PIPELINE, image, generator)
        
    elif model_selected=="Papercut_LCM-LoRA":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = convert_numpy_image_to_pil_image(image)
        image = run_lcm_lora(PIPELINE, image, generator)

    elif model_selected=="MidjourneyV1.6_LCM-LoRA":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = convert_numpy_image_to_pil_image(image)
        image = run_lcm_lora(PIPELINE, image, generator)

    elif model_selected=="AAM_XL_AnimeMix_LCM-LoRA":
        image = process_canny(cropped_image, canny_lower_threshold, canny_upper_threshold, canny_aperture)
        image = convert_numpy_image_to_pil_image(image)
        image = run_lcm_lora(PIPELINE, image, generator) 
    else:
        image = input_img

    # determine to invert colors
    if color_invert:
        image = cv.bitwise_not(np.array(image))

    ### FOR SDXS -->  cropped_image = cv.bitwise_not(cropped_image) if color_invert else cropped_image #send different canny image

    ### ADD THING BACK
    # print(full_size_image.shape)
    # print(cropped_image.shape)
    # print(image.shape)
    # print(type(image))
    # with_mask_image = full_size_image.copy()

     # Overlay the cropped image on the original image
    # output_img = overlay_image(full_size_image, cropped_image, (center_y - DEFAULT_HEIGHT // 2, center_x - DEFAULT_WIDTH // 2))
    # with_mask_image[center_y:center_y+DEFAULT_HEIGHT, center_x:center_x+DEFAULT_WIDTH,:] = image #cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    RAW_IMAGE = cropped_image
    PROCESSED_IMAGE = image

    return image


############
############ PRINTER STUFF
############

def show_info_message(text):
    gr.Info(f"{text}")


def list_printers():
    system = platform.system()
    
    if system == "Windows":
        try:
            import win32print
        except ModuleNotFoundError:
            print("win32print module not found. Installing pywin32... (you may need to restart app.py)")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
            import win32print
            
        printers = [printer[2] for printer in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)]
        return printers
        
    elif system == "Linux" or system == "Darwin":  # Darwin is macOS
        try:
            import cups
            
        except ModuleNotFoundError:
            print("CUPS module not found. Installing pycups...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycups"])
            import cups        
        
        conn = cups.Connection()
        printers = conn.getPrinters()
        return list(printers.keys())
        
    else:
        gr.Warning(f"Listing printers not supported on {system} systems.")
        return []

def print_file(selected_printer):
    print(f"Using printer: {selected_printer}")

    fnum = len(get_image_files(OUTPUT_PREFIX, SAVE_FOLDER)) # get most recent image number
    file_path = os.path.join(SAVE_FOLDER,f'output_{fnum:03}.png') # get path to most recent image
    
    system = platform.system()
    
    if system == "Windows":
        try:
            import win32api
        except ModuleNotFoundError:
            print("win32print or win32api module not found. Installing pywin32...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])
            import win32api
        
        win32api.ShellExecute(0, "print", file_path, f'/d:"{selected_printer}"', ".", 0)
        
    elif system == "Linux" or system == "Darwin":
        try:
            import cups
        except ModuleNotFoundError:
            print("CUPS module not found. Installing pycups...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycups"])
            import cups
        
        conn = cups.Connection()
        conn.printFile(selected_printer, file_path, "Python_Print_Job", {})
        show_info_message("Printing in progress.")
        
    else:
        print(f"Printing not supported on {system} systems.")
        gr.Warning(f"Printing not supported on {system} systems.")

