###
### PRE-ORDER BOOK on DIY AI! https://a.co/d/eDxJXJ0
###

import cv2 as cv
import numpy as np
from PIL import Image
from diffusers import (StableDiffusionControlNetPipeline,
                       ControlNetModel)
import torch

def choose_device(torch_device = None):
    print("...Is CUDA available in your computer?",\
          "\n... Yes!" if torch.cuda.is_available() else "\n... No D': ")
    print("...Is MPS available in your computer?",\
          "\n... Yes" if torch.backends.mps.is_available() else "\n... No D':")

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

#################
################# CONSTANTS
#################

DEFAULT_PROMPT                = "van gogh in the style of van gogh"

MODEL = "lcm"
LCM_MODEL_LOCATION = '/Users/rolando/Documents/PROJECTS/YouTube/DIYAI_WebcamDiffusion/tutorial_scripts/models/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION = "/Users/rolando/Documents/PROJECTS/YouTube/DIYAI_WebcamDiffusion/tutorial_scripts/models/control_v11p_sd15_canny" 
TORCH_DEVICE, TORCH_DTYPE = choose_device()  
GUIDANCE_SCALE = 3 # 
INFERENCE_STEPS = 2 #4 for lcm (high quality) 
DEFAULT_NOISE_STRENGTH = 0.7 # 0.5 works well too
CONDITIONING_SCALE = .7 # .5 works well too
GUIDANCE_START = 0.
GUIDANCE_END = 1.
RANDOM_SEED = 21
HEIGHT = 512 #384 #512
WIDTH = 512 #384 #512

def prepare_seed():
    generator = torch.manual_seed(RANDOM_SEED)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"
    
    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]

    return frame, cutout

def process_edges(image, lower_threshold = 100, upper_threshold = 100, aperture=3): 
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold,apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def prepare_lcm_controlnet():
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_LOCATION, 
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True
    )

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        LCM_MODEL_LOCATION,
        controlnet=controlnet, 
        torch_dtype=TORCH_DTYPE, safety_checker=None
    ).to(TORCH_DEVICE)
    
    return pipeline

def run_lcm(pipeline, ref_image):
    generator = prepare_seed()
    gen_image = pipeline(
        prompt = DEFAULT_PROMPT,
        num_inference_steps = INFERENCE_STEPS, 
        guidance_scale = GUIDANCE_SCALE,
        width = WIDTH, 
        height = HEIGHT, 
        generator = generator,
        image = ref_image, # reference image processed to be edges!
        controlnet_conditioning_scale = CONDITIONING_SCALE, 
        control_guidance_start = GUIDANCE_START, 
        control_guidance_end = GUIDANCE_END, 
    ).images[0]

    return gen_image

def run_model():

    ###
    ### PREPARE MODEL
    ###

    pipeline  = prepare_lcm_controlnet()
    
    processor  = process_edges

    run_model  = run_lcm

    ###
    ### PREPARE WEBCAM 
    ###

    # Open a connection to the webcam
    cap = cv.VideoCapture(0)

    CAP_WIDTH  = cap.get(cv.CAP_PROP_FRAME_WIDTH)  #320
    CAP_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT) #240

    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH/2) 
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT/2)

    ###
    ### RUN WEBCAM AND DIFFUSION
    ###

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()

        # break if cap returns false
        if not ret:
            print("Error: Failed to capture frame.")
            break
    
        # Calculate the center position for the black and white filter
        center_x = (image.shape[1] - WIDTH) // 2
        center_y = (image.shape[0] - HEIGHT) // 2

        result_image, masked_image = get_result_and_mask(image, center_x, center_y, WIDTH, HEIGHT)

        numpy_image = processor(masked_image)
        pil_image   = convert_numpy_image_to_pil_image(numpy_image)
        pil_image   = run_model(pipeline, pil_image)

        result_image[center_y:center_y+HEIGHT, center_x:center_x+WIDTH] = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        # Display the resulting frame
        cv.imshow("output", result_image)

        # Break the loop when 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv.destroyAllWindows()

###
### RUN SCRIPT
###

run_model()
