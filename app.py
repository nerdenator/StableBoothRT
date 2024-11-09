# https://discuss.huggingface.co/t/how-to-programmatically-enable-or-disable-components/52350/3

import gradio as gr
import numpy as np
import util



###
### PRE-ORDER BOOK on DIY AI! https://a.co/d/eDxJXJ0
###



###
### GENERAL FUNCTIONS
###


def show_warning(selection: gr.SelectData):
    # gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")
    gr.Warning(f"Your choice is #{selection.index}!")


css = """ """

# css = """
#     #prompt_textbox {
#        /* background-color: #f97316;  /* Primary color */
#         color: white;
#         border: 5px solid #007bff;
#     }
#     """

with gr.Blocks(css=css) as demo:


    ## PARAMAETER STATE
    # state = gr.State(initial_state)

    ## WEBCAM
    #####################
    with gr.Row():

        # MODEL SELECTION
        with gr.Column(scale=4):

            with gr.Row():
                NewSessionButton = gr.Button("Start New Session.", scale=3, variant="primary")

                with gr.Column(scale=1):
                    previous_session_button = gr.Button("Previous")
                    next_session_button = gr.Button("Next")
            
            #### CURRENT LOCATION STUFF
            markdown_gallery_location = gr.Markdown(f"<center><h3>{util.IMAGE_GALLERY_LOCATION.split('/')[-1]}</h3></center>")

            with gr.Row():
                # gr.Markdown("### Model Selection\nSelect a model from the dropdown menu below.")
                model_selected = gr.Dropdown(choices=util.AVAILABLE_MODELS, value="canny", label="Select Model", scale=2)
            with gr.Row():
                 # Output to display the selected model
                selected_model_text = gr.Markdown(label="Selected Model", scale=2, visibile=False)
            with gr.Row():
                output_width     = gr.Radio([384,512,1024], value= util.PARAMETER_STATE['WIDTH'], label="Output Width", info="Some models do better at 512, others at 1024",interactive=True)
                output_height    = gr.Radio([384,512,1024], value= util.PARAMETER_STATE['HEIGHT'], label="Output Height", info="Some models do better at 512, others at 1024",interactive=True)

            

        with gr.Column(scale=6):
            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    canny_markdown = gr.Markdown("### Canny Edge Detection Settings")
                with gr.Row():
                    canny_lower_threshold = gr.Slider(0, 300, value=util.PARAMETER_STATE["CANNY_LOWER"], label="Lower Threshold")
                    canny_upper_threshold = gr.Slider(0, 600, value=util.PARAMETER_STATE["CANNY_HIGHER"], label="Upper Threshold")
                    canny_aperture        = gr.Slider(3, 7, step=2, value=util.PARAMETER_STATE["CANNY_APERTURE"], label="Aperture Size")
                    color_invert          = gr.Checkbox(value=util.PARAMETER_STATE["COLOR_INVERT"], label="Invert Colors")
                with gr.Row():

                    stable_markdown  = gr.Markdown("### Stable Diffusion Model Settings")
                with gr.Row():
                    seed             = gr.Slider(randomize=True, minimum=0, maximum=11023012000, label="Random Seed",step=1,interactive=True)
                    guidance_scale   = gr.Slider(minimum=0, maximum=50, value= util.PARAMETER_STATE['GUIDANCE_SCALE'], label="Guidance Scale",step=.1,interactive=True)            
                    inference_steps  = gr.Slider(minimum=0, maximum=20, value= util.PARAMETER_STATE['INFERENCE_STEPS'], label="Inference Steps",step=1,interactive=True)           
                    noise_strength   = gr.Slider(minimum=0, maximum=1.0, value= util.PARAMETER_STATE['NOISE_STRENGTH'], label="Noise Strength",step=0.1,interactive=True)
                    eta              = gr.Slider(minimum=0, maximum=10.0, value= util.PARAMETER_STATE['ETA'], label="eta",step=0.1,interactive=True)        
        
                with gr.Row():
                    cond_scale       = gr.Slider(minimum=0., maximum=3, value= util.PARAMETER_STATE['CONDITIONING_SCALE'], label="Conditioning Scale",step=0.1,interactive=True) 
                    guidance_start   = gr.Slider(minimum=0., maximum=1.0, value= util.PARAMETER_STATE['GUIDANCE_START'], label="Reference Guidance Start",step=0.1,interactive=True) 
                    guidance_end     = gr.Slider(minimum=0., maximum=1.0, value= util.PARAMETER_STATE['GUIDANCE_END'], label="Reference Guidance End",step=0.1,interactive=True) 

    with gr.Row():
        prompt = gr.Textbox(max_lines=1, label="Edit Prompt",interactive=True,elem_id="prompt_textbox")

        # enter_prompt = gr.Button("Run", scale=0, variant="primary")
        # output_prompt = gr.Textbox(label="Updated Prompt")

    with gr.Row(scale=12):    
        with gr.Row():
            input_img = gr.Image(label="Input", source="webcam", streaming=True, live=True, streaming_update_interval=0.01,scale=4)#shape=(400,720))  # Enable streaming and live updates
            print("INPUT: ", np.shape(input_img))
        # with gr.Colum(stretch=True, scale=10):
            output_img = gr.Image(label="Output",min_width=util.PARAMETER_STATE['WIDTH'],width=util.PARAMETER_STATE['WIDTH'], height=util.PARAMETER_STATE['HEIGHT'], scale=7)
            
        # with gr.Column(stretch=True):
            ## Capture and Save Button
            capture_button = gr.Button("Capture and Save Images", scale=2,variant="primary")
            

    ## IMAGE GALLERY /  ## REFRESH BUTTON
    #####################
    with gr.Row():
        image_gallery = gr.Gallery(label="Image Gallery",columns=8)
    with gr.Row():
        refresh_button = gr.Button("Refresh")
        selected_images_txt = gr.Text(util.SELECTED_IMAGES, label="Images for Print")
        reset_selection_btn = gr.Button("Reset Selection")

    # OUTPUT PHOTO_IMAGE
    with gr.Row():
        with gr.Column():
            nStrips = gr.Slider(minimum=3, maximum=6, value=4, label="Number of Strips",step=1,interactive=True)
            paper_width_mm = gr.Number(value = 148, label = "Paper Width (mm)")
            paper_height_mm = gr.Number(value = 100, label = "Paper Height (mm)")
            gen_photo_strip= gr.Button("Generate Strip",variant="primary")
        photo_strip_image = gr.Image(label="Image for print")
        with gr.Column(stretch=True):
            printer = gr.Dropdown(choices=util.list_printers(), label="Select Printer", scale=2)
            print_strip_btn = gr.Button("Print", label="Print", scale=6,variant="primary")

    ## WEBCAM FUNCTION CALL
    ###############
        
    inputs = [
        model_selected, 
        input_img,
        canny_lower_threshold,
        canny_upper_threshold,
        canny_aperture,
        color_invert,
        seed,
        prompt,
        output_width,
        output_height,
        guidance_scale, 
        inference_steps,               
        noise_strength,        
        cond_scale,      
        guidance_start,   
        guidance_end, 
        eta,
                ]
    outputs = [output_img]
    
    # MAIN FUNCTION STUFF
    input_img.stream(fn=util.process_image, inputs=inputs, outputs=outputs, show_progress=False)
    model_selected.change(fn=util.prepare_lcm_controlnet_or_sdxlturbo_pipeline, inputs=model_selected,outputs=selected_model_text)
    model_selected.change(fn=util.change_parameters, inputs=model_selected, outputs= [seed, 
                                                                                 prompt, 
                                                                                 output_width,
                                                                                    output_height,
                                                                                    stable_markdown,
                                                                                    guidance_scale, 
                                                                                    inference_steps,               
                                                                                    noise_strength,        
                                                                                    cond_scale,      
                                                                                    guidance_start,   
                                                                                    guidance_end, 
                                                                                    canny_markdown,
                                                                                    canny_lower_threshold,
                                                                                    canny_upper_threshold,
                                                                                    canny_aperture,
                                                                                    color_invert,
                                                                                    eta] )
    

    # seed.change(fn=process_image,inputs=inputs, outputs=outputs) # works
    # output_width.change(fn=process_image,inputs=inputs,outputs=outputs) # works
    # output_height.change(fn=process_image,inputs=inputs,outputs=outputs) # works
    
    # enter_prompt.click(fn=process_image, inputs=inputs,outputs=outputs)
    # enter_prompt.click(fn=update_state, inputs=["PROMPT", prompt])
    # prompt.change(fn=lambda: (update_state, process_image), inputs=[["PROMPT",prompt], inputs], outputs=[[],outputs])
    # enter_prompt.click(fn=update_state("PROMPT", prompt),inputs=[],outputs=prompt)


    # OTHER FUNCTIONALITIES
    # any inputs into GR objects need to be created by GR objects.
    refresh_button.click(fn=util.refresh_gallery, inputs=[], outputs=image_gallery) # no input, cus it always looks into folder
    capture_button.click(fn=util.capture_and_save_images, inputs=[], outputs=image_gallery) #no input cus it just saves the global image variable
    NewSessionButton.click(fn=lambda: (util.StartNewSessionFolder(), util.update_gallery_markdown()), inputs=[],outputs=[image_gallery,markdown_gallery_location])
    # NewSessionButton.click(fn=,update_gallery_markdown, inputs=[], outputs=markdown_gallery_location)

    # image_gallery.select(fn= lambda images: images, inputs=image_gallery, outputs=selected_images_txt)
    image_gallery.select(fn = show_warning, inputs = None)
    image_gallery.select(fn=util.select_images_for_print, inputs=[], outputs=selected_images_txt)
    reset_selection_btn.click(util.deselect_images_for_print, inputs=[], outputs= selected_images_txt)


    gen_photo_strip.click(fn=util.create_photo_strip, inputs=[nStrips, selected_images_txt, paper_width_mm, paper_height_mm], outputs=photo_strip_image)

    print_strip_btn.click(fn=util.print_file,inputs=printer, outputs=[])


    # previous_session_button.click(fn=NavigateGalleryFolders, inputs=[gr.Textbox(value="Previous")], outputs=image_gallery)
    # next_session_button.click(fn=NavigateGalleryFolders, inputs=[gr.Textbox(value="Next")], outputs=image_gallery)

    previous_session_button.click(fn=lambda: (util.NavigateGalleryFolders("Previous"), util.update_gallery_markdown()), 
                                  inputs=[], outputs=[image_gallery, markdown_gallery_location])

    next_session_button.click(fn=lambda: (util.NavigateGalleryFolders("Next"), util.update_gallery_markdown()), 
                              inputs=[],outputs=[image_gallery, markdown_gallery_location])
    

    ## ON LOAD:
    demo.load(fn=util.refresh_gallery, inputs=[], outputs=image_gallery)







demo.queue()
demo.launch()




   
