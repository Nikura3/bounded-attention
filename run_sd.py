import os
import torch

from diffusers import DDIMScheduler
from pipeline_stable_diffusion_opt import StableDiffusionPipeline
# from pytorch_lightning import seed_everything

from injection_utils import register_attention_editor_diffusers
from bounded_attention import BoundedAttention
import utils

import pandas as pd
from logger import logger
from utils import seed_everything
import time
import math
import torchvision.utils
import torchvision.transforms.functional as tf
from torchvision import transforms


def get_tokens_for_phrases(tokenizer, prompt, phrases):
    """
    Given a tokenizer, prompt, and list of phrases,
    returns the list of token indices corresponding to each phrase.
    """
    # Tokenize the full prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids[0]  # tensor shape (sequence_length,)

    # Decode the full prompt tokens to string tokens
    decoded_tokens = [tokenizer.decode([token_id]).strip() for token_id in input_ids]

    phrase_tokens_list = []

    for phrase in phrases:
        # Tokenize the phrase
        phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
        phrase_tokens = [tokenizer.decode([token_id]).strip() for token_id in phrase_ids]

        # Find the matching token span
        match = find_sublist(decoded_tokens, phrase_tokens)

        if match is not None:
            phrase_tokens_list.append(list(range(match[0], match[1]+1)))
        else:
            print(f"Warning: Could not find phrase '{phrase}' in prompt!")
            phrase_tokens_list.append([])

    return phrase_tokens_list

def find_sublist(full_list, sublist):
    """
    Finds the start and end indices of sublist in full_list.
    Returns (start_idx, end_idx) or None if not found.
    """
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i:i+len(sublist)] == sublist:
            return (i, i + len(sublist) - 1)
    return None


def normalize_data(data, size=512):
    return [ [coord / size for coord in box] for box in data ]

def read_prompts_csv(path):
    df = pd.read_csv(path, dtype={'id': str})
    conversion_dict = {}
    for i in range(len(df)):
        conversion_dict[df.at[i, 'id']] = {
            'prompt': df.at[i, 'prompt'],
            'obj1': df.at[i, 'obj1'],
            'bbox1': df.at[i, 'bbox1'],
            # 'token1': df.at[i, 'token1'],
            'obj2': df.at[i, 'obj2'],
            'bbox2': df.at[i, 'bbox2'],
            # 'token2': df.at[i, 'token2'],
            'obj3': df.at[i, 'obj3'],
            'bbox3': df.at[i, 'bbox3'],
            # 'token3': df.at[i, 'token3'],
            'obj4': df.at[i, 'obj4'],
            'bbox4': df.at[i, 'bbox4'],
            # 'token4': df.at[i, 'token4'],
        }
    return conversion_dict


def load_model(device):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}, torch_dtype=torch.float16, use_safetensors=True).to(device)
    model.enable_xformers_memory_efficient_attention()
    model.enable_sequential_cpu_offload()
    return model


def main():
    seeds = range(1,17)
    bench = read_prompts_csv(os.path.join("prompts","promptCollection.csv"))

    model_name="promptCollection-BA"
        
    if (not os.path.isdir("./results/"+model_name)):
        os.makedirs("./results/"+model_name)
    
    #intialize logger
    l=logger.Logger("./results/"+model_name+"/")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device)

    to_pil = transforms.ToPILImage()

    # ids to iterate the dict
    ids = []
    for i in range(0,len(bench)):
        ids.append(str(i).zfill(3))

    print("Start of generation process")

    for id in ids:
        bboxes = []
        phrases = []
        tokens = []  # e.g. [[2,4], [6,9]]

        if not (isinstance(bench[id]['obj1'], (int, float)) and math.isnan(bench[id]['obj1'])):
            phrases.append(bench[id]['obj1'])
            # tokens.append([int(bench[id]['token1'])])
            bboxes.append([int(x) for x in bench[id]['bbox1'].split(',')])
        if not (isinstance(bench[id]['obj2'], (int, float)) and math.isnan(bench[id]['obj2'])):
            phrases.append(bench[id]['obj2'])
            # tokens.append([int(bench[id]['token2'])])
            bboxes.append([int(x) for x in bench[id]['bbox2'].split(',')])
        if not (isinstance(bench[id]['obj3'], (int, float)) and math.isnan(bench[id]['obj3'])):
            phrases.append(bench[id]['obj3'])
            # tokens.append([int(bench[id]['token3'])])
            bboxes.append([int(x) for x in bench[id]['bbox3'].split(',')])
        if not (isinstance(bench[id]['obj4'], (int, float)) and math.isnan(bench[id]['obj4'])):
            phrases.append(bench[id]['obj4'])
            # tokens.append([int(bench[id]['token4'])])
            bboxes.append([int(x) for x in bench[id]['bbox4'].split(',')])

        output_path = "./results/"+model_name+"/"+ id +'_'+bench[id]['prompt'] + "/"

        if (not os.path.isdir(output_path)):
            os.makedirs(output_path)

        print("Sample number ", id)
        torch.cuda.empty_cache()

        gen_images=[]
        gen_bboxes_images=[]
        #BB: [xmin, ymin, xmax, ymax] normalized between 0 and 1

        prompt = bench[id]['prompt']
        tokens = get_tokens_for_phrases(model.tokenizer, prompt, phrases)
        normalized_boxes = normalize_data(bboxes)


        print(f"Prompt: {prompt}")
        print(f"# of bboxes: {len(tokens)}")
        
        for seed in seeds:
            print(f"Current seed is : {seed}")
            seed_everything(seed)

            # start stopwatch
            start = time.time()

            start_code = torch.randn([1, 4, 64, 64], device=device) # decoded into 512×512 pixel images
            
            editor = BoundedAttention(
                normalized_boxes,
                prompt,
                tokens,
                list(range(12, 20)),
                list(range(12, 20)),
                cross_mask_layers=list(range(14, 20)),
                self_mask_layers=list(range(14, 20)),
                filter_token_indices=None,
                eos_token_index=None,
                cross_loss_coef=1.5,
                self_loss_coef=0.5,
                max_guidance_iter=15,
                max_guidance_iter_per_step=5,
                start_step_size=8,
                end_step_size=2,
                loss_stopping_value=0.2,
                min_clustering_step=15,
                num_clusters_per_box=3,
            )

            register_attention_editor_diffusers(model, editor)
            images = model(prompt, latents=start_code, guidance_scale=7.5)

            # end stopwatch
            end = time.time()
            # save to logger
            l.log_time_run(start, end)

            #save the newly generated image
            image = images[0]
            image_pil = to_pil(image) # convert tensor to PIL image

            image_pil.save(output_path + "/" + str(seed) + ".jpg")  # save
            gen_images.append(image)
            image_for_draw = (image * 255).clamp(0, 255).to(torch.uint8)
            #draw the bounding boxes
            image=torchvision.utils.draw_bounding_boxes(image_for_draw,
                                                        torch.Tensor(bboxes),
                                                        labels=phrases,
                                                        colors=['green', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green'],
                                                        width=4,
                                                        font='font.ttf',
                                                        font_size=20)
            #list of tensors
            gen_bboxes_images.append(image)
            tf.to_pil_image(image).save(output_path+str(seed)+"_bboxes.png")

        # save a grid of results across all seeds without bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + ".png")

        # save a grid of results across all seeds with bboxes
        tf.to_pil_image(torchvision.utils.make_grid(tensor=gen_bboxes_images,nrow=4,padding=0)).save(output_path +"/"+ bench[id]['prompt'] + "_bboxes.png")

    # log gpu stats
    l.log_gpu_memory_instance()
    # save to csv_file
    l.save_log_to_csv(model_name)
    print("End of generation process")

if __name__ == "__main__":
    main()
