import argparse
import torch
import json
import os

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token

from transformers import TextStreamer
from PIL import Image
from io import BytesIO
import requests

direction_dict = {"front":0, "left":1,"back":2,"right":3}

def load_image(image_file):
    """Load an image from a file or URL."""
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        image_path = os.path.join(parent_dir, image_file)
        image = Image.open(image_path).convert('RGB')
    return image

def load_json_data(json_file):
    """Load a specific entry from a JSON file by ID."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data



def process_image(image, processor, config, overwrite_image_aspect_ratio=None):
    image_aspect_ratio = config.image_aspect_ratio
    image_size = image.size
    if overwrite_image_aspect_ratio is not None:
        image_aspect_ratio = overwrite_image_aspect_ratio
        
    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
        image = process_anyres_image(image, processor, config.image_grid_pinpoints)
    elif image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image, image_size, "image"

def process_images_with_tokens(obs_paths, map_paths, image_processor, config):
    """Process images and generate their corresponding tokens."""
    images = []
    for obs_path, map_path in zip(obs_paths, map_paths):
        obs_image = load_image(obs_path)
        images.append(obs_image)
        map_image = load_image(map_path)
        images.append(map_image)
        
    if type(images) is list:
        
        # Handling multi images
        # overwrite to process with simple pad 
        if len(images) > 1:
            image = [process_image(m, image_processor,config, "pad") for m in images]
            image = [[im[0], im[1], "image"] for im in image]
        else:
            image = [process_image(m,image_processor,config) for m in images]
    else:
        image = [process_image(images,image_processor,config)]

    #return torch.cat(image_tensors, dim=0)
    return image



import re
# def get_answer(output):
#     # 匹配 <...><...> 格式的字符串
#     match = re.match(r"<([^>]+)><([^>]+)>$", output)
#     if match:
#         reason = f"{match.group(1)}"  # 提取 <close>
#         direction = f"{match.group(2)}"  # 提取 <front>
#         return reason, direction
#     return None, None

def get_answer(output):
    match = re.search(r"<(\w+)><(\w+)>", output)

    if match:
        first_word = match.group(1)  # Captures the first word (back)
        second_word = match.group(2)  # Captures the second word (unseen)
        return first_word, second_word
    else:
        return None, None
        
    

def main(args):
    # Initialize model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, "auto", args.torch_type)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_1_5"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode



    # Load JSON data
    data = load_json_data(args.json_file)
    total_count = 0
    correct = 0
    view_count = 0
    locate_count = 0
    view_correct = 0
    locate_correct = 0
    unseen_count = 0
    unseen_correct = 0
    walkable_count = 0
    walkable_correct = 0
    collision_count = 0
    for item in data:
        if item["id"]>args.id:
            break
        current_obs_rgb = item["current_obs_rgb"]
        current_map_rgb = item["map_path"]
        conversations = item["conversations"]
        truth = item["conversations"][1]["value"]
        # Process images
        images = process_images_with_tokens(current_obs_rgb, current_map_rgb, image_processor, model.config)
        image_tensor = [im[0].to(dtype=torch.bfloat16) for im in images]

        # Build conversation
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles
        
        conv.append_message(conv.roles[0], conversations[0]["value"])
        conv.append_message(conv.roles[1], None)

        # Generate prompt
        prompt = conv.get_prompt()
        #print("Generate prompt:", prompt)
        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        #print(input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # Run inference
        # print("img size:", [single_image_tensor.shape for single_image_tensor in image_tensor])
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                image_sizes=[(384, 384)] #[torch([384, 384])]
            )

        # Decode output
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(f"Model Output: f{outputs}. Truth: {truth}")
        answer,reason  = get_answer(outputs)

        truth_answer, truth_reason = get_answer(truth)
        id = item["id"]
        print(f"Evaling: {id},  model: {outputs}/ {reason},{answer}, truth: {truth_reason},{truth_answer}  ")
        
        if "origin" in args.result_name:
            if reason:
                total_count += 1
                if item["viewed"] == 1:
                    view_count += 1
                if item["located"] == 1:
                    locate_count += 1           
        else:
            total_count += 1
            if item["viewed"] == 1:
                view_count += 1
            if item["located"] == 1:
                locate_count += 1
            if truth_reason == "unseen":
                unseen_count +=1
            elif truth_reason == "walkable":
                walkable_count += 1
        if answer == truth_answer:
            correct += 1
            if item["viewed"] == 1 and reason == truth_reason:
                view_correct += 1
            if item["located"] == 1 and reason == truth_reason:
                locate_correct += 1                
            if truth_reason == "unseen" and reason == truth_reason:
                unseen_correct +=1
            elif truth_reason == "walkable" and reason == truth_reason:
                walkable_correct += 1
        else:
            if item["cur_union_list"][direction_dict[answer]] < 70:
                collision_count +=1
        
    correct_rate = correct/total_count if total_count>0 else 0
    viewed_correct_rate = view_correct/view_count if view_count>0 else 0
    located_correct_rate = locate_correct/locate_count if locate_count>0 else 0
    
    unseen_correct_rate = unseen_correct/unseen_count if unseen_count>0 else 0
    walkable_correct_rate = walkable_correct/walkable_count if walkable_count>0 else 0
    collision_rate = collision_count/(total_count - correct) if (total_count - correct)>0 else 0
    # 文件路径
    output_file = f"/data2/zejinw/ON-MLLM/LLaVA-NeXT-wzj/data/result/{args.result_name}.txt"

    # 写入 TXT 文件
    with open(output_file, "w") as f:
        f.write(f"Total Count: {total_count}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"View Count: {view_count}\n")
        f.write(f"Locate Count: {locate_count}\n")
        f.write(f"View Correct: {view_correct}\n")
        f.write(f"Locate Correct: {locate_correct}\n")
        
        f.write(f"Correct Rate: {correct_rate}\n")
        f.write(f"View Correct Rate: {viewed_correct_rate}\n")
        f.write(f"Locate Correct Rate: {located_correct_rate}\n")
        
        f.write(f"unseen Count: {unseen_count}\n")
        f.write(f"walkable Count: {walkable_count}\n")
        f.write(f"unseen Rate: {unseen_correct_rate}\n")
        f.write(f"walkable Correct Rate: {walkable_correct_rate}\n")
        f.write(f"Collision Rate: {collision_rate}\n")
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode for the model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--json-file", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--id", type=int, required=True, help="ID of the entry in the JSON file")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--torch-type", type=str, default="bfloat16")
    parser.add_argument("--result_name", type=str, default="no_name")
    args = parser.parse_args()
    main(args)
