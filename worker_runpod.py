import os, json, requests, random, time, cv2, ffmpeg, psutil, itertools, runpod
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch.nn.functional as F
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS, load_custom_node
from comfy_extras import nodes_wan, nodes_model_advanced

load_custom_node("/content/ComfyUI/custom_nodes/comfyui_controlnet_aux")

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()

CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
ImageBatch = NODE_CLASS_MAPPINGS["ImageBatch"]()

WanVaceToVideo = nodes_wan.NODE_CLASS_MAPPINGS["WanVaceToVideo"]()
TrimVideoLatent = nodes_wan.NODE_CLASS_MAPPINGS["TrimVideoLatent"]()

KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

DWPreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("wan2.1_vace_14B_fp8_e4m3fn.safetensors", "default")[0]
    clip = CLIPLoader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan")[0]
    lora = LoraLoaderModelOnly.load_lora_model_only(unet, "FusionX/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors", 1.0)[0]
    vae = VAELoader.load_vae("wan_2.1_vae.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def images_to_mp4(images, output_path, fps=24):
    try:
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            if img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            frames.append(img)
        temp_files = [f"temp_{i:04d}.png" for i in range(len(frames))]
        for i, frame in enumerate(frames):
            success = cv2.imwrite(temp_files[i], frame[:, :, ::-1])
            if not success:
                raise ValueError(f"Failed to write {temp_files[i]}")
        if not os.path.exists(temp_files[0]):
            raise FileNotFoundError("Temporary PNG files were not created")
        stream = ffmpeg.input('temp_%04d.png', framerate=fps)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        for temp_file in temp_files:
            os.remove(temp_file)
    except Exception as e:
        print(f"Error: {e}")

def lazy_get_audio(video_path, start_time=0, duration=None):
    with VideoFileClip(video_path) as clip:
        if hasattr(clip, 'subclip'):
            if duration is not None:
                sub_clip = clip.subclip(start_time, start_time + duration)
            else:
                sub_clip = clip.subclip(start_time)
            audio = sub_clip.audio.to_soundarray(fps=44100) if sub_clip.audio else None
        else:
            audio = clip.audio.to_soundarray(fps=44100) if clip.audio else None
    return audio

def resized_cv_frame_gen(video_path, target_size=(640, 360)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    target_frame_time = 1.0 / fps if fps > 0 else 0
    alpha = False

    new_width, new_height = target_size

    yield (width, height, fps, duration, total_frames, target_frame_time, None, new_width, new_height, alpha)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, target_size)
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # FIX HERE
        frame_float = frame_resized.astype(np.float32) / 255.0
        yield frame_float

    cap.release()

def load_video(video_path):
    gen = resized_cv_frame_gen(video_path)

    width, height, fps, duration, total_frames, target_frame_time, _, new_width, new_height, alpha = next(gen)

    try:
        memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2**27
    except Exception:
        print("Memory check failed. Disabling memory limit.")
        memory_limit = float("inf")

    frame_mem = width * height * 3 * 0.1  # rough estimate
    max_frames = int(memory_limit // frame_mem)

    original_gen = gen
    gen = itertools.islice(gen, max_frames)
    img_shape = (new_height, new_width, 4 if alpha else 3)

    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, img_shape))))

    try:
        next(original_gen)
        raise RuntimeError(f"Memory limit hit after loading {len(images)} frames.")
    except StopIteration:
        pass

    if len(images) == 0:
        raise RuntimeError("No frames generated.")

    audio = lazy_get_audio(video_path, start_time=0, duration=len(images) * target_frame_time)

    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1 / target_frame_time if target_frame_time > 0 else 0,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }

    return images, len(images), audio, video_info

def resize_and_pad_image(
    image, 
    width=512, height=512, 
    keep_proportion="stretch", 
    pad_mode="color", 
    pad_color="0, 0, 0", 
    extra_padding=0,
    mask=None,
    upscale_method="bilinear",
    divisible_by=1,
    crop_position="center"
):
    """
    Resize and optionally pad the image (and mask).
    
    Args:
        image: torch tensor, shape (B,H,W,C), float32 in [0,1]
        width, height: target size
        keep_proportion: "stretch", "resize", "crop", "pad", "pad_edge"
        pad_mode: "color" or "edge"
        pad_color: string like "0, 0, 0" for RGB padding color
        extra_padding: int additional padding pixels
        mask: optional torch tensor mask (B,H,W)
        upscale_method: interpolation method for resize
        divisible_by: make width/height divisible by this
        crop_position: "center", "top", "bottom", "left", "right" for cropping/padding anchor
        
    Returns:
        resized_padded_image (torch tensor B,H,W,C),
        new_width (int),
        new_height (int),
        mask (torch tensor or zeros)
    """
    B, H, W, C = image.shape
    
    # Calculate new size based on keep_proportion
    if keep_proportion == "stretch":
        new_w, new_h = width, height
    else:
        # Aspect ratio
        aspect = W / H
        target_aspect = width / height
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                ratio = height / H
                new_w = round(W * ratio)
                new_h = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_w = width
                new_h = round(H * ratio)
            else:
                ratio = min(width / W, height / H)
                new_w = round(W * ratio)
                new_h = round(H * ratio)
        elif keep_proportion == "crop":
            if aspect > target_aspect:
                # wider -> crop width
                new_h = height
                new_w = round(height * aspect)
            else:
                # taller -> crop height
                new_w = width
                new_h = round(width / aspect)
        else:
            new_w, new_h = width, height
    
    # Adjust divisible_by
    if divisible_by > 1:
        new_w -= new_w % divisible_by
        new_h -= new_h % divisible_by
    
    # Resize image
    image_resized = F.interpolate(image.movedim(-1,1), size=(new_h,new_w), mode=upscale_method).movedim(1,-1)
    
    # Resize mask if present
    if mask is not None:
        mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=(new_h,new_w), mode='nearest').squeeze(1)
    else:
        mask_resized = None
    
    # Crop if keep_proportion=="crop"
    if keep_proportion == "crop":
        x, y = 0, 0
        if crop_position == "center":
            x = (new_w - width)//2
            y = (new_h - height)//2
        elif crop_position == "top":
            x = (new_w - width)//2
            y = 0
        elif crop_position == "bottom":
            x = (new_w - width)//2
            y = new_h - height
        elif crop_position == "left":
            x = 0
            y = (new_h - height)//2
        elif crop_position == "right":
            x = new_w - width
            y = (new_h - height)//2
        
        image_resized = image_resized[:, y:y+height, x:x+width, :]
        if mask_resized is not None:
            mask_resized = mask_resized[:, y:y+height, x:x+width]
        new_w, new_h = width, height
    
    # Add extra padding
    pad_left = extra_padding
    pad_right = extra_padding
    pad_top = extra_padding
    pad_bottom = extra_padding
    
    padded_w = new_w + pad_left + pad_right
    padded_h = new_h + pad_top + pad_bottom
    
    # Prepare output image tensor
    out_image = torch.zeros((B, padded_h, padded_w, C), dtype=image.dtype, device=image.device)
    
    # Parse pad color
    bg_color = [int(c.strip())/255 for c in pad_color.split(",")]
    if len(bg_color) == 1:
        bg_color = bg_color * 3
    bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)
    
    for b in range(B):
        if pad_mode == "edge":
            # Pad edges with edge pixel values
            top_edge = image_resized[b, 0, :, :].mean(dim=0)
            bottom_edge = image_resized[b, -1, :, :].mean(dim=0)
            left_edge = image_resized[b, :, 0, :].mean(dim=0)
            right_edge = image_resized[b, :, -1, :].mean(dim=0)
            
            out_image[b, :pad_top, :, :] = top_edge
            out_image[b, padded_h-pad_bottom:, :, :] = bottom_edge
            out_image[b, :, :pad_left, :] = left_edge
            out_image[b, :, padded_w-pad_right:, :] = right_edge
            out_image[b, pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = image_resized[b]
        else:
            # Pad with color
            out_image[b, :, :, :] = bg_color.view(1,1,-1)
            out_image[b, pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = image_resized[b]
    
    # Pad mask if any
    if mask_resized is not None:
        out_mask = F.pad(mask_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    else:
        out_mask = torch.zeros((B, padded_h, padded_w), dtype=image.dtype, device=image.device)
    
    return out_image, padded_w, padded_h, out_mask

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        input_image = values['input_image']
        input_image = download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')

        input_video = values['input_video']
        input_video = download_file(url=input_video, save_dir='/content/ComfyUI/input', file_name='input_video')

        positive_prompt = values['positive_prompt'] # A mid-shot of a asian woman in a sparkly pink crop top and low-rise cargo pants, dancing sharply in sync with the beat while singing straight into the camera. Her hair is styled in voluminous waves with front strands pulled into mini pigtails. Behind her, colored spotlights flash across a silver sequin curtain backdrop. Pure early-2000s pop performance.
        negative_prompt = values['negative_prompt'] # 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿
        width = values['width'] # 1024
        height = values['height'] # 576
        length = values['length'] # 121
        batch_size = values['batch_size'] # 1
        strength = values['strength'] # 1
        shift = values['shift'] # 1.0
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # uni_pc
        scheduler = values['scheduler'] # simple
        steps = values['steps'] # 8
        seed = values['seed'] # 1.0
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        fps = values['fps'] # 24
        filename_prefix = values['filename_prefix'] # wan_i2v_vace_CausVid

        images, frame_count, audio, video_info = load_video(input_video)
        resized_images, resized_images_new_w, resized_images_new_h, resized_images_mask = resize_and_pad_image(images, width=width, height=height, keep_proportion="crop", pad_mode="color", pad_color="0, 0, 0", extra_padding=0, upscale_method="bilinear", divisible_by=2, crop_position="center")
        pose_images = DWPreprocessor.estimate_pose(resized_images, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=1024, bbox_detector="yolox_l.onnx", pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")["result"][0]
        input_image = LoadImage.load_image(input_image)[0]
        resized_input_image, resized_input_image_new_w, resized_input_image_new_h, resized_input_image_mask = resize_and_pad_image(input_image, width=resized_images_new_w, height=resized_images_new_h, keep_proportion="stretch", pad_mode="color", pad_color="0, 0, 0", extra_padding=0, upscale_method="bilinear", divisible_by=2, crop_position="center")


        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
        model = ModelSamplingSD3.patch(lora, shift)[0]

        positive, negative, out_latent, trim_latent = WanVaceToVideo.encode(positive, negative, vae, resized_images_new_w, resized_images_new_h, frame_count, batch_size, strength, control_video=pose_images, reference_image=resized_input_image)
        samples = KSampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, out_latent)[0]
        out_samples = TrimVideoLatent.op(samples, trim_latent)[0]

        decoded_images = VAEDecode.decode(vae, out_samples)[0].detach()
        images_to_mp4(decoded_images, f"/content/wan2.1-i2v-vace-causvid-{seed}-tost.mp4", fps)
        
        result = f"/content/wan2.1-i2v-vace-causvid-{seed}-tost.mp4"

        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})