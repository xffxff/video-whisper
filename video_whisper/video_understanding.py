import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def load_video(video_file, num_frames=128):
    from decord import VideoReader
    vr = VideoReader(video_file)
    duration = len(vr)
    fps = vr.get_avg_fps()
            
    frame_timestamps = [int(duration / num_frames * (i+0.5)) / fps for i in range(num_frames)]
    actual_frame_indices = [int(duration / num_frames * (i+0.5)) for i in range(num_frames)]

    frames_data = vr.get_batch(actual_frame_indices).asnumpy()
    frames = [Image.fromarray(frame_data).convert("RGB") for frame_data in frames_data]
        
    return frames, frame_timestamps


def construct_messages(frames: list[Image.Image], timestamps: list[float], prompt: str) -> list[dict]:
    messages = []
    if not timestamps:
        for i, _ in enumerate(frames):
            messages.append({"text": None, "type": "image"})
        messages.append({"text": "\n", "type": "text"})
    else:
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            messages.extend(
                [
                    {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                    {"text": None, "type": "image"},
                    {"text": "\n", "type": "text"}
                ]
            )
    messages.append({"text": prompt, "type": "text"})

    return [
        {"role": "user", "content": messages}
    ]



class VideoUnderstandingWithAria:

    def __init__(self, model_id_or_path: str, device_map: str = "auto"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
    
    def _default_prompt(self) -> str:
        return """Please split this video into scenes, providing start time, end time, a title and detailed descriptions for each scene. 
        Ignore scenes less than 5 seconds in duration.
        Format the output as JSON with the following structure:
        {
            "scenes": [
                {
                    "start_time": "MM:SS",
                    "end_time": "MM:SS", 
                    "title": "Scene title",
                    "description": "Detailed scene description"
                }
            ]
        }
        And make sure the output can be loaded by json.loads() in Python. Do not include any other text than the JSON, such as newlines or markdown.
        """
    
    def __call__(self, frames: list[Image.Image], timestamps: list[float], prompt: str = None) -> str:
        prompt = prompt or self._default_prompt()
        messages = construct_messages(frames, timestamps, prompt)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=frames, return_tensors="pt", max_image_size=490)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                stop_strings=["<|im_end|>"],
                tokenizer=self.processor.tokenizer,
                do_sample=False,
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = self.processor.decode(output_ids, skip_special_tokens=True)
        return result
