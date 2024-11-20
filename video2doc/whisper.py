from transformers import pipeline
import torch


import os
from moviepy import VideoFileClip


def extract_audio(video_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.wav'
    
    if os.path.exists(output_path):
        return output_path
    
    video = VideoFileClip(video_path)
    
    audio = video.audio
    
    audio.write_audiofile(output_path)
    
    video.close()
    
    return output_path



class Whisper:

    def __init__(self, model_name: str = "openai/whisper-tiny.en", device: str = "cuda"):
        self.model = pipeline("automatic-speech-recognition", model=model_name, torch_dtype=torch.bfloat16, device=device)
    
    def __call__(self, audio_path: str) -> str:
        return self.model(audio_path, return_timestamps=True, chunk_length_s=30)


def merge_chunks_by_timerange(chunks, start_time, end_time):
    relevant_chunks = []
    
    for chunk in chunks:
        chunk_start, chunk_end = chunk['timestamp']
        
        if not (chunk_end < start_time or chunk_start > end_time):
            relevant_chunks.append(chunk)
    
    merged_text = ' '.join(chunk['text'].strip() for chunk in relevant_chunks)
    
    return merged_text