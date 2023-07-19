from diffusers import AudioLDMPipeline
import torch
import scipy
from datetime import datetime
import sys

def predict(prompt):
    repo_id = "cvssp/audioldm-s-full-v2"
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)
    # pipe = pipe.to("cuda")
    # commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")
    # prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
    audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

    # save the audio sample as a .wav file
    path = f"./static/sounds/{date_str}.wav"
    scipy.io.wavfile.write(path, rate=16000, data=audio)
    return path

if __name__ == "__main__":
    prompt = sys.argv[1]
    # prompt = input("プロンプトを入力");
    soudnsfile_path = predict(prompt)
    print(soudnsfile_path)
    

    
