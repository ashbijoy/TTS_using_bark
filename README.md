# README for TTS using Bark and Voice Cloning
This repository demonstrates how to generate Text-to-Speech (TTS) audio using the Bark model, a state-of-the-art text-to-speech model, and perform voice cloning to create personalized voices. The project leverages transformers for Bark integration and the TTS library for voice cloning.

Requirements
To use this repository, you must have the following dependencies installed:

Python 3.x

transformers

scipy

torch (for CPU support)

TTS

bark

You can install the required dependencies using the following command:

bash
Copy
pip install transformers scipy torch TTS
Setting Up
TTS with Bark
This section demonstrates how to generate TTS audio using the Bark model from the transformers library.

Code Example:
python
Copy
from transformers import AutoProcessor, BarkModel
import scipy

# Load the pre-trained processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to("cpu")  # Using CPU since CUDA is not enabled

# Function to generate audio from text
def generate_audio(text, preset, output):
    inputs = processor(text, voice_preset=preset)
    for k, v in inputs.items():
        inputs[k] = v.to("cpu")  # Move tensors to CPU
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output, rate=sample_rate, data=audio_array)

# Generate audio using a preset
generate_audio(
    text="Hi, this is my project that I am Ashwin Bijoy", 
    preset="v2/en_speaker_6",  # Specify speaker preset
    output="output.wav"
)
Explanation:
AutoProcessor and BarkModel from transformers are used to load the pre-trained Bark model.

generate_audio function takes in the text, preset for the voice, and the output file name.

It generates audio and saves it as a .wav file using the scipy.io.wavfile.write() function.

Voice Cloning with Bark
The following code demonstrates how to clone a voice using the Bark model, where the synthesized speech can be personalized with a specific voice preset.

Code Example:
python
Copy
from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from scipy.io.wavfile import write as write_wav

# Load the Bark model with custom configuration
config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)
model.to("cpu")

# Text to be synthesized
text = "Hi, this is my project and I am Ashwin Bijoy. This is my voice."

# Generate audio with voice cloning
output_dict = model.synthesize(
    text, 
    config, 
    speaker_id="speaker",  # Specify the speaker's voice ID
    voice_dirs="bark_voices",  # Directory containing voice data
    temperature=0.95,  # Control speech randomness
)

# Save the generated speech as a .wav file
write_wav("output1.wav", 24000, output_dict["wav"])
Explanation:
BarkConfig and Bark from the TTS library are used to initialize the Bark model.

synthesize method generates audio with the specified text, voice cloning settings (like speaker_id and voice_dirs), and other parameters like temperature.

The output is saved as a .wav file using write_wav().

Cloning Your Voice
To clone your own voice, you need to:

Prepare a dataset of your voice recordings. These should be in a directory (e.g., bark_voices/).

Train or fine-tune the Bark model on your voice dataset, specifying the speaker's voice ID in the speaker_id parameter when synthesizing.

Voice cloning can take some time and resources depending on the quality and quantity of the voice data provided. Make sure to have a good amount of data for accurate voice synthesis.

Running the Code
To run the code examples, simply save the provided Python code into a .py file and execute it. Ensure you have the correct file paths for the voice directories and checkpoints when required.

bash
Copy
python generate_tts.py
Notes
CPU-only support: The code is written to work on CPU. If you have access to a CUDA-enabled GPU, you can modify the .to("cpu") calls to .to("cuda") to improve performance.

Preset selection: The preset parameter in the Bark model refers to predefined voice styles. You can find various voice presets available in the Bark model repository or define your own.

Conclusion
This repository demonstrates the power of Bark and voice cloning technologies for creating high-quality TTS outputs. With proper voice datasets, you can synthesize personalized voices, making your applications more engaging and interactive.
