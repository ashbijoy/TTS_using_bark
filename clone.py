from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
from scipy.io.wavfile import write as write_wav

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True) 
model.to ("cpu")

text = "Hi this is my project and I am Ashwin Bijoy and This is my voice."

output_dict = model. synthesize(
    text, 
    config, 
    speaker_id="speaker", 
    voice_dirs="bark_voices", 
    temperature=0.95,
)

write_wav("output1.wav", 24000, output_dict["wav"])
