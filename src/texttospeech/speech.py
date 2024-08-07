# pip install elevenlabs
# https://github.com/elevenlabs/elevenlabs-python

from elevenlabs.client import ElevenLabs
from elevenlabs import play

client = ElevenLabs(
  api_key="a90dda7c61c0107bf10f6b00ce20508f", # Defaults to ELEVEN_API_KEY
)

def speak(text: str):
    audio = client.generate(
      text=f"Yoooooo {text}! Now, let's explore and kill humans",
      voice="Nicole",
      model="turbo_v2"
    )
    play(audio)

client.models.get_all()

# Now you can call the function like this:
animal = "Tesla Cybertruck"
speak(f"I found {animal}")
