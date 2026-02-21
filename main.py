import asyncio

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from agents import Agent
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

load_dotenv()


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Be concise.",
    model="gpt-4o-mini",
)


async def main():
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

    # Record 3 seconds of audio from the microphone
    print("Speak now! Recording for 3 seconds...")
    sample_rate = 24000
    duration = 3
    recording = sd.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16,
    )
    sd.wait()
    print("Recording complete. Processing...")

    audio_input = AudioInput(buffer=recording.flatten())
    result = await pipeline.run(audio_input)

    # Play back the response
    player = sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.int16)
    player.start()
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)
    player.stop()
    player.close()


if __name__ == "__main__":
    asyncio.run(main())
