import time
import json
import requests

from tts_engine.codec import SNACCodec, get_or_load_tokenizer
from tts_engine.encoder import svara_text_to_tokens
from tts_engine.mapper import SvaraMapper

LLAMA_SERVER_URL = "http://localhost:8090/completion"
TOKENIZER_MODEL = "kenpath/svara-tts-v1"

TEXT = "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்? நான் நலம்."
SPEAKER_ID = "Tamil (Female)"

# ---------------------------------------------------------------------
# Step 0: Load tokenizer + codec (GPU) + warm up
# ---------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = get_or_load_tokenizer(TOKENIZER_MODEL)

print("Loading SNAC codec (GPU)...")
codec = SNACCodec(device="cuda")

print("Warming up SNAC decoder...")
codec.decode_window([1] * 28)
print("Warmup done.\n")

# ---------------------------------------------------------------------
# Step 1: Build prompt
# ---------------------------------------------------------------------
prompt = svara_text_to_tokens(
    text=TEXT,
    speaker_id=SPEAKER_ID,
    tokenizer=tokenizer,
    return_decoded=True,
)

# ---------------------------------------------------------------------
# Step 2: Stream from llama-server, decode windows as they arrive
# ---------------------------------------------------------------------
mapper = SvaraMapper(window_size=28)

pcm_chunks = []
request_start = time.time()
ttfb = None  # time to first audio byte
first_token_time = None
n_tokens_seen = 0

with requests.post(
    LLAMA_SERVER_URL,
    json={
        "prompt": prompt,
        "n_predict": 800,
        "temperature": 0.75,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "cache_prompt": True,
        "stream": True,
    },
    stream=True,
) as response:
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        token_text = data.get("content", "")
        if token_text and first_token_time is None:
            first_token_time = time.time()
        n_tokens_seen += 1

        # Feed into mapper -> get ready windows -> decode immediately
        windows = mapper.feed_text(token_text)
        for win in windows:
            pcm = codec.decode_window(win)
            if pcm and ttfb is None:
                ttfb = time.time() - request_start
                print(f">>> First audio byte at: {ttfb:.3f}s (TTFB)")
            pcm_chunks.append(pcm)

        if data.get("stop", False):
            break

total_time = time.time() - request_start
total_bytes = sum(len(c) for c in pcm_chunks)
audio_seconds = total_bytes / 2 / codec.sample_rate

print(f"\n=== STREAMING SUMMARY ===")
print(f"Time to first token   : {(first_token_time - request_start):.3f}s" if first_token_time else "N/A")
print(f"Time to first audio   : {ttfb:.3f}s (TTFB)" if ttfb else "N/A - no audio decoded")
print(f"Total generation time : {total_time:.3f}s")
print(f"Audio length          : {audio_seconds:.2f}s")
print(f"Real-time factor      : {total_time / audio_seconds:.2f}x" if audio_seconds > 0 else "N/A")

if total_bytes > 0:
    import numpy as np
    import soundfile as sf
    pcm = b"".join(pcm_chunks)
    audio_np = np.frombuffer(pcm, dtype="int16")
    sf.write("bench_streaming_output.wav", audio_np, codec.sample_rate)
    print("\nSaved: bench_streaming_output.wav")