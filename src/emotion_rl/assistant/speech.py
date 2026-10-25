"""Text-to-speech and optional speech-to-text.

SpeechEngine: one background thread reads sentences from a queue and speaks
them, so the video never freezes. On Windows it calls the SAPI speech API
directly: pyttsx3 reuses (and caches) one engine, and in a background thread
that engine only speaks the *first* sentence; every later runAndWait() returned
in ~0.1 s in silence. That was v1's "speech only plays greeting" bug. On other
systems pyttsx3 is used with a fresh engine per sentence.

SpeechRecognizer is optional push-to-talk: it records a few seconds from the
microphone and transcribes them locally with faster-whisper, so what the user
says can go into the LLM prompt and the safety check. Install with
`pip install -e ".[voice]"`.
"""

import queue
import sys
import threading
import time

import numpy as np

NO_SPEAKER_HINTS = ("display audio", "hdmi", "displayport", "nvidia high definition")


def choose_output(names, wanted=None):
    """Index of the audio output to speak through.

    wanted: part of a device name ("Speakers", "Headphones"), or None/"auto".
    "auto" skips monitor audio (HDMI/DisplayPort "Display Audio"): Windows had
    made a monitor without speakers the default output, so every reply was silent.
    """
    lowered = [n.lower() for n in names]
    if wanted and wanted != "auto":
        return next((i for i, n in enumerate(lowered) if wanted.lower() in n), 0)
    return next((i for i, n in enumerate(lowered) if not any(h in n for h in NO_SPEAKER_HINTS)), 0)


class _SapiVoice:
    """Windows SAPI via COM; must be created in the thread that uses it."""

    def __init__(self, rate, volume, hint, output=None):
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self.voice = win32com.client.Dispatch("SAPI.SpVoice")
        self.voice.Rate = max(-10, min(10, round((rate - 180) / 15)))  # words/min -> SAPI scale
        self.voice.Volume = int(volume * 100)
        voices = self.voice.GetVoices()
        for i in range(voices.Count):
            if hint and hint.lower() in voices.Item(i).GetDescription().lower():
                self.voice.Voice = voices.Item(i)
                break
        outputs = self.voice.GetAudioOutputs()
        names = [outputs.Item(i).GetDescription() for i in range(outputs.Count)]
        if names:
            index = choose_output(names, output)
            self.voice.AudioOutput = outputs.Item(index)
            print(f"[Speech] speaking through: {names[index]}")

    def say(self, text, interrupted):
        """Speaks asynchronously so `interrupted()` can cut it short (the user pressed Talk)."""
        self.voice.Speak(text, 1)  # SVSFlagsAsync
        while not self.voice.WaitUntilDone(50):
            if interrupted():
                self.voice.Speak("", 3)  # SVSFlagsAsync | SVSFPurgeBeforeSpeak: stop now
                return


class _Pyttsx3Voice:
    def __init__(self, rate, volume, hint, output=None):  # output: system default only
        self.settings = (rate, volume, hint)

    def say(self, text, interrupted):  # pyttsx3 cannot be interrupted mid-sentence
        import gc

        import pyttsx3

        rate, volume, hint = self.settings
        engine = pyttsx3.init()  # fresh engine each time: a reused one goes silent
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        for voice in engine.getProperty("voices"):
            if hint and hint.lower() in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
        gc.collect()


class SpeechEngine:
    def __init__(self, rate=165, volume=1.0, voice_hint="zira", output="auto"):
        self.queue = queue.Queue()
        self.is_speaking = False
        self.spoken = 0
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._settings = (rate, volume, voice_hint, output)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            voice = (_SapiVoice if sys.platform == "win32" else _Pyttsx3Voice)(*self._settings)
        except ImportError:
            voice = _Pyttsx3Voice(*self._settings)
        while not self._stop.is_set():
            try:
                text = self.queue.get(timeout=0.3)
            except queue.Empty:
                continue
            self.is_speaking = True
            self._interrupt.clear()
            try:
                voice.say(text, self._interrupt.is_set)
                self.spoken += 1
            except Exception as exc:
                print(f"[Speech] failed: {exc}")
            finally:
                self.is_speaking = False

    def speak(self, text):
        """Non-blocking; replaces anything still waiting so speech never lags behind."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.queue.put(text)

    @property
    def busy(self):
        return self.is_speaking or not self.queue.empty()

    def interrupt(self):
        """Stop talking now and drop anything queued (the user wants to speak)."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self._interrupt.set()

    def stop(self):
        self._stop.set()


class SilentSpeech:
    """Drop-in replacement for --mute: prints instead of speaking."""

    is_speaking = False
    busy = False

    def speak(self, text):
        print(f"[Speech muted] {text}")

    def interrupt(self):
        pass

    def stop(self):
        pass


BLOCK_SECONDS = 0.1


# Whisper's usual inventions for near-silence; dropped unless clearly spoken at length
WHISPER_SILENCE_PHRASES = {"thank you.", "thank you", "thanks for watching!", "you", "bye.",
                           "thanks.", "okay.", "."}


def end_of_utterance(block_rms, end_silence=1.2, max_wait=5.0, max_seconds=12.0,
                     min_speech_blocks=3):
    """Decide when to stop recording from per-block loudness (one value per 0.1 s).

    Speech = blocks louder than 4x the room's noise floor (10th percentile, robust to the
    user starting to talk at once and to the all-zero blocks some mics send first), and
    at least `min_speech_blocks` of them, so a key click is not mistaken for talking.
    Stop after `end_silence` s of quiet once speech has started, after `max_wait` s if
    nobody speaks, or at `max_seconds`. Returns (stop, speech_started).
    """
    n = len(block_rms)
    if n == 0:
        return False, False
    threshold = max(4 * float(np.percentile(block_rms, 10)), 0.003)
    loud = [r > threshold for r in block_rms]
    started = sum(loud) >= min_speech_blocks
    if n * BLOCK_SECONDS >= max_seconds:
        return True, started
    if not started:
        return n * BLOCK_SECONDS >= max_wait, False
    last_loud = max(i for i, is_loud in enumerate(loud) if is_loud)
    return (n - 1 - last_loud) * BLOCK_SECONDS >= end_silence, True


def normalise(audio, peak=0.9):
    """Scale quiet recordings up: a webcam mic at arm's length records very softly."""
    top = float(np.abs(audio).max()) if len(audio) else 0.0
    return audio * (peak / top) if top > 1e-4 else audio


class SpeechRecognizer:
    def __init__(self, model_size="small.en", sample_rate=16_000, device=None, save_dir=None,
                 end_silence=1.2, max_seconds=12.0):
        """
        model_size : faster-whisper model; small.en (~2 s per sentence on CPU) is far more
                     robust to accents and room noise than base.en
        device     : microphone name (or part of it, e.g. "Brio"); None = system default
        save_dir   : if set, keep each recording as a WAV there (for debugging; off by default)
        """
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.device = device
        self.sample_rate = sample_rate
        self.save_dir = save_dir
        self.end_silence = end_silence
        self.max_seconds = max_seconds
        self.seconds = max_seconds  # longest possible recording (shown in the UI)
        self.listening = False  # True only while the microphone is actually recording
        self.waiting = False  # True while waiting for the assistant to stop talking
        self.last_text = None
        self._lock = threading.Lock()

    def transcribe(self, audio):
        """audio: float32 mono array at `sample_rate`, or a path to an audio file."""
        if not isinstance(audio, str):
            audio = normalise(np.asarray(audio, dtype=np.float32))
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True, beam_size=5,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def record(self):
        """Record until the user stops talking (see end_of_utterance).
        Returns (audio, speech_detected)."""
        import sounddevice as sd

        block = int(BLOCK_SECONDS * self.sample_rate)
        chunks, rms = [], []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            device=self.device, blocksize=block) as stream:
            while True:
                data, _ = stream.read(block)
                chunks.append(data[:, 0].copy())
                rms.append(float(np.sqrt(np.mean(data ** 2))))
                stop, started = end_of_utterance(rms, self.end_silence,
                                                 max_seconds=self.max_seconds)
                if stop:
                    break
        audio = np.concatenate(chunks)
        if self.save_dir:
            self._save(audio)
        return audio, started

    def _save(self, audio):
        import os
        import wave

        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, time.strftime("talk_%Y%m%d-%H%M%S.wav"))
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    def listen_async(self, on_text, wait_until_quiet=None):
        """Record in the background until the user pauses, then call on_text(transcript).

        wait_until_quiet: optional callable returning True while the assistant is still
        talking; recording starts only after it returns False, so the microphone does not
        transcribe the assistant's own voice (that happened in the first live test).
        """
        if not self._lock.acquire(blocking=False):
            return False

        def work():
            try:
                self.waiting = True
                deadline = time.time() + 3.0
                while wait_until_quiet and wait_until_quiet() and time.time() < deadline:
                    time.sleep(0.05)
                time.sleep(0.25)  # let the speaker's last syllable die away
                self.waiting, self.listening = False, True
                audio, speech = self.record()
                self.listening = False
                # Whisper invents text ("Thank you.") for pure room noise, so skip it
                text = self.transcribe(audio) if speech else ""
                if text.lower() in WHISPER_SILENCE_PHRASES:
                    text = ""
                self.last_text = text
                on_text(text)
            except Exception as exc:
                print(f"[Speech input] failed: {exc}")
            finally:
                self.waiting = self.listening = False
                self._lock.release()

        threading.Thread(target=work, daemon=True).start()
        return True

    @property
    def busy(self):
        return self.waiting or self.listening
