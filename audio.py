"""Plays the capture card's sound on the PC's default speakers, with an optional delay to stay in sync."""
import numpy as np
import sounddevice as sd


def _find(hostapi_name, wanted):
    apis = sd.query_hostapis()
    api = next((i for i, a in enumerate(apis) if hostapi_name in a["name"]), None)
    if api is None:
        return None
    devs = sd.query_devices()
    inp = next((i for i, d in enumerate(devs) if d["hostapi"] == api and d["max_input_channels"] > 0
                and wanted.lower() in d["name"].lower()), None)
    out = apis[api]["default_output_device"]
    if inp is None or out < 0:
        return None
    return inp, out, devs[inp]["name"], devs[out]["name"], int(devs[inp]["default_samplerate"])


class AudioSync:
    """Keeps the card's sound in step with the picture. Sound and picture leave the card together; the picture then
    takes Capture Smooth's added delay (measured every second) plus, on average, half a screen refresh before it
    shows, while the sound takes its own input/output buffering. The sound is delayed by the difference, plus a
    manual offset (F2 / F3, --audio-delay). Changed only when it moves by STEP ms or more: each change is a tiny
    jump in the sound."""
    STEP = 8.0

    def __init__(self, audio, refresh_hz, manual_ms=0.0, auto=True):
        self.audio, self.auto, self.manual = audio, auto, float(manual_ms)
        self.screen_ms = 500.0 / max(refresh_hz, 1)
        self.video_ms = None  # smoothed added delay of the picture

    def target(self):
        base = 0.0
        if self.auto and self.video_ms is not None:
            base = max(0.0, self.video_ms + self.screen_ms - self.audio.path_ms())
        return max(0.0, base + self.manual)

    def lead_ms(self):
        """How much earlier the sound would come than the picture without any delay (negative: it already comes
        later, as the sound path is longer; delaying the sound cannot help then)."""
        if self.video_ms is None:
            return None
        return self.video_ms + self.screen_ms - self.audio.path_ms()

    def update(self, video_ms):
        """Called once a second with the measured added delay (0 when no real frame was shown). True if changed."""
        if video_ms > 0:
            self.video_ms = video_ms if self.video_ms is None else 0.7 * self.video_ms + 0.3 * video_ms
        t = self.target()
        if abs(t - self.audio.delay_ms) >= self.STEP:
            self.audio.set_delay(t)
            return True
        return False

    def nudge(self, ms):
        self.manual += ms
        self.audio.set_delay(self.target())


class AudioPassthrough:
    def __init__(self, wanted="HD60", delay_ms=0.0):  # wanted: the video device name, or part of it
        self.muted = False
        self.volume = 1.0
        self.level = 0.0
        last_error = None
        # WASAPI has the lowest latency; DirectSound resamples if the two devices disagree on the rate.
        # The audio input usually carries the video device's name, sometimes without the brand word.
        words = wanted.split()
        keys = [wanted] + ([" ".join(words[1:])] if len(words) > 1 else [])
        for api, key in [(a, k) for a in ("WASAPI", "DirectSound") for k in keys]:
            found = _find(api, key)
            if not found:
                continue
            inp, out, self.in_name, self.out_name, rate = found
            try:
                self._setup(rate, delay_ms)
                self.stream = sd.Stream(device=(inp, out), samplerate=rate, channels=2, dtype="float32",
                                        latency="low", callback=self._callback)
                self.api = api
                return
            except Exception as e:  # try the next audio system
                last_error = e
        raise RuntimeError(f"son du boîtier indisponible ({last_error or f'aucune entrée audio « {wanted} »'})")

    def _setup(self, rate, delay_ms):
        self.rate = rate
        self.ring = np.zeros((rate, 2), np.float32)  # one second of history
        self.w = 0
        self.set_delay(delay_ms)

    def set_delay(self, delay_ms):
        self.delay_ms = max(0.0, min(delay_ms, 500.0))
        self.delay = int(self.rate * self.delay_ms / 1000)

    def _callback(self, indata, outdata, frames, time_info, status):
        n = len(self.ring)
        idx = (np.arange(frames) + self.w) % n
        self.ring[idx] = indata
        out = self.ring[(idx - self.delay) % n]
        self.w = (self.w + frames) % n
        self.level = float(np.abs(indata).max()) if frames else 0.0
        if self.muted:
            outdata.fill(0)
        else:
            outdata[:] = out * self.volume

    def path_ms(self):
        """The sound's own delay through this PC: input plus output buffering."""
        return sum(self.stream.latency) * 1000

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
