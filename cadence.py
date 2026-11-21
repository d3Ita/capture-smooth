"""Copy or new frame, and the game's frame period.

A 30 fps game in a 60 Hz signal sends every frame twice. Through a capture card the copy is identical to the
bit (or nearly: capture noise), so a strict test is enough: fewer than DUP_SHARE of the pixels of the reduced
grayscale copy changed by DUP_LEVEL or more (see engines.py).

Through PS Remote Play the picture is compressed: the copy comes back slightly different. Measured on GTA 6
footage compressed like Remote Play (NVIDIA H.264, 1080p60), copies changed up to 1.2% of the pixels at 5 Mbit/s,
while new frames of a still scene changed as few as 0.003%: no threshold separates them. The cadence does:
the copy is the very next frame after the new one it repeats. So, in tolerant mode, the frame right after a
new one that changed little compared with recent new frames is a copy; a big change, or a frame further on,
is a new frame (a game stutter, a 60 fps game). On 20 s of that footage this took the copies taken for new
frames from 40 to 6 (12 Mbit/s) and from 174 to 11 (5 Mbit/s), with or without network jitter.
"""
import collections

DUP_SHARE = 4e-5    # same value as engines.DUP_SHARE

SMALL_REL = 0.1     # "changed little": under this share of the median change of recent new frames...
SMALL_ABS = 0.02    # ...and under 2% of the pixels at most (a real motion is never called a copy)
COPIES_SEEN = 5     # copies among the last 30 frames before the cadence is trusted (30 fps in 60 Hz: ~15)


class Cadence:
    def __init__(self, tolerant, period):
        self.tolerant = tolerant
        self.period = period                          # game frame period (s): median of recent intervals
        self.last_t = None                            # arrival of the last new frame
        self.intervals = collections.deque(maxlen=15)
        self.changes = collections.deque(maxlen=16)   # change share of recent new frames
        self.since = 0                                # frames received since the last new one
        self.recent = collections.deque(maxlen=30)    # copy (True) or new frame, for the last frames

    def is_dup(self, share, t):
        """share: fraction of the reduced pixels that changed; t: arrival time of the frame (s)."""
        dup = self.judge(share)
        self.recent.append(dup)
        if not dup:
            self.accept(share, t)
        return dup

    def judge(self, share):
        self.since += 1
        if share < DUP_SHARE:
            return True
        if self.tolerant and len(self.intervals) >= 3:
            # "Early" is counted in frames, not measured in time: the copy is the very next frame after the new
            # one it repeats, and network jitter cannot change that (a time test lost 10-13 new frames in 20 s
            # with 8 ms of jitter). Only when copies do arrive (a game slower than the stream): with a 60 fps
            # game, or a capture that skips unchanged frames, every frame comes right after a new one.
            copies_seen = sum(self.recent) >= COPIES_SEEN
            changes = sorted(self.changes)
            typical = changes[len(changes) // 2]
            if self.since == 1 and copies_seen and share < min(SMALL_REL * typical, SMALL_ABS):
                return True
        return False

    def accept(self, share, t):
        if self.last_t is not None:
            # The median sets aside gaps (pause, missed frame); the mean of the intervals near it is not pulled
            # by network jitter, as one late frame makes one interval longer and the next one shorter
            # (the median alone read 30.9 ms for 33.3 with 15 ms of jitter).
            self.intervals.append(min(t - self.last_t, 0.1))
            iv = sorted(self.intervals)
            med = iv[len(iv) // 2]
            near = [x for x in iv if 0.5 * med <= x <= 1.5 * med]
            self.period = min(max(sum(near) / len(near), 1 / 240), 1 / 20)
        self.last_t = t
        self.since = 0
        self.changes.append(share)



# Scene cuts (camera cut, cutscene, respawn): an in-between of two unrelated pictures is a ghostly blend,
# so the new picture is shown directly. Measured on GTA 6 driving footage with fast turns: on a 32x18
# thumbnail, motion never jumped above 12 levels AND 3x the recent average at once (0 / 1952 pairs),
# while 98.7% of cuts did.
CUT_LEVEL = 12.0
CUT_RATIO = 3.0


class SceneCuts:
    def __init__(self):
        self.motion = collections.deque(maxlen=8)  # recent thumbnail changes between new frames

    def is_cut(self, change, enabled=True):
        """change: mean absolute difference of the 32x18 thumbnails of two new frames (0-255 levels)."""
        if enabled and len(self.motion) >= 4 and change > CUT_LEVEL:
            m, n = sorted(self.motion), len(self.motion)
            if change > CUT_RATIO * ((m[(n - 1) // 2] + m[n // 2]) / 2 + 1.0):  # median, as before
                self.motion.clear()
                return True
        self.motion.append(change)
        return False
