"""
ctypes wrapper around libmp3lame for VBR MP3 encoding.

We ended up writing this because neither lameenc nor pyav could produce
VBR output — the file size was always constant regardless of how the
parameters were passed. Calling libmp3lame directly via ctypes was the
only in-process solution that worked.
I spoke with Gemini and Sonnet and tried all their ideas, none worked.
I searched the internet for other libraries, found no other option. 
"""
import ctypes
import ctypes.util
import numpy as np

LAME_HEADER_OVERHEAD = 7200

VBR_MTRH = 2

def _load_lame():
    name = ctypes.util.find_library("mp3lame") or "libmp3lame.so.0"
    lib = ctypes.CDLL(name)
    lib.lame_init.restype = ctypes.c_void_p
    lib.lame_set_num_channels.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lame_set_in_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lame_set_VBR.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lame_set_VBR_quality.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.lame_set_quality.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lame_init_params.argtypes = [ctypes.c_void_p]
    lib.lame_encode_buffer.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short),
        ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
    ]
    lib.lame_encode_buffer.restype = ctypes.c_int
    lib.lame_encode_flush.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
    lib.lame_encode_flush.restype = ctypes.c_int
    lib.lame_close.argtypes = [ctypes.c_void_p]
    return lib


_lame = _load_lame()


def encode_mp3(
    pcm_s16: np.ndarray,  # mono int16 samples
    sample_rate: int,
    vbr_quality: int = 4,   # 0=best/largest, 9=worst/smallest
    algorithm_quality: int = 5, # 0=best/slowest, 9=worst/fastest
) -> bytes:
    n_samples = len(pcm_s16)
    gfp = _lame.lame_init()
    if not gfp:
        raise RuntimeError("lame_init() failed")
    try:
        _lame.lame_set_num_channels(gfp, 1)
        _lame.lame_set_in_samplerate(gfp, sample_rate)
        _lame.lame_set_VBR(gfp, VBR_MTRH)
        _lame.lame_set_VBR_quality(gfp, vbr_quality)
        _lame.lame_set_quality(gfp, algorithm_quality)
        _lame.lame_init_params(gfp)

        # Lame recommends having at least 25% more samples than PCM, as a worst-case upper bound
        buf_size = n_samples + n_samples // 4 + LAME_HEADER_OVERHEAD
        mp3_buf = (ctypes.c_ubyte * buf_size)()
        pcm = pcm_s16.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        n = _lame.lame_encode_buffer(gfp, pcm, pcm, n_samples, mp3_buf, buf_size)
        if n < 0:
            raise RuntimeError(f"lame_encode_buffer() failed: {n}")
        n2 = _lame.lame_encode_flush(gfp, ctypes.cast(ctypes.addressof(mp3_buf) + n, ctypes.POINTER(ctypes.c_ubyte)), buf_size - n)
        if n2 < 0:
            raise RuntimeError(f"lame_encode_flush() failed: {n2}")
    finally:
        _lame.lame_close(gfp)

    return bytes(mp3_buf[:n + n2])
