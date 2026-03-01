"""
SDR Worker — background thread that reads IQ from FobosSDR (real or stub),
computes FFT, and pushes frames into an asyncio queue for WebSocket delivery.
"""

import asyncio
import threading
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shared.fwrapper import FobosSDR, FobosException

# Plasma colormap: 256 RGB triplets (uint8) precomputed from matplotlib
_PLASMA = [
    (13,8,135),(16,7,136),(19,7,137),(22,7,138),(25,6,140),(27,6,141),
    (29,6,142),(32,6,143),(34,5,144),(37,5,145),(39,5,146),(42,5,147),
    (44,5,148),(47,5,150),(49,5,151),(52,5,152),(54,5,153),(57,5,154),
    (59,5,155),(62,4,156),(64,4,157),(67,4,158),(69,4,160),(72,4,161),
    (74,4,162),(77,4,163),(79,4,164),(82,4,165),(84,4,166),(87,4,168),
    (89,4,169),(92,4,170),(94,4,171),(97,4,172),(99,4,173),(102,4,174),
    (104,4,176),(107,4,177),(109,4,178),(112,4,179),(114,4,180),(117,4,181),
    (119,5,182),(122,5,184),(124,5,185),(127,5,186),(129,5,187),(132,6,188),
    (134,6,189),(137,6,190),(139,7,192),(142,7,193),(144,8,194),(147,8,195),
    (149,9,196),(152,10,197),(154,10,198),(157,11,199),(159,12,200),(162,13,201),
    (164,14,203),(167,15,204),(169,16,205),(172,18,206),(174,19,207),(177,20,208),
    (179,22,209),(182,23,210),(184,25,211),(187,26,212),(189,28,213),(192,30,214),
    (194,32,215),(196,34,216),(199,36,217),(201,38,218),(203,40,219),(206,42,220),
    (208,44,221),(210,47,222),(213,49,223),(215,52,224),(217,55,225),(219,58,226),
    (222,61,227),(224,64,228),(226,68,229),(228,71,230),(230,75,231),(232,79,232),
    (234,83,233),(236,87,234),(238,91,235),(240,96,236),(242,100,236),(244,105,237),
    (246,110,238),(247,115,239),(249,120,239),(251,125,240),(252,130,241),(254,136,241),
    (254,141,242),(254,147,242),(254,152,243),(254,158,243),(254,163,244),(253,169,244),
    (253,174,244),(253,180,245),(253,185,245),(252,191,245),(252,196,246),(251,202,246),
    (251,207,246),(250,212,246),(250,218,247),(249,223,247),(249,228,247),(248,234,247),
    (247,239,248),(247,244,248),(246,250,248),(240,249,240),(233,248,232),(227,247,223),
    (220,246,215),(214,245,207),(207,244,198),(201,243,190),(194,242,182),(188,241,173),
    (181,240,165),(175,239,157),(168,238,148),(162,237,140),(155,236,132),(149,234,123),
    (142,233,115),(136,232,107),(129,231,98),(123,230,90),(116,229,82),(110,228,73),
    (103,227,65),(97,226,57),(90,225,48),(84,224,40),(77,222,32),(71,221,23),
    (64,220,15),(58,219,7),(51,218,0),(45,217,0),(38,215,0),(32,214,0),
    (25,213,0),(19,212,0),(12,211,0),(6,210,0),(0,208,0),(0,207,0),
    (0,206,0),(0,205,0),(0,204,0),(0,202,0),(0,201,0),(0,200,0),
    (0,199,0),(0,198,0),(0,196,0),(0,195,0),(0,194,0),(0,193,0),
    (0,192,0),(0,190,0),(0,189,0),(0,188,0),(0,187,0),(0,185,0),
    (0,184,0),(0,183,0),(0,182,0),(0,180,0),(0,179,0),(0,178,0),
    (0,177,0),(0,175,0),(5,174,0),(12,173,0),(18,171,0),(24,170,0),
    (30,169,0),(35,167,0),(41,166,0),(46,164,0),(52,163,0),(57,162,0),
    (62,160,0),(67,159,0),(72,157,0),(77,156,0),(82,155,0),(87,153,0),
    (92,152,0),(96,150,0),(101,149,0),(106,147,0),(110,146,0),(115,144,0),
    (120,143,0),(124,141,0),(129,140,0),(133,138,0),(138,137,0),(142,135,0),
    (147,134,0),(151,132,0),(155,131,0),(160,129,0),(164,128,0),(168,126,0),
    (173,124,0),(177,123,0),(181,121,0),(185,120,0),(189,118,0),(194,116,0),
    (198,115,0),(202,113,0),(206,111,0),(210,110,0),(214,108,0),(218,106,0),
    (222,104,0),(226,103,0),(229,101,0),(233,99,0),(237,97,0),(241,95,0),
    (244,93,0),(248,91,0),(252,89,0),(253,87,2),(254,85,6),(254,83,10),
    (254,81,14),(254,78,18),(254,76,22),(254,74,26),(254,72,31),(254,69,35),
    (254,67,40),(253,65,44),(253,62,49),(253,60,54),(252,58,59),(252,55,64),
    (251,53,69),(251,50,74),(250,48,79),(249,45,85),(249,43,90),(248,40,96),
    (247,38,101),(247,35,107),(246,32,112),(245,30,118),(244,27,124),(243,25,130),
    (242,22,136),(241,19,142),(240,17,148),(239,14,155),(237,12,161),(236,9,167),
    (235,7,174),(233,4,180),(232,2,187),(230,0,193),(229,0,200),(227,0,206),
    (225,0,213),(224,0,220),(222,0,226),(220,0,233),(218,0,240),(216,0,247),
    (240,249,33),(252,255,164),(253,231,37),(240,249,33),(252,255,164),(253,231,37),
]

# Trim / pad to exactly 256
_PLASMA = (_PLASMA * 2)[:256]


def _power_to_colormap_row(power_db: np.ndarray, vmin: float = -90.0, vmax: float = -20.0) -> list:
    """Map a power_db array to a list of [r,g,b] entries using the plasma palette."""
    normalized = np.clip((power_db - vmin) / (vmax - vmin), 0.0, 1.0)
    indices = (normalized * 255).astype(np.uint8)
    return [list(_PLASMA[i]) for i in indices]


class SDRWorker:
    """
    Reads IQ from FobosSDR in a background thread, computes FFT per frame,
    and pushes frame dicts to an asyncio queue.

    Frame dict keys:
        freqs        list[float]   — frequencies relative to center (Hz)
        spectrum     list[float]   — power in dB, length = fft_size
        iq_raw       list[float]   — interleaved I,Q floats (first IQ_RAW_MAX samples)
        waterfall_row list[list]   — [[r,g,b], ...] plasma-mapped row
        center_freq  float
        sample_rate  float
    """

    IQ_RAW_MAX = 2048    # max IQ pairs sent to browser for WebFFT / constellation

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        lib_path: str = None,
        fft_size: int = 1024,
        buf_length: int = 32768,
        center_freq: float = 100e6,
        sample_rate: float = 2.048e6,
        lna_gain: int = 1,
        vga_gain: int = 10,
    ):
        self._loop = loop
        self._queue = queue
        self._lib_path = lib_path
        self._stop_event = threading.Event()
        self._thread = None  # type: threading.Thread | None
        self._lock = threading.Lock()

        # SDR config — mutable via apply_config()
        self.fft_size = fft_size
        self.buf_length = max(buf_length, fft_size * 2)
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain

        # Pending config change to apply at next read iteration
        self._pending_config = None  # type: dict | None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sdr-worker")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def apply_config(self, config: dict):
        """Thread-safe config update; applied at start of next frame."""
        with self._lock:
            self._pending_config = config

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self):
        try:
            sdr = FobosSDR(lib_path=self._lib_path)
        except OSError as e:
            logger.error("SDRWorker: failed to load library: %s", e)
            return

        try:
            count = sdr.get_device_count()
            if count == 0:
                logger.error("SDRWorker: no Fobos SDR devices found")
                return
            sdr.open(0)
            sdr.set_frequency(self.center_freq)
            sdr.set_samplerate(self.sample_rate)
            sdr.set_lna_gain(self.lna_gain)
            sdr.set_vga_gain(self.vga_gain)
            sdr.start_rx_sync(self.buf_length)
            logger.info("SDRWorker: started — %.3f MHz, %.3f Msps", self.center_freq / 1e6, self.sample_rate / 1e6)

            while not self._stop_event.is_set():
                self._apply_pending_config(sdr)
                try:
                    iq = sdr.read_rx_sync()
                except FobosException as e:
                    logger.warning("SDRWorker: read error: %s", e)
                    continue

                frame = self._compute_frame(iq)
                asyncio.run_coroutine_threadsafe(self._queue.put(frame), self._loop)

        except Exception as e:
            logger.exception("SDRWorker: fatal error: %s", e)
        finally:
            try:
                sdr.stop_rx_sync()
                sdr.close()
            except Exception:
                pass
            logger.info("SDRWorker: stopped")

    def _apply_pending_config(self, sdr: FobosSDR):
        with self._lock:
            cfg = self._pending_config
            self._pending_config = None
        if cfg is None:
            return

        if "center_freq" in cfg:
            self.center_freq = float(cfg["center_freq"])
            sdr.set_frequency(self.center_freq)
        if "sample_rate" in cfg:
            self.sample_rate = float(cfg["sample_rate"])
            sdr.stop_rx_sync()
            sdr.set_samplerate(self.sample_rate)
            sdr.start_rx_sync(self.buf_length)
        if "lna_gain" in cfg:
            self.lna_gain = int(cfg["lna_gain"])
            sdr.set_lna_gain(self.lna_gain)
        if "vga_gain" in cfg:
            self.vga_gain = int(cfg["vga_gain"])
            sdr.set_vga_gain(self.vga_gain)
        if "fft_size" in cfg:
            self.fft_size = int(cfg["fft_size"])

        logger.info("SDRWorker: config applied: %s", cfg)

    def _compute_frame(self, iq: np.ndarray) -> dict:
        n = min(len(iq), self.fft_size)
        window = np.hanning(n)
        fft_out = np.fft.fftshift(np.fft.fft(iq[:n] * window, n))
        power_db = 20.0 * np.log10(np.abs(fft_out) + 1e-10)

        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / self.sample_rate)).tolist()

        # Raw IQ for WebFFT / constellation — interleaved float32, capped at IQ_RAW_MAX pairs
        raw_len = min(len(iq), self.IQ_RAW_MAX)
        iq_raw = np.empty(raw_len * 2, dtype=np.float32)
        iq_raw[0::2] = iq[:raw_len].real
        iq_raw[1::2] = iq[:raw_len].imag

        return {
            "freqs": freqs,
            "spectrum": power_db.tolist(),
            "iq_raw": iq_raw.tolist(),
            "waterfall_row": _power_to_colormap_row(power_db),
            "center_freq": self.center_freq,
            "sample_rate": self.sample_rate,
        }
