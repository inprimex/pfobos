/*
 * libfobos_stub.c — Fobos SDR stub library for testing without hardware.
 *
 * Implements all 16 fobos_rx_* API functions.  Signal synthesis is driven
 * by tests/stub/signals.json (path overridable via FOBOS_STUB_SIGNALS env).
 *
 * Signal types supported in JSON:
 *   {"type":"noise",  "amplitude":0.1}
 *   {"type":"tone",   "freq_hz":1000,  "amplitude":0.8}
 *   {"type":"fm",     "audio_hz":1000, "deviation":75000, "amplitude":0.8}
 *
 * Signals are additive.  The stub fills each read buffer with their sum.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * Minimal JSON parser — enough for our signals.json schema.
 * We parse only the fields we need without external dependencies.
 * ------------------------------------------------------------------------- */

#define MAX_SIGNALS 16
#define MAX_STR     64

typedef struct {
    char    type[MAX_STR];   /* "noise", "tone", "fm" */
    double  freq_hz;         /* tone: baseband frequency */
    double  audio_hz;        /* fm: audio tone frequency */
    double  deviation;       /* fm: frequency deviation (Hz) */
    double  amplitude;
} SignalDef;

typedef struct {
    double     sample_rate;
    int        n_signals;
    SignalDef  signals[MAX_SIGNALS];
} StubConfig;

/* Phase accumulators — persist across calls so waveforms are continuous */
static double g_phase_tone[MAX_SIGNALS]  = {0};
static double g_phase_fm_carrier[MAX_SIGNALS] = {0};
static double g_phase_fm_audio[MAX_SIGNALS]   = {0};

static StubConfig g_config = {
    .sample_rate = 2048000.0,
    .n_signals   = 2,
    .signals = {
        { "noise", 0, 0, 0, 0.05 },
        { "fm",    0, 1000, 75000, 0.8 }
    }
};

/* -- tiny helpers --------------------------------------------------------- */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int read_str(const char *p, char *out, int max_len) {
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1)
        out[i++] = *p++;
    out[i] = '\0';
    return (*p == '"') ? 1 : 0;
}

/* Very small JSON double reader — no error recovery */
static double read_double(const char *p, const char **end) {
    char buf[64]; int i = 0;
    while ((*p == '-' || (*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+') && i < 63)
        buf[i++] = *p++;
    buf[i] = '\0';
    if (end) *end = p;
    return atof(buf);
}

/* Find first occurrence of key:"  in json, return pointer past the colon */
static const char *find_key(const char *json, const char *key) {
    char needle[MAX_STR + 4];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (!p) return NULL;
    p += strlen(needle);
    p = skip_ws(p);
    if (*p != ':') return NULL;
    return skip_ws(p + 1);
}

static void parse_config(const char *json) {
    const char *p;

    /* sample_rate */
    p = find_key(json, "sample_rate");
    if (p) g_config.sample_rate = read_double(p, NULL);

    /* signals array */
    p = strstr(json, "\"signals\"");
    if (!p) return;
    p = strchr(p, '[');
    if (!p) return;
    p++;

    g_config.n_signals = 0;
    while (*p && g_config.n_signals < MAX_SIGNALS) {
        p = skip_ws(p);
        if (*p == ']') break;
        if (*p != '{') { p++; continue; }

        /* find closing brace of this object */
        const char *obj_start = p;
        int depth = 0;
        const char *obj_end = p;
        while (*obj_end) {
            if (*obj_end == '{') depth++;
            else if (*obj_end == '}') { depth--; if (depth == 0) { obj_end++; break; } }
            obj_end++;
        }
        /* copy object into temp buffer for parsing */
        int len = (int)(obj_end - obj_start);
        if (len >= 512) { p = obj_end; continue; }
        char obj[512];
        memcpy(obj, obj_start, len);
        obj[len] = '\0';

        SignalDef *s = &g_config.signals[g_config.n_signals];
        memset(s, 0, sizeof(*s));

        const char *v;
        v = find_key(obj, "type");
        if (v && *v == '"') read_str(v, s->type, MAX_STR);

        v = find_key(obj, "freq_hz");
        if (v) s->freq_hz = read_double(v, NULL);

        v = find_key(obj, "audio_hz");
        if (v) s->audio_hz = read_double(v, NULL);

        v = find_key(obj, "deviation");
        if (v) s->deviation = read_double(v, NULL);

        v = find_key(obj, "amplitude");
        if (v) s->amplitude = read_double(v, NULL);

        g_config.n_signals++;
        p = obj_end;
        p = skip_ws(p);
        if (*p == ',') p++;
    }
}

/* -- library constructor: load config ------------------------------------ */
__attribute__((constructor))
static void stub_init(void) {
    const char *path = getenv("FOBOS_STUB_SIGNALS");
    if (!path) {
        /* default: same directory as this .so */
        /* We use /proc/self/maps trick — simplest is just relative path */
        path = "tests/stub/signals.json";
    }
    FILE *f = fopen(path, "r");
    if (!f) {
        /* Try absolute path relative to CWD variants */
        static char alt[512];
        snprintf(alt, sizeof(alt), "%s", path);
        f = fopen(alt, "r");
    }
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        char *buf = malloc(sz + 1);
        if (buf) {
            fread(buf, 1, sz, f);
            buf[sz] = '\0';
            parse_config(buf);
            free(buf);
        }
        fclose(f);
    }
    /* else: use compiled-in defaults */
}

/* -------------------------------------------------------------------------
 * Signal synthesis
 * ------------------------------------------------------------------------- */
static void fill_buffer(float *buf, uint32_t n_floats) {
    /* n_floats = number of float values = 2 * n_iq_pairs (interleaved I,Q) */
    uint32_t n_samples = n_floats / 2;
    double sr = g_config.sample_rate;

    /* zero first */
    memset(buf, 0, n_floats * sizeof(float));

    for (int s = 0; s < g_config.n_signals; s++) {
        SignalDef *sig = &g_config.signals[s];

        if (strcmp(sig->type, "noise") == 0) {
            /* white noise via LCG — reproducible but not cryptographic */
            static uint32_t seed = 0xdeadbeef;
            for (uint32_t i = 0; i < n_samples; i++) {
                seed = seed * 1664525u + 1013904223u;
                float ni = ((int32_t)seed) / (float)0x80000000 * (float)sig->amplitude;
                seed = seed * 1664525u + 1013904223u;
                float nq = ((int32_t)seed) / (float)0x80000000 * (float)sig->amplitude;
                buf[i * 2]     += ni;
                buf[i * 2 + 1] += nq;
            }

        } else if (strcmp(sig->type, "tone") == 0) {
            /* Pure complex tone: e^(j*2π*f*t) */
            double phase_inc = 2.0 * M_PI * sig->freq_hz / sr;
            for (uint32_t i = 0; i < n_samples; i++) {
                buf[i * 2]     += (float)(sig->amplitude * cos(g_phase_tone[s]));
                buf[i * 2 + 1] += (float)(sig->amplitude * sin(g_phase_tone[s]));
                g_phase_tone[s] += phase_inc;
                if (g_phase_tone[s] > M_PI)  g_phase_tone[s] -= 2.0 * M_PI;
            }

        } else if (strcmp(sig->type, "fm") == 0) {
            /*
             * FM modulation: carrier modulated by audio sine.
             * Phase of carrier: φ(t) = 2π * ∫ (deviation * audio(τ)) dτ
             * audio(t) = sin(2π * audio_hz * t)
             * → φ(n) = φ(n-1) + 2π * deviation/sr * sin(audio_phase)
             */
            double audio_phase_inc    = 2.0 * M_PI * sig->audio_hz / sr;
            double carrier_phase_incr = 2.0 * M_PI * sig->deviation / sr;

            for (uint32_t i = 0; i < n_samples; i++) {
                double audio_sample = sin(g_phase_fm_audio[s]);
                g_phase_fm_carrier[s] += carrier_phase_incr * audio_sample;
                /* wrap carrier phase */
                while (g_phase_fm_carrier[s] >  M_PI) g_phase_fm_carrier[s] -= 2.0 * M_PI;
                while (g_phase_fm_carrier[s] < -M_PI) g_phase_fm_carrier[s] += 2.0 * M_PI;

                buf[i * 2]     += (float)(sig->amplitude * cos(g_phase_fm_carrier[s]));
                buf[i * 2 + 1] += (float)(sig->amplitude * sin(g_phase_fm_carrier[s]));

                g_phase_fm_audio[s] += audio_phase_inc;
                if (g_phase_fm_audio[s] > M_PI) g_phase_fm_audio[s] -= 2.0 * M_PI;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Fake device struct
 * ------------------------------------------------------------------------- */
typedef struct fobos_dev_t {
    double   frequency;
    double   sample_rate;
    uint32_t buf_length;   /* set by start_sync */
    int      sync_started;
    unsigned int lna_gain;
    unsigned int vga_gain;
} fobos_dev_t;

/* -------------------------------------------------------------------------
 * API implementation
 * ------------------------------------------------------------------------- */
int fobos_rx_get_api_info(char *lib_version, char *drv_version) {
    if (lib_version) strncpy(lib_version, "stub-1.0", 255);
    if (drv_version) strncpy(drv_version, "stub-1.0", 255);
    return 0;
}

int fobos_rx_get_device_count(void) {
    return 1;
}

int fobos_rx_list_devices(char *serials) {
    if (serials) strncpy(serials, "STUB0000001", 1023);
    return 0;
}

int fobos_rx_open(fobos_dev_t **out_dev, uint32_t index) {
    (void)index;
    fobos_dev_t *dev = calloc(1, sizeof(fobos_dev_t));
    if (!dev) return -3; /* NO_MEM */
    dev->frequency   = 100e6;
    dev->sample_rate = 2048000.0;
    *out_dev = dev;
    return 0;
}

int fobos_rx_close(fobos_dev_t *dev) {
    if (dev) free(dev);
    return 0;
}

int fobos_rx_reset(fobos_dev_t *dev) {
    (void)dev;
    return 0;
}

int fobos_rx_get_board_info(fobos_dev_t *dev,
                             char *hw_revision, char *fw_version,
                             char *manufacturer, char *product, char *serial) {
    (void)dev;
    if (hw_revision)  strncpy(hw_revision,  "1.0",           255);
    if (fw_version)   strncpy(fw_version,   "1.0",           255);
    if (manufacturer) strncpy(manufacturer, "StubCo",        255);
    if (product)      strncpy(product,      "StubSDR",       255);
    if (serial)       strncpy(serial,       "STUB0000001",   255);
    return 0;
}

int fobos_rx_set_frequency(fobos_dev_t *dev, double value, double *actual) {
    if (!dev) return -2;
    dev->frequency = value;
    if (actual) *actual = value;
    return 0;
}

int fobos_rx_set_direct_sampling(fobos_dev_t *dev, unsigned int enabled) {
    (void)dev; (void)enabled;
    return 0;
}

int fobos_rx_set_lna_gain(fobos_dev_t *dev, unsigned int value) {
    if (!dev) return -2;
    dev->lna_gain = value;
    return 0;
}

int fobos_rx_set_vga_gain(fobos_dev_t *dev, unsigned int value) {
    if (!dev) return -2;
    dev->vga_gain = value;
    return 0;
}

int fobos_rx_get_samplerates(fobos_dev_t *dev, double *values, unsigned int *count) {
    (void)dev;
    static const double rates[] = { 1000000.0, 2048000.0, 8000000.0, 20000000.0 };
    static const unsigned int n = 4;
    if (count) *count = n;
    if (values) {
        for (unsigned int i = 0; i < n; i++) values[i] = rates[i];
    }
    return 0;
}

int fobos_rx_set_samplerate(fobos_dev_t *dev, double value, double *actual) {
    if (!dev) return -2;
    dev->sample_rate = value;
    g_config.sample_rate = value;   /* keep synthesis in sync */
    if (actual) *actual = value;
    return 0;
}

/* Async: callback type matches fwrapper cdef */
typedef void(*fobos_rx_cb_t)(float *buf, uint32_t buf_length, void *ctx);

static volatile int g_async_cancel = 0;

int fobos_rx_read_async(fobos_dev_t *dev, fobos_rx_cb_t cb,
                         void *ctx, uint32_t buf_count, uint32_t buf_length) {
    if (!dev || !cb) return -2;
    if (buf_length == 0) buf_length = 32768;

    /* Empirically (2026-06-18 OPI5-1) the wrapper expects buf_length floats
     * in the buffer for async, not buf_length * 2. The libfobos source
     * suggests IQ pair count semantics but actual hardware behavior matches
     * the float-count interpretation. See pfobos/fwrapper.py async NOTE for
     * the full reasoning. Stub matches the wrapper's expectation so tests
     * align with hardware reality. */
    g_async_cancel = 0;
    float *buf = malloc(buf_length * sizeof(float));
    if (!buf) return -3;

    for (uint32_t i = 0; i < buf_count && !g_async_cancel; i++) {
        fill_buffer(buf, buf_length);
        cb(buf, buf_length, ctx);
    }
    free(buf);
    return 0;
}

int fobos_rx_cancel_async(fobos_dev_t *dev) {
    (void)dev;
    g_async_cancel = 1;
    return 0;
}

int fobos_rx_set_user_gpo(fobos_dev_t *dev, uint8_t value) {
    (void)dev; (void)value;
    return 0;
}

int fobos_rx_set_clk_source(fobos_dev_t *dev, int value) {
    (void)dev; (void)value;
    return 0;
}

int fobos_max2830_set_frequency(fobos_dev_t *dev, double value, double *actual) {
    (void)dev;
    if (actual) *actual = value;
    return 0;
}

int fobos_rffc507x_set_lo_frequency_hz(fobos_dev_t *dev,
                                        uint64_t lo_freq, uint64_t *tune_freq_hz) {
    (void)dev;
    if (tune_freq_hz) *tune_freq_hz = lo_freq;
    return 0;
}

int fobos_rx_start_sync(fobos_dev_t *dev, uint32_t buf_length) {
    if (!dev) return -2;
    dev->buf_length   = buf_length;
    dev->sync_started = 1;
    return 0;
}

int fobos_rx_read_sync(fobos_dev_t *dev, float *buf, uint32_t *actual_buf_length) {
    if (!dev || !dev->sync_started) return -7; /* SYNC_NOT_STARTED */
    /* Per real libfobos contract: buf_length is IQ pair count; the float*
     * buffer holds 2 floats per pair (interleaved I, Q). Write
     * dev->buf_length * 2 floats and report dev->buf_length pairs. */
    fill_buffer(buf, dev->buf_length * 2);
    if (actual_buf_length) *actual_buf_length = dev->buf_length;
    return 0;
}

int fobos_rx_stop_sync(fobos_dev_t *dev) {
    if (!dev) return -2;
    dev->sync_started = 0;
    return 0;
}

int fobos_rx_read_firmware(fobos_dev_t *dev, const char *file_name, int verbose) {
    (void)dev; (void)file_name; (void)verbose;
    return 0;
}

int fobos_rx_write_firmware(fobos_dev_t *dev, const char *file_name, int verbose) {
    (void)dev; (void)file_name; (void)verbose;
    return 0;
}

const char *fobos_rx_error_name(int error) {
    switch (error) {
        case  0: return "OK";
        case -1: return "NO_DEV";
        case -2: return "NOT_OPEN";
        case -3: return "NO_MEM";
        case -4: return "CONTROL";
        case -5: return "ASYNC_IN_SYNC";
        case -6: return "SYNC_IN_ASYNC";
        case -7: return "SYNC_NOT_STARTED";
        case -8: return "UNSUPPORTED";
        case -9: return "LIBUSB";
        default: return "UNKNOWN";
    }
}
