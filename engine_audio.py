#!/usr/bin/env python3
import math
import time
import os

import numpy as np
import sounddevice as sd

# ==========================
# Paramètres base moteur
# ==========================
SAMPLE_RATE = 44100

IDLE_RPM = 900        # ralenti
MAX_RPM = 7000        # régime max théorique
LIMIT_RPM = 6800      # rupteur (limiteur)

NUM_CYLINDERS = 4     # valeur par défaut

BASS_BOOST = 1.0      # 1.0 = neutre, >1 = plus de graves

RPM_TIME_CONSTANT = 0.15

CAM_LUMP = 0.5        # 0 = lisse, 1 = très “lumpy”
UNEVEN_FIRE = 0.0     # firing irrégulier
WHINE_GAIN = 0.0      # sifflement

PRESETS = {
    "1": {
        "name": "4 cylindres route",
        "cyl": 4,
        "idle": 900,
        "max": 6500,
        "bass": 1.0,
        "cam_lump": 0.3,
        "uneven": 0.1,
        "whine": 0.15,
    },
    "2": {
        "name": "V8 muscle",
        "cyl": 8,
        "idle": 800,
        "max": 6000,
        "bass": 2.0,
        "cam_lump": 0.8,
        "uneven": 0.6,
        "whine": 0.10,
    },
    "3": {
        "name": "Moto 2 cylindres",
        "cyl": 2,
        "idle": 1300,
        "max": 10000,
        "bass": 0.8,
        "cam_lump": 0.6,
        "uneven": 0.3,
        "whine": 0.35,
    },
}

current_preset_name = "4 cylindres route"

# Variables dynamiques partagées
engine_rpm = IDLE_RPM
throttle   = 0.0        # 0..1, piloté par le potentiomètre
running    = True

phase = 0.0
env_phase = 0.0
whine_phase = 0.0

rumble_state = 0.0
roar_state = 0.0
hiss_state = 0.0

_last_audio_log = 0.0  # log audio périodique

# Flux audio global (pour pouvoir stop proprement)
_audio_stream = None


def clamp(x, a, b):
    return max(a, min(b, x))


def apply_preset(key: str):
    """
    Change les paramètres moteur selon un preset (1, 2, 3).
    """
    global NUM_CYLINDERS, BASS_BOOST, IDLE_RPM, MAX_RPM, LIMIT_RPM
    global engine_rpm, current_preset_name, CAM_LUMP, UNEVEN_FIRE, WHINE_GAIN

    preset = PRESETS.get(key)
    if not preset:
        return

    NUM_CYLINDERS = preset["cyl"]
    BASS_BOOST = preset["bass"]
    IDLE_RPM = preset["idle"]
    MAX_RPM = preset["max"]
    LIMIT_RPM = int(MAX_RPM * 0.97)
    CAM_LUMP = preset.get("cam_lump", 0.5)
    UNEVEN_FIRE = preset.get("uneven", 0.0)
    WHINE_GAIN = preset.get("whine", 0.0)

    engine_rpm = max(engine_rpm, IDLE_RPM)
    current_preset_name = preset["name"]

    print(
        f"-> Preset {key}: {current_preset_name} "
        f"({NUM_CYLINDERS} cyl, ralenti={IDLE_RPM} RPM, "
        f"rupteur≈{LIMIT_RPM} RPM, bass={BASS_BOOST:.2f}, cam={CAM_LUMP:.2f})"
    )


def set_throttle(value: float):
    """
    Met à jour la position de gaz (0..1) depuis l'extérieur (joystick).
    """
    global throttle
    throttle = clamp(float(value), 0.0, 1.0)


def stop_engine():
    """
    Demande l'arrêt du moteur et stoppe le flux audio.
    """
    global running, _audio_stream
    running = False
    if _audio_stream is not None:
        try:
            _audio_stream.stop()
            _audio_stream.close()
        except Exception:
            pass
        _audio_stream = None


def engine_sample(rpm, frames, decel_amount=0.0):
    """
    Génère 'frames' échantillons audio pour un régime donné.
    """
    global phase, env_phase, whine_phase, throttle
    global rumble_state, roar_state, hiss_state
    global NUM_CYLINDERS, BASS_BOOST, IDLE_RPM, MAX_RPM, CAM_LUMP, UNEVEN_FIRE, WHINE_GAIN

    firings_per_rev = max(1, NUM_CYLINDERS // 2)

    base_freq = (rpm / 60.0) * firings_per_rev
    if base_freq < 5.0:
        base_freq = 5.0

    phase_inc = 2.0 * math.pi * base_freq / SAMPLE_RATE
    phase_array = phase + phase_inc * np.arange(frames, dtype=np.float32)
    phase = (phase + phase_inc * frames) % (2.0 * math.pi)

    rpm_ratio = clamp((rpm - IDLE_RPM) / max(1.0, (MAX_RPM - IDLE_RPM)), 0.0, 1.0)

    # ================= Tonal (harmoniques) =================
    fundamental = np.sin(phase_array)
    harm2 = np.sin(2.0 * phase_array)
    harm3 = np.sin(3.0 * phase_array)
    harm4 = np.sin(4.0 * phase_array)

    w1 = 0.8 - 0.4 * rpm_ratio
    w2 = 0.25 + 0.15 * rpm_ratio
    w3 = 0.10 + 0.25 * rpm_ratio
    w4 = 0.05 + 0.25 * rpm_ratio

    bass_tone_factor = 0.7 + 0.3 * BASS_BOOST
    w1 *= bass_tone_factor

    tone = w1 * fundamental + w2 * harm2 + w3 * harm3 + w4 * harm4

    # ================= Pulses par cylindre =================
    width = clamp(0.5 - 0.25 * rpm_ratio, 0.12, 0.5)
    cyl_offsets = np.linspace(0.0, 2.0 * math.pi, NUM_CYLINDERS, endpoint=False, dtype=np.float32)
    phases_cyl = phase_array[:, None] + cyl_offsets[None, :]
    phases_cyl_wrapped = (phases_cyl + math.pi) % (2.0 * math.pi) - math.pi
    pulses_cyl = np.exp(- (phases_cyl_wrapped / width) ** 2).astype(np.float32)
    pulses = pulses_cyl.mean(axis=1)

    low_rpm_factor = 1.0 - rpm_ratio
    if CAM_LUMP > 0.0:
        shape_power = 1.0 + 2.5 * CAM_LUMP * low_rpm_factor
        pulses = pulses ** shape_power

    # ================= Bruits filtrés =================
    white = np.random.randn(frames).astype(np.float32)
    rumble = np.empty(frames, dtype=np.float32)
    roar = np.empty(frames, dtype=np.float32)
    hiss = np.empty(frames, dtype=np.float32)

    rumble_alpha = 0.995
    roar_alpha = 0.96
    hiss_alpha = 0.80

    for i in range(frames):
        x = white[i]
        rumble_state = rumble_alpha * rumble_state + (1.0 - rumble_alpha) * x
        rumble[i] = rumble_state
        roar_state = roar_alpha * roar_state + (1.0 - roar_alpha) * x
        roar[i] = roar_state - 0.7 * rumble_state
        hiss_state = hiss_alpha * hiss_state + (1.0 - hiss_alpha) * x
        hiss[i] = x - hiss_state

    rumble_gain = (0.5 * (1.0 - rpm_ratio) + 0.2) * BASS_BOOST
    roar_gain   = 0.3 + 0.6 * rpm_ratio
    hiss_gain   = 0.05 + 0.4 * throttle * rpm_ratio

    noise_layer = rumble_gain * rumble + roar_gain * roar + hiss_gain * hiss

    # ================= Pops à la décélération =================
    pops = np.zeros(frames, dtype=np.float32)
    if decel_amount > 0.05 and throttle < 0.2:
        pop_density = 0.005 + 0.04 * decel_amount
        pop_noise = np.random.randn(frames).astype(np.float32)
        mask = (np.random.rand(frames) < pop_density).astype(np.float32)
        pops = pop_noise * mask
        pops = np.tanh(3.0 * pops) * (0.2 + 0.8 * decel_amount)

    # ================= Enveloppe basse fréquence =================
    lump_freq = base_freq * (4.0 / max(4, NUM_CYLINDERS))
    env_inc = 2.0 * math.pi * lump_freq / SAMPLE_RATE
    env_phase_array = env_phase + env_inc * np.arange(frames, dtype=np.float32)
    env_phase = (env_phase + env_inc * frames) % (2.0 * math.pi)

    envelope = 0.5 + 0.5 * np.sin(env_phase_array)
    envelope = 0.25 + 0.75 * envelope
    envelope *= (0.8 + 0.4 * rpm_ratio)

    if UNEVEN_FIRE > 0.0:
        alt_env = 0.5 + 0.5 * np.sin(0.5 * env_phase_array + 1.3)
        envelope *= 1.0 + UNEVEN_FIRE * 0.6 * (alt_env - 0.5)

    # ================= Sifflement (whine) =================
    whine = np.zeros(frames, dtype=np.float32)
    if WHINE_GAIN > 0.0:
        whine_freq = base_freq * (5.0 + 5.0 * rpm_ratio)
        whine_inc = 2.0 * math.pi * whine_freq / SAMPLE_RATE
        whine_phases = whine_phase + whine_inc * np.arange(frames, dtype=np.float32)
        whine_phase_end = whine_phase + whine_inc * frames
        whine_phase_end = math.fmod(whine_phase_end, 2.0 * math.pi)
        whine_phase = whine_phase_end
        whine[:] = np.sin(whine_phases).astype(np.float32)
        whine *= WHINE_GAIN * (0.2 + 0.8 * rpm_ratio)

    # ================= Mix final =================
    base_engine = tone * (0.4 + 0.6 * pulses) + noise_layer + whine
    wave = base_engine * envelope + pops

    # Petit jitter pour "vivre" un peu
    jitter_amount = 0.003 + 0.01 * rpm_ratio
    jitter = 1.0 + jitter_amount * np.random.randn(frames).astype(np.float32)
    wave *= jitter

    wave = np.tanh(1.8 * wave)

    base_volume = 0.30
    volume = base_volume + 0.40 * rpm_ratio
    volume = clamp(volume, 0.15, 1.0)

    return (wave * volume).astype(np.float32)


def audio_callback(outdata, frames, time_info, status):
    """
    Callback PortAudio : génère le son du moteur en continu.
    """
    global engine_rpm, throttle, running
    global IDLE_RPM, MAX_RPM, LIMIT_RPM, _last_audio_log

    if status:
        print("Status PortAudio:", status)

    target_rpm = IDLE_RPM + throttle * (MAX_RPM - IDLE_RPM)
    target_rpm = clamp(target_rpm, IDLE_RPM, MAX_RPM)

    dt = frames / float(SAMPLE_RATE)
    alpha = math.exp(-dt / RPM_TIME_CONSTANT)

    prev_rpm = engine_rpm
    engine_rpm = alpha * engine_rpm + (1.0 - alpha) * target_rpm
    engine_rpm = clamp(engine_rpm, IDLE_RPM, MAX_RPM)

    decel_amount = max(0.0, prev_rpm - engine_rpm) / 1200.0
    decel_amount = clamp(decel_amount, 0.0, 1.0)

    samples = engine_sample(engine_rpm, frames, decel_amount=decel_amount)

    # Rupteur
    if engine_rpm >= LIMIT_RPM and throttle > 0.5:
        overshoot = clamp(
            (engine_rpm - LIMIT_RPM) / max(1.0, (MAX_RPM - LIMIT_RPM)), 0.0, 1.0
        )
        cut_prob = 0.3 + 0.5 * overshoot
        mask = (np.random.rand(frames) > cut_prob).astype(np.float32)
        samples *= mask

    outdata[:, 0] = samples

    now = time.time()
    if now - _last_audio_log > 1.0:
        print(f"[AUDIO] rpm={engine_rpm:.0f} throttle={throttle:.2f}")
        _last_audio_log = now


def choose_output_device():
    """
    Choisit le périphérique de sortie pour sounddevice.

    - Si AUDIO_DEVICE est défini (index ou nom), on l'utilise.
    - Sinon, on retourne None => sounddevice utilisera le périphérique
      de sortie par défaut, normalement le même que pygame.
    """
    audio_device_env = os.environ.get("AUDIO_DEVICE", None)

    if audio_device_env is None:
        print("AUDIO_DEVICE non défini -> utilisation du périphérique audio par défaut (comme pygame).")
        return None

    # Si AUDIO_DEVICE est défini, on tente de l'utiliser
    try:
        dev_param = int(audio_device_env)
    except ValueError:
        dev_param = audio_device_env

    try:
        info = sd.query_devices(dev_param)
        if info["max_output_channels"] > 0:
            print(f"Utilisation du device AUDIO_DEVICE={dev_param} : {info['name']}")
            return dev_param
        else:
            print(f"AUDIO_DEVICE={dev_param} n'a pas de sortie (max_output_channels=0) -> ignore.")
            return None
    except Exception as e:
        print(f"AUDIO_DEVICE={dev_param} invalide ({e}) -> utilisation du périphérique par défaut.")
        return None


def start_engine_audio():
    """
    Crée et démarre le flux audio du moteur.
    Utilise le périphérique par défaut (comme pygame),
    sauf si AUDIO_DEVICE est défini.
    """
    global _audio_stream

    try:
        print("Périphérique audio par défaut sounddevice:", sd.default.device)
    except Exception as e:
        print("Erreur en listant les devices audio:", e)

    device = choose_output_device()
    print(f"Device audio utilisé dans OutputStream : {device}")

    _audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        device=device,   # None => device par défaut
        blocksize=2048,
        latency='high'
    )
    _audio_stream.start()
    print("Flux audio moteur démarré.")
    return device
