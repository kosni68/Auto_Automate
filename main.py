#!/usr/bin/env python3
import sys
import time
import threading
import termios
import tty
import os

# ==========================
# evdev (potentiomètre + boutons)
# ==========================
try:
    from evdev import InputDevice, ecodes
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False
    ecodes = None
    InputDevice = None

import engine_audio as engine
from horn_siren import HornSirenManager, BTN_HORN, BTN_SIREN, BTN_NEXT_HORN, BTN_NEXT_SIREN

# ==========================
# Configuration joystick / potentiomètre
# ==========================

POT_DEVICE = os.environ.get(
    "POT_DEVICE",
    "/dev/input/by-id/usb-THRUSTMASTER_360_Modena_Pro-event-joystick"
)

POT_AXIS_NAME = os.environ.get("POT_AXIS", "ABS_Y")

if EVDEV_AVAILABLE:
    POT_AXIS_CODE = getattr(ecodes, POT_AXIS_NAME, ecodes.ABS_Y)
else:
    POT_AXIS_CODE = None

# Inversion éventuelle (True si 0 = plein gaz et max = relâché)
POT_INVERT = os.environ.get("POT_INVERT", "0").lower() in ("1", "true", "yes")


def keyboard_thread():
    """
    Thread de gestion du clavier : presets moteur, graves, quitter...
    """
    print("Contrôles clavier (en plus de la manette des gaz) :")
    print("  1 / 2 / 3 : presets (4c route, V8, moto)")
    print("  4 / 6 / 8 : régler manuellement le nombre de cylindres")
    print("  g / h    : plus / moins de graves")
    print("  q        : quitter")
    print("-----------------------------------------")

    try:
        while engine.running:
            ch = sys.stdin.read(1)

            if ch == 'q':
                print("Quit demandé depuis le clavier.")
                engine.stop_engine()
                break
            elif ch in ('1', '2', '3'):
                engine.apply_preset(ch)
            elif ch in ('4', '6', '8'):
                engine.NUM_CYLINDERS = int(ch)
                print(f"-> Cylindres réglés manuellement : {engine.NUM_CYLINDERS}")
            elif ch == 'g':
                engine.BASS_BOOST = engine.clamp(engine.BASS_BOOST + 0.1, 0.3, 2.5)
                print(f"-> Graves ++  (BASS_BOOST={engine.BASS_BOOST:.2f})")
            elif ch == 'h':
                engine.BASS_BOOST = engine.clamp(engine.BASS_BOOST - 0.1, 0.3, 2.5)
                print(f"-> Graves --  (BASS_BOOST={engine.BASS_BOOST:.2f})")
    finally:
        pass


def joystick_thread(horn_mgr: HornSirenManager):
    """
    Lit l'axe du joystick (potentiomètre) + les boutons pour klaxon/sirène.
    Met à jour engine.throttle et déclenche horn/siren.
    """
    if not EVDEV_AVAILABLE:
        print("evdev non installé -> manette des gaz + boutons désactivés.")
        return

    try:
        dev = InputDevice(POT_DEVICE)
        print(f"Joystick : utilisation de {POT_DEVICE}")
    except Exception as e:
        print(f"Impossible d'ouvrir {POT_DEVICE}: {e}")
        return

    # Lecture des bornes de l'axe
    try:
        absinfo = dev.absinfo(POT_AXIS_CODE)
        axis_min = absinfo.min
        axis_max = absinfo.max
        print(f"AXE {POT_AXIS_NAME}: min={axis_min}, max={axis_max}")
    except Exception as e:
        print(f"Impossible de lire absinfo {POT_AXIS_NAME}, fallback 0..255 ({e})")
        axis_min = 0
        axis_max = 255

    if axis_max == axis_min:
        axis_max = axis_min + 1

    last_print_time = 0.0

    # paramètres ressentis (deadzone + courbe)
    DEADZONE = float(os.environ.get("POT_DEADZONE", "0.05"))
    CURVE    = float(os.environ.get("POT_CURVE", "1.3"))

    try:
        for event in dev.read_loop():
            if not engine.running:
                break

            # Axe du potentiomètre -> throttle
            if event.type == ecodes.EV_ABS and event.code == POT_AXIS_CODE:
                val = event.value
                norm = (val - axis_min) / float(axis_max - axis_min)
                norm = engine.clamp(norm, 0.0, 1.0)
                if POT_INVERT:
                    norm = 1.0 - norm

                # deadzone
                if norm < DEADZONE:
                    norm = 0.0
                else:
                    norm = (norm - DEADZONE) / (1.0 - DEADZONE)

                # courbe
                norm = engine.clamp(norm, 0.0, 1.0) ** CURVE

                engine.set_throttle(norm)

                now = time.time()
                if now - last_print_time > 0.05:
                    print(f"[POT] brut={val} norm={norm:.2f} throttle={engine.throttle:.2f}")
                    last_print_time = now

            # Boutons -> klaxon / sirène
            elif event.type == ecodes.EV_KEY:
                horn_mgr.handle_button(event.code, event.value)
    finally:
        try:
            dev.close()
        except Exception:
            pass


def main():
    print("Simulateur complet : moteur + klaxon/sirène (manette THRUSTMASTER)")
    print("===============================================================")

    # 1) Appliquer un preset moteur par défaut
    engine.apply_preset("1")
    print(f"Preset moteur actuel : {engine.current_preset_name}")
    print(f"Ralenti : {engine.IDLE_RPM} RPM, Max : {engine.MAX_RPM} RPM, Rupteur ≈ {engine.LIMIT_RPM} RPM\n")
    print("La position du potentiomètre contrôle directement les gaz (throttle 0..1).")

    # 2) Initialisation des sons klaxon/sirène
    try:
        horn_mgr = HornSirenManager()
    except Exception as e:
        print(f"Erreur lors de l'initialisation des sons klaxon/sirène : {e}")
        horn_mgr = None

    # 3) Passer le terminal en mode cbreak pour lecture clavier non bloquante
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except Exception:
        old_settings = None

    if old_settings is not None:
        tty.setcbreak(fd)

    # 4) Démarrer les threads clavier + joystick
    kb_thread = threading.Thread(target=keyboard_thread, daemon=True)
    kb_thread.start()

    if horn_mgr is not None:
        js_thread = threading.Thread(target=joystick_thread, args=(horn_mgr,), daemon=True)
        js_thread.start()
    else:
        js_thread = None

    # 5) Démarrer le moteur audio
    device = engine.start_engine_audio()

    # 6) Boucle principale : tant que engine.running est True
    try:
        while engine.running:
            engine.poll_audio_status()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nCtrl+C reçu, arrêt...")
        engine.stop_engine()
    finally:
        # Restauration du terminal
        if old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if horn_mgr is not None:
            horn_mgr.shutdown()

        print("Arrêt du simulateur complet.")


if __name__ == "__main__":
    main()
