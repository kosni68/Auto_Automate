# Auto_Automate

## Préparer l'environnement
1) Prérequis systèmes (Debian/Ubuntu) :
   - `sudo apt-get install python3-venv python3-dev build-essential libportaudio2 libasound2-dev libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev libsdl2-ttf-dev libfreetype6-dev libjpeg-dev zlib1g-dev`
   - Branche la manette THRUSTMASTER si tu utilises le potentiomètre.
2) Crée un virtualenv et installe les dépendances Python :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3) Lance le simulateur :
   ```bash
   python main.py
   ```

## Variables d'environnement utiles
- `POT_DEVICE` : chemin vers le joystick (défaut `/dev/input/by-id/usb-THRUSTMASTER_360_Modena_Pro-event-joystick`)
- `POT_AXIS` : axe utilisé (défaut `ABS_Y`)
- `POT_INVERT` : `1` pour inverser la course des gaz
- `POT_DEADZONE` : zone morte (0..1, défaut `0.05`)
- `POT_CURVE` : exponentielle appliquée à la course (défaut `1.3`)

## Structure rapide
- `main.py` : boucle principale, lecture clavier + potentiomètre
- `engine_audio.py` : synthèse du son moteur (numpy + sounddevice)
- `horn_siren.py` : gestion des sons mp3 de klaxons/sirènes (pygame) stockés dans `horn/` et `siren/`
