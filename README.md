# Auto_Automate

## Préparer l'environnement
1) Version Python recommandée : 3.12 (pygame a une roue binaire pour 3.12 sur ARM, pas pour 3.13).
   - Si `python3.12` est dans apt :
     ```bash
     sudo apt-get update
     sudo apt-get install python3.12 python3.12-venv
     python3.12 -m venv .venv  # le service systemd pointe sur ce venv
     source .venv/bin/activate
     ```
   - Sinon, installation via pyenv (confirmé fonctionnel) :
     ```bash
     sudo apt-get install -y build-essential curl git zlib1g-dev libssl-dev \
       libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
       libffi-dev liblzma-dev
     curl https://pyenv.run | bash
     # Ajoute dans ~/.bashrc :
     # export PATH="$HOME/.pyenv/bin:$PATH"
     # eval "$(pyenv init -)"
       # eval "$(pyenv virtualenv-init -)"
       exec $SHELL  # recharge le shell

      pyenv install 3.12.7
      pyenv virtualenv 3.12.7 auto-automate
      pyenv local auto-automate  # dans le repo
      pip install --upgrade pip
      pip install --prefer-binary -r requirements.txt
      ```

2) Prérequis systèmes (Debian/Ubuntu) :
   - `sudo apt-get install python3-venv python3-dev build-essential libportaudio2 libasound2-dev libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev libsdl2-ttf-dev libfreetype6-dev libjpeg-dev zlib1g-dev`
   - Branche la manette THRUSTMASTER si tu utilises le potentiomètre.
3) Crée un virtualenv (en Python 3.12) et installe les dépendances Python :
   ```bash
   # si tu restes en Python 3.13, pygame officiel n'a pas de roue : on installe pygame-ce via pip
   pip install --upgrade pip
   pip install --prefer-binary -r requirements.txt
   ```
4) Lance le simulateur :
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

## Démarrage automatique (systemd)
1) Le service systemd utilise `.venv/bin/python` et ajoute `/usr/lib/python3/dist-packages` au `PYTHONPATH` pour récupérer `pygame` installé au niveau système. Ajuste ces chemins si besoin.
2) Copie le service :
   ```bash
   sudo cp systemd/auto-automate.service /etc/systemd/system/auto-automate.service
   sudo systemctl daemon-reload
   sudo systemctl enable auto-automate.service
   sudo systemctl start auto-automate.service
   ```
3) Journal des logs :
   ```bash
   sudo journalctl -u auto-automate.service -f
   ```
4) Ajuste `User=`, `WorkingDirectory=` et `ExecStart=` dans `/etc/systemd/system/auto-automate.service` si ton utilisateur ou ton chemin Python diffèrent.
