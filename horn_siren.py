#!/usr/bin/env python3
import os

import pygame
from evdev import ecodes

# Dossiers des sons (basés sur l'emplacement du script qui importe ce module)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HORN_DIR = os.path.join(BASE_DIR, "horn")
SIREN_DIR = os.path.join(BASE_DIR, "siren")

# Codes des boutons (d'après ton dump evtest)
BTN_HORN          = ecodes.BTN_THUMB2   # 290  -> klaxon
BTN_SIREN         = ecodes.BTN_TOP      # 291  -> sirène
BTN_NEXT_HORN     = ecodes.BTN_TRIGGER  # 288  -> changer klaxon
BTN_NEXT_SIREN    = ecodes.BTN_THUMB    # 289  -> changer sirène

# Canal audio dédié pour pouvoir jouer klaxon + sirène en même temps
HORN_CHANNEL_ID = 0
SIREN_CHANNEL_ID = 1


class HornSirenManager:
    """
    Gère le chargement et la lecture des sons de klaxon et de sirène via pygame.
    """

    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)  # marge au cas où
        self.horn_channel = pygame.mixer.Channel(HORN_CHANNEL_ID)
        self.siren_channel = pygame.mixer.Channel(SIREN_CHANNEL_ID)

        self.horn_names, self.horn_sounds = self._load_sounds(HORN_DIR)
        self.siren_names, self.siren_sounds = self._load_sounds(SIREN_DIR)

        self.horn_index = 0
        self.siren_index = 0

        self._print_available_sounds()

    def _load_sounds(self, directory):
        """
        Charge tous les fichiers .mp3 d'un dossier comme objets pygame.mixer.Sound.
        Retourne (liste_noms, liste_sounds).
        """
        if not os.path.isdir(directory):
            raise RuntimeError(f"Dossier introuvable : {directory}")

        files = [f for f in os.listdir(directory) if f.lower().endswith(".mp3")]
        files.sort()

        if not files:
            raise RuntimeError(f"Aucun fichier .mp3 trouvé dans {directory}")

        sounds = []
        for f in files:
            full_path = os.path.join(directory, f)
            print(f"[HornSiren] Chargement du son : {full_path}")
            sounds.append(pygame.mixer.Sound(full_path))

        return files, sounds

    def _print_available_sounds(self):
        print("=== SONS DISPONIBLES ===")
        print("Klaxons (horn) :")
        for i, name in enumerate(self.horn_names):
            prefix = "-> " if i == self.horn_index else "   "
            print(f"{prefix}{i}: {name}")

        print("Sirènes (siren) :")
        for i, name in enumerate(self.siren_names):
            prefix = "-> " if i == self.siren_index else "   "
            print(f"{prefix}{i}: {name}")
        print("========================")

    def handle_button(self, code, value):
        """
        À appeler depuis la boucle evdev.
        'value == 1' correspond à un appui de bouton.
        """
        if value != 1:
            return

        if code == BTN_HORN:
            sound = self.horn_sounds[self.horn_index]
            self.horn_channel.play(sound)
            print(f"[KLA XON] Lecture : {self.horn_names[self.horn_index]}")

        elif code == BTN_SIREN:
            sound = self.siren_sounds[self.siren_index]
            self.siren_channel.play(sound)
            print(f"[SIRENE] Lecture : {self.siren_names[self.siren_index]}")

        elif code == BTN_NEXT_HORN:
            self.horn_index = (self.horn_index + 1) % len(self.horn_sounds)
            print(f"[CHANGEMENT KLA XON] Nouveau : {self.horn_names[self.horn_index]}")

        elif code == BTN_NEXT_SIREN:
            self.siren_index = (self.siren_index + 1) % len(self.siren_sounds)
            print(f"[CHANGEMENT SIRENE] Nouvelle : {self.siren_names[self.siren_index]}")

    def shutdown(self):
        """
        À appeler pour fermer proprement le mixer pygame.
        """
        try:
            pygame.mixer.quit()
        except Exception:
            pass
