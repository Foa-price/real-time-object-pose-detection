# MediaPipe Multi-Detection (mains, visage, pose, objets)

Démo temps réel avec la webcam, basée sur **MediaPipe Tasks** et **OpenCV** :

- **Mains** (jusqu'à 2) : landmarks + reconnaissance de gestes simples (Poing, Ouvert, OK, Peace, Rock, Pouce en l'air, Pointer…)
- **Visage** : maillage filaire de style HUD
- **Pose** : squelette corporel (modèle `pose_landmarker_lite`)
- **Objets** : détection via EfficientDet-Lite2
- Affichage du FPS et mode debug

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python mediapipe_detection.py
```

Les modèles (`.task` / `.tflite`) sont téléchargés automatiquement au premier lancement.
Appuyez sur **q** pour quitter.

## Options

Dans le script :

- `USE_THREADING` : exécute les détections dans un thread (mettre `False` en cas de blocage)
- `DEBUG_PRINT` : affiche des logs détaillés (le désactiver améliore les performances)
- `CONFIG` : seuils de confiance, nombre de mains/visages/poses, taille de l'image, etc.

## Prérequis

- Python 3.9+
- Une webcam
