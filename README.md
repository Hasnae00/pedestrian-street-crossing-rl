
# Pedestrian Street Crossing – REINFORCE

 RL-Projekt, in dem ein Fußgänger lernen soll, eine zweispurige Straße sicher zu überqueren. Die Umgebung ist ein eigener Gym-ähnlicher Env mit Poisson-Verkehr, trainiert wird mit einem Monte‑Carlo Policy‑Gradient (REINFORCE).

## Struktur

- `env.py` – Straßenumgebung (Traffic, Reward, State‑Definition)
- `reinforce.py` – Policy‑Netz und REINFORCE‑Update
- `train.py` – Training, Logging und Auswertung

## Voraussetzungen

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

Installation (im Projektordner):

```bash
pip install torch numpy matplotlib
```

## Training starten

```bash
python train.py
```

Während des Trainings werden:

- Kennzahlen im Terminal gedruckt (Return, Success/Collision/Timeout, GO/WAIT).
- Plotbilder als `training_progress.png` im Projektordner gespeichert.

## Kurz zu Environment & Algorithmus

- **Environment:** Fußgänger mit zwei Entscheidungsphasen (Bordstein, Mittelinsel), Autos mit Poisson‑Ankünften und normalverteilten Geschwindigkeiten, dichte Abstände werden physikalisch verhindert.
- **Reward:** starker negativer Reward bei Kollision, moderater Reward für Median und erfolgreiches Überqueren, kleine Zeitstrafe pro Schritt, Timeout‑Strafe.
- **Algorithmus:** REINFORCE mit:
  
- Monte‑Carlo Returns
- Baseline (Return‑Mittelwert) zur Varianzreduktion
- Entropie‑Bonus (Exploration) mit linearer Decay‑Schedule
- Gradient Clipping

Damit solltest du das Projekt schnell verstehen und direkt mit `train.py` loslegen können.
