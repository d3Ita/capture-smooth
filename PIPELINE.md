# Pipeline Capture Smooth : carte de capture et PS Remote Play, x2 / x3 / x4

> **Message à coller dans Claude Code :**
>
> Lis `PIPELINE.md` en entier, puis lis `smooth.py`, `engines.py`, `rife.py`, `post.py`, `present.py`,
> `window_source.py` et `gl_interop.py` avant de modifier quoi que ce soit. Implémente les jalons dans l'ordre,
> une branche et une pull request par jalon. Ne commence un jalon qu'une fois le précédent fusionné. Si une
> hypothèse de ce document est fausse dans le code réel (signature d'une fonction, comportement d'OpenCV,
> API de `nvidia-vfx`), vérifie, dis-le dans la PR et adapte au lieu de forcer. Garde les textes affichés en
> français, ne casse pas le moteur DirectML (AMD/Intel) et ne copie aucun code de Veyra (GPL), le projet est MIT.

## 1. But

GTA 6 tourne à 30 fps sur PS5. Capture Smooth doit afficher le jeu en plein écran sans bordure sur le PC :

- **Qualité** : les vraies images sont au moins aussi bonnes que la sortie de la console (carte de capture) ou
  que le flux Remote Play, et plus propres dans le cas de Remote Play.
- **Fluidité** : 30 fps → 60 (x2), 90 (x3) ou 120 (x4).
- **Latence** : la plus faible possible, et surtout **régulière**, parce qu'un retard constant se sent bien moins
  qu'un retard qui varie.

## 2. Point de départ

La branche `rendu-ameliore` (fichier `rendu-ameliore.patch`, à appliquer avec `git am` s'il ne l'est pas déjà) :

- `present.py` : shader de mise à l'échelle finale Catmull-Rom (anti-halo en agrandissement, noyau élargi en
  réduction) et matrice de couleurs BT.601 → BT.709. Touches F et C.
- `CaptureSource` essaie YUY2, puis NV12, puis MJPG, puis le format par défaut, et garde le premier qui tient
  la taille et les fps demandés. Avec le backend DirectShow d'OpenCV, le FOURCC doit être réglé **après** la
  taille et les fps (chaque changement de taille ou de fps remet la carte dans son format par défaut,
  voir `modules/videoio/src/cap_dshow.cpp`). Ne change pas cet ordre.

Vérifie-la sur Windows avant le jalon 1 (`Tester sans console.cmd`, puis avec une console).

## 3. Les deux pipelines

```
CARTE DE CAPTURE                                   PS REMOTE PLAY
PS5 HDMI 60 Hz, SDR, VRR coupé                     PS5 → réseau → appli PS Remote Play (1080p, fréquence « Élevée »)
   │ YUY2/NV12 non compressé                          │ H.264/H.265 compressé, cadence irrégulière (réseau)
   ▼                                                  ▼
Capture (OpenCV DSHOW)                             Capture de la fenêtre (Windows Graphics Capture)
   │                                                  │ recadrage de la zone vidéo, SANS cv2.resize sur le CPU
   ▼                                                  ▼
Envoi au GPU tout de suite ──────────────┬──────── Envoi au GPU tout de suite
YUV → RGB BT.709 sur le GPU              │         (déjà en RGB)
                                         ▼
                      Détection des doublons (30 vraies images/s)
                         carte : seuil strict / Remote Play : seuil tolérant
                                         ▼
                      [Remote Play seulement] nettoyage de la compression
                         sur les vraies images, à la résolution source (30 fois/s)
                                         ▼
                      Interpolation x2 / x3 / x4 : VFG (RTX) ou RIFE (autres)
                         à la résolution de travail
                                         ▼
                      Agrandissement vers l'écran : shader Catmull-Rom
                         (RTX VSR en option si le budget GPU le permet, voir §6)
                                         ▼
                      Affichage sans bordure, cadencé, VRR si disponible
```

Le cœur (doublons → interpolation → affichage) est commun. Seules l'entrée et le nettoyage diffèrent.

## 4. Multiplicateurs x2 / x3 / x4

### Latence imposée par l'interpolation

Pour créer les images entre A et B, il faut avoir B. Avec un espacement régulier, B s'affiche
`(k-1)/k × période` après son arrivée (période = 33,3 ms à 30 fps) :

| Mode | Sortie (jeu 30 fps) | Retard de la vraie image (hors calcul) |
|---|---|---|
| x2 | 60 fps | 16,7 ms |
| x3 | 90 fps | 22,2 ms |
| x4 | 120 fps | 25,0 ms |

Plus de multiplicateur veut dire plus de fluidité, mais un peu plus de retard. Dans l'interface, présente x2
comme « Réactif » et x4 comme « Fluide ».

### Compatibilité avec l'écran

- Sortie ≤ fréquence de l'écran, sinon des images sont jetées.
- **Sans VRR**, la sortie doit diviser exactement la fréquence de l'écran. Sinon des saccades apparaissent :
  90 fps sur un écran 144 Hz saccade, sur 180 Hz c'est parfait.
- **Avec VRR** (G-Sync / FreeSync actifs pour la fenêtre), toute sortie jusqu'à la fréquence maximale convient.
- Dans le lanceur, grise les multiplicateurs incompatibles et explique pourquoi en une phrase. Garde le repli
  automatique existant (x3 → x2) quand le GPU ne suit pas.
- Jeu à 60 fps (autres jeux) : x2 = 120 fps. Les mêmes règles s'appliquent avec une période de 16,7 ms.

## 5. Jalons

### Jalon 1 : mesurer (avant d'optimiser)

- Mesure par étape, moyenne sur 1 s : capture → GPU, conversion, doublons, nettoyage, interpolation,
  agrandissement, affichage, retard total entre l'arrivée de la vraie image et son affichage.
- Utilise des événements CUDA (`torch.cuda.Event`) pour le GPU, pas `time.perf_counter` autour d'appels
  asynchrones.
- Affiche ces mesures dans le compteur (F1, deuxième page ou ligne supplémentaire) et dans le journal.
- **Critère** : en mode `--test`, les chiffres s'affichent et leur somme est cohérente avec le retard total.

### Jalon 2 : entrée directe vers le GPU

- **Carte** : OpenCV DSHOW **ne donne pas** l'image brute. `CAP_PROP_CONVERT_RGB = 0` ne vaut que pour Y16.
  YUY2 passe par le convertisseur RGB24 de DirectShow, et NV12 est converti par OpenCV sur le CPU avec
  `COLOR_YUV2BGR_NV12` (BT.601, plage limitée, vérifié dans le code source). Pour envoyer du YUV brut au GPU,
  il faut donc une autre API de capture, par exemple PyAV avec l'entrée `dshow`
  (`pixel_format=nv12` ou `yuyv422`) ou Media Foundation. Fais-en une source optionnelle (`--capture brute`)
  et garde la source OpenCV actuelle comme repli. Ensuite : copie vers le GPU (mémoire « pinned » +
  `non_blocking=True`), conversion YUV → RGB BT.709 dans un noyau PyTorch, plage configurable
  (`--plage auto|limitee|pleine`). La correction 601 → 709 du shader doit alors être désactivée pour cette
  source, puisque la conversion est déjà juste.
- **Remote Play** : supprime le `cv2.resize` CPU de `WindowSource`. Envoie la zone recadrée au GPU et laisse
  l'agrandissement au shader final. Copie le tampon pendant le rappel (il n'est valable que pendant l'appel).
- **Critère** : temps capture → GPU mesuré au jalon 1 en baisse, aucune régression des couleurs (comparaison
  avant/après sur une capture d'écran de la même scène).

### Jalon 3 : doublons et cadence robustes

- **Carte** : seuil strict (les doublons sont presque identiques au bit près).
- **Remote Play** : la compression rend les doublons légèrement différents. Utilise un seuil plus tolérant,
  calculé sur une vignette en luminance, et la cadence attendue (une vraie image toutes les ~33 ms) pour lever
  les cas ambigus.
- Période estimée à partir de l'horodatage des vraies images (moyenne glissante), pour absorber la gigue du
  réseau sans fausser l'espacement des images générées.
- **Critère** : sur un enregistrement Remote Play de GTA 6, environ 30 vraies images/s détectées, sans faux
  doublons visibles (saccade) ni faux changements de plan.

### Jalon 4 : NVIDIA Video Frame Generation (`nvidia-vfx`)

- Nouveau moteur, à côté de RIFE, avec la même interface que les moteurs existants dans `engines.py`. Lis
  cette interface d'abord.
- `pip install nvidia-vfx` (version 0.2.0.0 au moment de la rédaction). Le paquet sur PyPI n'est qu'un
  amorçage : les vrais binaires viennent de `https://pypi.nvidia.com/`, vérifie que PyInstaller les embarque.
  Modèles VFG : Low, Medium, High. Multiplicateur : 2 à 8, ou position explicite dans ]0, 1[. L'API est en
  `from nvvfx import VideoFrameGeneration`. Les images sont des tenseurs CUDA `(3, H, W)` float32 dans
  [0, 1], échangés sans copie via DLPack. Pour un multiplicateur N, appeler `run()` avec les index 1 à N-1,
  ou `run_at_timestep(prev, cur, t)`. Changements de plan : `shot_change=True`, ou la détection automatique
  à activer avant `load()`. **Toujours `.clone()` la sortie avant l'appel suivant.** Vérifie les noms exacts
  (modes, attributs) dans le paquet installé.
- Sélection : `--moteur auto` choisit VFG sur RTX si le paquet se charge, sinon CUDA/RIFE, sinon DirectML.
  Repli silencieux journalisé en cas d'échec.
- Garde la protection du HUD et la netteté de `post.py` après l'interpolation.
- **Licence** : `nvidia-vfx` est sous licence propriétaire NVIDIA. Avant de l'inclure dans l'installateur
  PyInstaller, vérifie les droits de redistribution. Sinon, propose un téléchargement au premier lancement.
  Mentionne-le dans `THIRD_PARTY.md`.
- **Critère** : x2/x3/x4 fonctionnent. Compare le coût de l'IA et la qualité avec RIFE sur la même séquence,
  et mets les chiffres dans la PR.

### Jalon 5 : nettoyage Remote Play et RTX VSR

- Remote Play, sur les vraies images seulement (30 fois/s), à la résolution source, avant l'interpolation :
  `VideoSuperRes` en mode même taille `DENOISE_*` ou `DEBLUR_*`. Le choix par défaut se fait sur des captures
  réelles. Une image propre donne aussi une meilleure interpolation.
- Agrandissement par IA (`VideoSuperRes`, familles `STREAMING_*` pour Remote Play, `HIGHBITRATE_*` pour la
  carte) seulement si le budget du §6 le permet. Sinon le shader Catmull-Rom reste la règle.
- Option dans le lanceur : « Nettoyage de l'image (RTX) » activé par défaut en Remote Play sur RTX.
- **Critère** : comparaison avant/après sur la même scène (captures d'écran), coût mesuré.

### Jalon 6 : affichage et son

- **Affichage** : pas de V-Sync par défaut, cadence par `sleep_until` (existant), VRR quand il est
  disponible. Ajoute au README et au lanceur le réglage NVIDIA « Méthode de présentation Vulkan/OpenGL →
  Préférer superposé sur la chaîne d'échange DXGI » (pilotes 526.47 et suivants). Il fait présenter la
  fenêtre OpenGL par une chaîne d'échange DXGI, ce qui réduit la latence d'affichage et autorise le VRR en
  fenêtre. Quelques utilisateurs signalent des problèmes de cadence avec ce réglage selon les applis :
  mesure avec et sans au jalon 1 avant de le recommander.
- **Son, carte** : le délai audio suit automatiquement le retard vidéo mesuré (jalon 1), avec un réglage
  manuel en plus.
- **Son, Remote Play** : aujourd'hui l'appli Remote Play joue le son sans retard, alors que l'image arrive
  ~20 à 40 ms plus tard. Étudie la capture audio du processus Remote Play (WASAPI process loopback,
  `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`) avec le même délai, en coupant la sortie de l'appli.
  **Minimum : Windows 10 build 20348**, donc Windows 11 en pratique : les Windows 10 grand public (19045)
  ne l'ont pas. Sur ces systèmes, garde le son actuel et affiche la limite. Si c'est trop fragile, documente
  la limite au lieu de bricoler.
- **Critère** : son et image synchronisés à l'oreille sur une scène de dialogue, dans les deux modes.

## 6. Budget GPU par image de sortie

Temps disponible par image : 16,7 ms (x2), 11,1 ms (x3), 8,3 ms (x4). Le moteur doit rester sous ~70 % de ce
budget, mesuré au jalon 1, sinon des saccades apparaissent.

- Ce qui tourne 30 fois/s (conversion, doublons, nettoyage des vraies images) coûte peu.
- Ce qui tourne à chaque image de sortie (images générées, agrandissement IA) est multiplié par k. C'est
  pourquoi RTX VSR n'est raisonnable qu'en x2, ou sur un petit agrandissement.
- Si le budget est dépassé, réduis dans cet ordre : agrandissement IA → shader ; nettoyage → plus léger ;
  multiplicateur → repli automatique.

## 7. Réglages console et PC à documenter (README + lanceur)

- **PS5 (carte)** : sortie 60 Hz (en 120 Hz chaque image est répétée 4 fois pour rien), VRR coupé, HDR coupé
  au départ, résolution = celle de l'écran PC si la carte la passe sans compression.
- **PS Remote Play** : dans Paramètres → Qualité vidéo, fréquence d'images « Élevée ». Sur PS5, la
  résolution n'est pas toujours proposée : l'appli choisit jusqu'à 1080p selon la qualité du réseau. PC et
  console en Ethernet de préférence (Sony recommande au moins 15 Mbit/s).
- **Manette** : branchée à la PS5 (carte) ou au PC (Remote Play).

## 8. Hors périmètre pour l'instant

- **DLSS 5 Neural Rendering (les « NR » de Veyra)** : Veyra charge `nvngx_dlssnr.dll`, extrait de NBA 2K27,
  ou des versions modifiées par la communauté (ShortFuse, Lecram, portages AMD via ZLUDA), qu'il ne met pas
  dans son dépôt. C'est ce qui « repeint » son image, et c'est en grande partie pour ça qu'elle paraît plus
  belle. Ne l'intègre pas : redistribuer un fichier NVIDIA extrait d'un jeu ou modifié pose un problème de
  licence, ça change le style du jeu (contraire au but de fidélité), et c'est un gros modèle, donc plus de
  latence.

- **Décoder directement le flux Remote Play (chiaki-ng)** : plus net et moins de latence que la capture de
  fenêtre, mais chiaki-ng est sous AGPL. L'intégrer obligerait sans doute à changer la licence du projet.
  Décision à prendre par le propriétaire du dépôt avant toute ligne de code.
- **FSR 3/4, XeSS ou DLSS temporels** : il faut des vecteurs de mouvement, donc du flux optique, et du
  C++/DirectX 12. Pas avant que tout le reste soit stable.
- **HDR (P010)**.

## 9. Ce que l'analyse de Veyra (Likely7/Veyra-NRVideo, GPL-3.0) a appris

S'en inspirer, ne rien copier (licence GPL, projet MIT) :
- Mise à l'échelle finale en Catmull-Rom au lieu du bilinéaire (déjà fait, `present.py`).
- Format de pixel imposé et matrice BT.709 explicite à la capture (fait en partie, jalon 2 pour le reste).
- RTX Video SR via le SDK NVIDIA Video Effects, trois familles (standard, high bitrate, streaming) ;
  interpolation NVIDIA VFG à la place de RIFE sur RTX (mesure de Veyra en 4K : environ 6,6 ms contre
  32 ms pour RIFE, qualité un peu meilleure).
- FSR 3.1/4 et XeSS temporels alimentés par du flux optique avec une profondeur à zéro (hors périmètre, §8).
- Tout sur le GPU sans copie (C++/DirectX 12) : c'est son avantage de latence ; le jalon 2 réduit l'écart.

## 10. Règles pour chaque PR

- Une PR par jalon, avec les mesures avant/après du jalon 1 et les limites connues.
- Les options gardent l'ancien comportement accessible (touche ou option) pour comparer.
- Mise à jour du README (FR et EN) quand une touche, une option ou un réglage change.
- Aucune régression du moteur DirectML ni du mode `--test`.
