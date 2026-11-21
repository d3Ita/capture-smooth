# Capture Smooth : notes pour Claude Code

- Feuille de route : `PIPELINE.md`. Lis-la avant tout travail sur la capture, l'interpolation,
  l'agrandissement, l'affichage ou le son. Les jalons se font dans l'ordre, une branche et une PR par jalon.
- Projet sous licence MIT : ne copie aucun code de Veyra (GPL-3.0) ni de chiaki-ng (AGPL) sans accord
  explicite du propriétaire du dépôt.
- Textes affichés à l'utilisateur en français. README en français et en anglais.
- Ne casse jamais le moteur DirectML (AMD/Intel) ni le mode `--test`.
- `docs/` contient le site public (`index.html`) : n'y mets pas de notes internes.
- Avec OpenCV DirectShow, le FOURCC se règle après la taille et les fps (voir `CaptureSource.open_format`).
- Tests sans console : `Tester sans console.cmd`, ou `python smooth.py --test`.
