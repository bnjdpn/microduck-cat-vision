# Microduck — reconnaissance de Nooby et Jaina

Ce projet conserve les photos originales dans `Nooby/`, `Jaina/` et `Distributeurs/`, entraîne localement un classifieur léger et branche sa prédiction sur le simulateur officiel Microduck.

## Préparer et entraîner

```bash
uv sync
uv run microduck-prepare
uv run microduck-train
```

La préparation détecte les chats, refuse les photos ambiguës contenant plusieurs chats, exclut les doublons contradictoires, recadre le chat et sépare les sessions de prise de vue entre entraînement, validation et test. Au runtime, un détecteur YOLO11s plus robuste analyse à la fois le cadre complet et cinq zones qui se chevauchent, puis fusionne les doublons : les deux chats peuvent ainsi être identifiés séparément dans la même image, y compris si l'un est petit ou partiellement masqué.

Les variantes photoréalistes validées sous `dataset/synthetic/` sont ajoutées uniquement au jeu d'entraînement. Les jeux de validation et de test restent constitués exclusivement de photos réelles.

Les rapports se trouvent dans `dataset/reports/`. Les modèles finaux et métriques se trouvent dans `artifacts/model/`.

## Construire et démarrer le simulateur

```bash
cd simulator/app
npm install
npm run build
cd ../..
uv run microduck-serve
```

Ouvrir ensuite <http://127.0.0.1:8000>. Le panneau « Cat vision » accepte une photo ou la webcam. À la première détection, le robot prononce « Nooby » ou « Jaina » avec une voix française générée localement. Tant que Nooby reste visible dans le flux webcam, il répète un petit saut toutes les secondes ; le premier résultat sans Nooby arrête immédiatement la boucle. Un garde-fou l'arrête aussi après trois secondes sans nouveau résultat vidéo. Jaina ne déclenche aucun mouvement. Une photo reste une observation ponctuelle : Nooby produit alors un seul saut. Le sélecteur « Caméra » permet de choisir explicitement la webcam ; une caméra physique est proposée avant les sources virtuelles comme VCam.

La webcam doit cadrer les deux Xiaomi Smart Pet Food Feeder de première génération, avec la Petlibro Dockstream 2 au centre. Le détecteur affiche leur zone de repas avec l'affectation `Nooby · gauche` et `Jaina · droite`. L'arène 3D représente aussi cette installation : deux distributeurs blancs, la fontaine centrale, les deux plaques d'affectation et deux points de repas stables. Lorsqu'une inversion est confirmée, une silhouette stylisée du chat vu de dos apparaît devant la mauvaise gamelle avec un halo d'alerte.

Il faut deux analyses consécutives montrant l'avant de la silhouette du chat dans la mauvaise zone avant de déclencher l'alerte : Jaina à gauche ou Nooby à droite fait prononcer une seconde fois son nom et donne au Microduck une commande de course vers les coordonnées 3D de cette gamelle. La course continue pendant que l'inversion reste confirmée, ralentit pour tourner, s'arrête à 20 cm de la cible et s'annule au premier résultat qui ne montre plus l'inversion. Les zones détectées sont gardées huit secondes pour résister à l'occultation du bol par le chat. Le cadrage étant prévu depuis l'arrière des chats, seules la tête et les épaules (les 58 % supérieurs de leur boîte de détection) servent à décider qu'ils mangent ; la queue ou la croupe qui passe devant un bol ne suffit pas.

Le pipeline Stable Audio 3 Small SFX trouvé dans Échappée est conçu pour les ambiances et effets sonores, pas pour prononcer fidèlement un nom. Les deux annonces WAV sont donc produites hors runtime avec la voix française locale `Thomas` de macOS (`Noubi` et `Jaïna` pour guider la prononciation), puis embarquées dans `simulator/app/public/assets/voices/cats/`. Les photos sources et les images issues de la webcam restent sur le Mac.

## Limites connues

- Le seuil « Inconnu » rejette les prédictions incertaines, mais il n'est pas encore calibré sur un véritable jeu de photos d'autres chats.
- Le simulateur local reçoit la webcam du Mac. Le branchement à la caméra physique du Microduck dépendra de la surface WebRTC/SDK disponible sur le robot.
- La scène 3D suppose que la webcam cadre l'installation fixe représentée dans l'arène. Les coordonnées des deux gamelles sont donc connues dans le simulateur ; cela ne constitue pas encore une localisation métrique ni une navigation physique sur le vrai robot.
- Les poids pré-entraînés sont téléchargés au premier lancement ; les photos restent locales.
