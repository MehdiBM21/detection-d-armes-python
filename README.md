# detection-d-armes-python
![Capture](https://github.com/MehdiBM21/detection-d-armes-python/assets/126264088/d950d142-d9f7-42c1-86ff-504a451c7e3f)
![Capture2](https://github.com/MehdiBM21/detection-d-armes-python/assets/126264088/40c9091a-fab0-4d58-aa14-31faf3559ad5)


## Etapes à suivre:

1. Créez un environnement virtuelle et un kernel pour Jupyter.
```bash
  python -m venv test
```
2. L'activer
```bash
  .\test\Scripts\activate
```
3. Installer des dépendances et ajouter un environnement virtuel au Kernel Python
```bash
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=test
```
4. Commencez par éxecuter l'étape 0 et 1 de du notebook Preparation_et_entrainement.(Paths et download TFOD).

5. executer le code_finale.py (assurez vous que le chemin vers le fichier chkpt-7 et le fichier pipeline est bien valide. (ligne 35 et ligne 40)).

6. aller vers le lien locale donné par flask.
