"""Édition personnelle : catalogue et téléchargement de modèles depuis un dépôt GitHub privé.

L'édition personnelle se reconnaît au fichier edition_personnelle.json placé à la racine de
l'extension par outils/empaqueter.py. L'édition publiée sur le dépôt QGIS n'a pas ce fichier :
ce module y reste inactif.

Le jeton GitHub (lecture seule sur le dépôt des modèles) est rangé dans le gestionnaire
d'authentification de QGIS, dans une configuration « Basic » dont le mot de passe est le jeton.
Il n'est envoyé qu'à api.github.com : à la redirection vers le fichier, QGIS ne recopie pas
l'en-tête d'autorisation.
"""

import hashlib
import json
import os

from qgis.core import QgsApplication, QgsAuthMethodConfig, QgsBlockingNetworkRequest, QgsSettings, QgsTask
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

FICHIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "edition_personnelle.json")
CLE_AUTH = "batiscan/prive_authcfg"


def actif():
    return os.path.isfile(FICHIER)


def _config():
    with open(FICHIER, encoding="utf-8") as f:
        return json.load(f)


def catalogue():
    """Modèles du dépôt privé, au format du catalogue de models.py."""
    if not actif():
        return []
    c = _config()
    return [dict(m, id="prive_" + m["id"], type="segmenteur", prive=True, depot=c["depot"], version=c["version"],
                 licence=m.get("licence", "privé")) for m in c["modeles"]]


def authcfg():
    return QgsSettings().value(CLE_AUTH, "") or ""


def definir_authcfg(identifiant):
    QgsSettings().setValue(CLE_AUTH, identifiant or "")


def _jeton():
    ident = authcfg()
    if not ident:
        raise ValueError("aucun accès au dépôt privé n'est configuré (onglet Modèles)")
    cfg = QgsAuthMethodConfig()
    if not QgsApplication.authManager().loadAuthenticationConfig(ident, cfg, True):
        raise ValueError("configuration d'authentification illisible (mot de passe principal de QGIS ?)")
    jeton = cfg.config("password")
    if not jeton:
        raise ValueError("la configuration choisie ne contient pas de jeton dans le champ mot de passe")
    return jeton


def _requete(url, accept, jeton):
    r = QNetworkRequest(QUrl(url))
    r.setRawHeader(b"Accept", accept)
    if jeton:
        r.setRawHeader(b"Authorization", b"Bearer " + jeton.encode("ascii"))
    r.setRawHeader(b"X-GitHub-Api-Version", b"2022-11-28")
    return r


class TacheTelechargementPrive(QgsTask):
    """Télécharge un modèle du dépôt privé, vérifie son empreinte et le range dans le profil."""

    def __init__(self, modele, dossier):
        super().__init__(f"BatiScan : téléchargement de {modele['nom']}", QgsTask.Flag.CanCancel)
        self.modele = modele
        self.cible = os.path.join(dossier, modele["id"], modele["fichier"])
        self.erreur = None
        self.jeton = _jeton()  # lu dans le fil principal (le mot de passe principal peut être demandé)

    def run(self):
        try:
            m = self.modele
            req = QgsBlockingNetworkRequest()
            url = f"https://api.github.com/repos/{m['depot']}/releases/tags/{m['version']}"
            if req.get(_requete(url, b"application/vnd.github+json", self.jeton)) != QgsBlockingNetworkRequest.ErrorCode.NoError:
                self.erreur = "dépôt privé inaccessible : " + req.errorMessage()
                return False
            fichiers = json.loads(bytes(req.reply().content())).get("assets", [])
            asset = next((a for a in fichiers if a["name"] == m["fichier"]), None)
            if asset is None:
                self.erreur = f"{m['fichier']} absent de la version {m['version']} du dépôt privé"
                return False
            dl = QgsBlockingNetworkRequest()
            dl.downloadProgress.connect(lambda recu, total: self.setProgress(100 * recu / total if total > 0 else 0))
            if dl.get(_requete(asset["url"], b"application/octet-stream", self.jeton)) != QgsBlockingNetworkRequest.ErrorCode.NoError:
                self.erreur = "téléchargement échoué : " + dl.errorMessage()
                return False
            donnees = bytes(dl.reply().content())
            if self.isCanceled():
                return False
            if m.get("sha256") and hashlib.sha256(donnees).hexdigest() != m["sha256"]:
                self.erreur = "le fichier reçu ne correspond pas à l'empreinte attendue"
                return False
            os.makedirs(os.path.dirname(self.cible), exist_ok=True)
            tmp = self.cible + ".part"
            with open(tmp, "wb") as f:
                f.write(donnees)
            os.replace(tmp, self.cible)
            return True
        except Exception as exc:  # message affiché à l'utilisateur
            self.erreur = str(exc)
            return False
