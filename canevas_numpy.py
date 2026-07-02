"""
============================================================
 CANEVAS NUMPY — Les briques de ton futur reseau de neurones
============================================================

REGLE DU JEU
------------
Tu ne fais PAS des exercices au hasard. Chaque fonction que tu codes
ici est un morceau du reseau de neurones que tu assembleras la semaine
prochaine. Rien n'est jete.

Remplace chaque  # TODO  par ton code. Quand tu lances le fichier
(python canevas_numpy.py), il se corrige tout seul :
  - s'il affiche des erreurs -> ce n'est pas fini.
  - s'il affiche "TOUT EST VERT" -> tu as gagne, tu es pret pour la S2.

Tu n'as DROIT A AUCUNE boucle 'for' dans la Partie B. Tout doit etre
vectorise avec NumPy. Si tu ecris 'for', c'est que tu n'as pas compris
le broadcasting -> relis la page W3Schools 'Array Shape'.

Interdiction d'importer autre chose que numpy.
"""

import numpy as np

# On fixe le hasard pour que tes resultats soient toujours les memes
# (et que la correction automatique fonctionne).
np.random.seed(0)


# ============================================================
# PARTIE A — MAITRISE DES SHAPES (le nerf de la guerre)
# ============================================================
# 90% des bugs d'un reseau de neurones sont des erreurs de dimensions.
# Avant de coder quoi que ce soit d'intelligent, tu dois savoir
# PREDIRE la shape d'un tableau AVANT de lancer le code.

def partie_A():
    # A1. Cree une matrice X de 5 exemples, chacun ayant 3 "features"
    #     (donc shape (5, 3)), remplie de nombres aleatoires.
    #     Indice : np.random.rand(lignes, colonnes)
    X = np.random.rand(5, 3)  # TODO

    # A2. Cree une matrice de poids W qui permet de transformer
    #     3 features d'entree en 4 valeurs de sortie.
    #     Quelle doit etre sa shape ? Reflechis avant de coder.
    W = np.random.rand(3, 4)
    # W.shape = (3, 4) # TODO

    # A3. Cree un vecteur de biais b avec une valeur par sortie.
    #     Shape attendue : (4,).
    b = np.zeros(4) # TODO

    # A4. Sans lancer le code, ECRIS dans ce commentaire la shape que
    #     tu obtiendrais en calculant X @ W :  ( 5 , 4 )
    #     Puis verifie en decommentant la ligne suivante.
    print((X @ W).shape)

    return X, W, b


# ============================================================
# PARTIE B — LES BRIQUES DU RESEAU
# ============================================================
# Ces fonctions sont EXACTEMENT celles que tu reutiliseras en S2.
# Garde-les precieusement. Aucune boucle autorisee.

def sigmoid(z):
    """Fonction d'activation. Ecrase n'importe quel nombre entre 0 et 1.
    Formule : 1 / (1 + e^(-z)).  Indice : np.exp()
    Doit marcher aussi bien sur un nombre que sur un tableau entier."""
    return ((1 + np.exp(-z))** -1)  # TODO


def sigmoid_derivative(z):
    """Derivee de la sigmoid. Astuce connue : si s = sigmoid(z),
    alors la derivee vaut s * (1 - s). Tu en auras besoin pour la
    backpropagation en S2."""
    s = sigmoid(z)
    return s * (1 - s)  # TODO


def relu(z):
    """Activation ReLU : renvoie z si z > 0, sinon 0.
    Indice : np.maximum(...) fait ca sans aucune boucle."""
    return np.maximum(0, z)  # TODO


def relu_derivative(z):
    """Derivee de ReLU : 1 la ou z > 0, sinon 0.
    Indice : un masque booleen (z > 0) converti en float."""
    return (z > 0).astype(float)  # TODO


def mse(y_true, y_pred):
    """Erreur quadratique moyenne : la moyenne des (vrai - predit)^2.
    C'est la fonction de cout : elle mesure a quel point le reseau
    se trompe. Renvoie UN seul nombre."""
    return np.mean((y_true - y_pred)**2)  # TODO


def dense_forward(X, W, b):
    """Le coeur du reseau : le passage d'UNE couche.
    Prend un batch d'entrees X (plusieurs exemples en lignes),
    applique les poids W et ajoute le biais b.
    Formule : X @ W + b
    Le biais b doit s'ajouter a CHAQUE ligne (c'est du broadcasting)."""
    return X @ W + b  # TODO


# ============================================================
# CORRECTION AUTOMATIQUE — n'y touche pas
# ============================================================

def _check():
    erreurs = []

    # --- Partie A ---
    X, W, b = partie_A()
    try:
        assert X.shape == (5, 3), f"X devrait etre (5,3), pas {X.shape}"
        assert W.shape == (3, 4), f"W devrait etre (3,4), pas {W.shape}"
        assert b.shape == (4,),  f"b devrait etre (4,), pas {b.shape}"
        assert (X @ W).shape == (5, 4), "X @ W devrait donner (5,4)"
    except Exception as e:
        erreurs.append(f"[Partie A] {e}")

    # --- sigmoid ---
    try:
        assert abs(sigmoid(0) - 0.5) < 1e-9, "sigmoid(0) doit valoir 0.5"
        assert sigmoid(1000) > 0.99, "sigmoid(grand) doit tendre vers 1"
        assert sigmoid(-1000) < 0.01, "sigmoid(petit) doit tendre vers 0"
    except Exception as e:
        erreurs.append(f"[sigmoid] {e}")

    # --- sigmoid_derivative ---
    try:
        assert abs(sigmoid_derivative(0) - 0.25) < 1e-9, \
            "la derivee en 0 doit valoir 0.25"
    except Exception as e:
        erreurs.append(f"[sigmoid_derivative] {e}")

    # --- relu ---
    try:
        assert relu(3) == 3 and relu(-2) == 0, "relu(3)=3, relu(-2)=0"
        out = relu(np.array([-1.0, 0.0, 2.0]))
        assert np.allclose(out, [0, 0, 2]), "relu sur un tableau est faux"
    except Exception as e:
        erreurs.append(f"[relu] {e}")

    # --- relu_derivative ---
    try:
        d = relu_derivative(np.array([-1.0, 0.5, 2.0]))
        assert np.allclose(d, [0, 1, 1]), "relu_derivative est faux"
    except Exception as e:
        erreurs.append(f"[relu_derivative] {e}")

    # --- mse ---
    try:
        yt = np.array([1.0, 2.0, 3.0])
        yp = np.array([1.0, 2.0, 3.0])
        assert abs(mse(yt, yp)) < 1e-9, "mse de valeurs egales doit valoir 0"
        assert abs(mse(np.array([0.0]), np.array([2.0])) - 4.0) < 1e-9, \
            "mse([0],[2]) doit valoir 4"
    except Exception as e:
        erreurs.append(f"[mse] {e}")

    # --- dense_forward ---
    try:
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0]])          # 2 exemples, 2 features
        W = np.array([[1.0, 0.0],
                      [0.0, 1.0]])          # matrice identite
        b = np.array([10.0, 20.0])
        out = dense_forward(X, W, b)
        assert out.shape == (2, 2), f"sortie devrait etre (2,2), pas {out.shape}"
        attendu = np.array([[11.0, 22.0],
                            [13.0, 24.0]])
        assert np.allclose(out, attendu), \
            "le calcul X@W+b (avec broadcasting du biais) est faux"
    except Exception as e:
        erreurs.append(f"[dense_forward] {e}")

    # --- Verdict ---
    print("=" * 50)
    if not erreurs:
        print("  TOUT EST VERT — tu es pret pour la semaine 2.")
        print("  (tu viens d'ecrire la moitie de ton reseau)")
    else:
        print(f"  {len(erreurs)} chose(s) a corriger :")
        for e in erreurs:
            print("   -", e)
    print("=" * 50)


if __name__ == "__main__":
    _check()
