---

## Table des matières

1. [Problème posé et contexte](#1-problème-posé-et-contexte)
2. [Prétraitement : conversion en niveaux de gris et flou gaussien](#2-prétraitement--conversion-en-niveaux-de-gris-et-flou-gaussien)
3. [Soustraction de fond : formulation générale](#3-soustraction-de-fond--formulation-générale)
4. [Méthode 1 — Différence entre images](#4-méthode-1--différence-entre-images)
5. [Méthode 2 — Moyenne glissante](#5-méthode-2--moyenne-glissante)
6. [Méthode 3 — MOG2 : mélange de gaussiennes](#6-méthode-3--mog2--mélange-de-gaussiennes)
7. [Méthode 4 — KNN : soustraction de fond par k plus proches voisins](#7-méthode-4--knn--soustraction-de-fond-par-k-plus-proches-voisins)
8. [Détection des ombres](#8-détection-des-ombres)
9. [Post-traitement : morphologie mathématique](#9-post-traitement--morphologie-mathématique)
10. [Composantes connexes et filtrage par aire](#10-composantes-connexes-et-filtrage-par-aire)
11. [Détection de contours et boîtes englobantes](#11-détection-de-contours-et-boîtes-englobantes)
12. [Guide des paramètres](#12-guide-des-paramètres)

---

## 1. Problème posé et contexte

La détection de mouvement consiste à identifier automatiquement **quels pixels d'une image vidéo appartiennent à un objet en mouvement**, par opposition à un fond statique. C'est un problème fondamental en vision par ordinateur, avec des applications en surveillance, suivi du trafic, interaction homme-machine, analyse sportive et robotique.

Formellement, considérons une vidéo comme une suite discrète d'images $I_t : \Omega \to \mathbb{R}^C$, où $\Omega \subset \mathbb{Z}^2$ est la grille spatiale des pixels, $t \in \mathbb{N}$ est l'indice temporel, et $C = 3$ pour une image couleur (canaux BGR). L'objectif est de calculer, pour chaque image $I_t$, un **masque binaire de premier plan** :

$$M_t(x, y) = \begin{cases} 255 & \text{si le pixel } (x,y) \text{ appartient à un objet en mouvement} \\ 0 & \text{sinon} \end{cases}$$

Ce problème est plus difficile qu'il n'y paraît. Le fond est rarement parfaitement statique : les variations d'éclairage, les arbres qui bougent, les vibrations de la caméra et les dérives progressives de luminosité modifient tous les pixels du fond au cours du temps. Un détecteur robuste doit distinguer ces variations *parasites* du véritable mouvement des objets.

Les quatre méthodes implémentées dans cette application couvrent une progression allant de l'approche la plus simple possible (différence entre images) jusqu'à des modèles statistiques appris (MOG2, KNN), et représentent les principales familles d'algorithmes classiques de soustraction de fond.

---

## 2. Prétraitement : conversion en niveaux de gris et flou gaussien

### 2.1 Conversion en niveaux de gris

Les quatre méthodes opèrent sur des **images d'intensité à un seul canal**. Chaque image couleur $I_t$ est convertie en une image en niveaux de gris $G_t : \Omega \to [0, 255]$ à l'aide de la formule de luminance ITU-R BT.601 :

$$G_t(x,y) = 0.299 \cdot R(x,y) + 0.587 \cdot V(x,y) + 0.114 \cdot B(x,y)$$

où $R$, $V$, $B$ désignent respectivement les valeurs des canaux rouge, vert et bleu. Les coefficients reflètent la sensibilité non uniforme de la vision humaine à chaque couleur primaire, le vert étant le contributeur perceptif dominant.

Travailler en niveaux de gris réduit le coût de calcul d'un facteur 3 et est suffisant, car le mouvement se manifeste principalement comme une variation d'*intensité*, et non de teinte.

### 2.2 Flou gaussien

Avant toute opération de différenciation, chaque image en niveaux de gris est convoluée par un **noyau gaussien** afin d'atténuer le bruit haute fréquence :

$$\tilde{G}_t = G_t * h_\sigma$$

où le noyau gaussien 2D est donné par :

$$h_\sigma(u, v) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{u^2 + v^2}{2\sigma^2}\right)$$

En pratique, on utilise un noyau discret de taille $(k \times k)$, avec $k$ impair, et $\sigma$ est implicitement fixé par OpenCV selon $\sigma = 0.3 \cdot \frac{k-1}{2} + 0.8$. Le rayon de flou $k$ est exposé dans l'interface par le paramètre **Blur kernel size**.

**Pourquoi flouter ?** Le bruit du capteur et les artefacts de compression introduisent des pixels isolés qui scintillent d'une image à l'autre. Sans flou, ils produiraient des milliers de fausses détections de premier plan. Le flou échange une petite perte de résolution spatiale contre une amélioration très nette du rapport signal sur bruit dans l'image de différence.

> **Important** : le flou gaussien est appliqué **uniquement dans les méthodes Frame Difference et Running Average**, où les différences pixel à pixel sont calculées manuellement. Les soustracteurs MOG2 et KNN gèrent eux-mêmes statistiquement une partie du bruit, mais le flou est également appliqué à l'image avant de la transmettre à ces méthodes, ce qui assure un prétraitement cohérent dans tous les modes.

---

## 3. Soustraction de fond : formulation générale

Les quatre méthodes reposent sur le même paradigme général : maintenir, implicitement ou explicitement, une estimation du **modèle de fond** $B_t(x,y)$, puis déclarer un pixel comme appartenant au premier plan s'il s'écarte suffisamment de cette estimation :

$$M_t(x,y) = \mathbf{1}\!\left[\,d\!\left(G_t(x,y),\; B_t(x,y)\right) > \tau\,\right]$$

où $d(\cdot, \cdot)$ est une mesure de distance, $\tau$ est un seuil, et $\mathbf{1}[\cdot]$ est la fonction indicatrice. Les quatre méthodes diffèrent par (i) la manière dont $B_t$ est représenté, (ii) la définition de $d$, et (iii) la manière dont le modèle est mis à jour au cours du temps.

---

## 4. Méthode 1 — Différence entre images

### 4.1 Principe

La différence entre images est l'approche la plus directe : le modèle de fond à l'instant $t$ est simplement **l'image précédente** $G_{t-1}$. L'image de différence vaut :

$$D_t(x,y) = \left| G_t(x,y) - G_{t-1}(x,y) \right|$$

Un pixel est déclaré au premier plan si cette différence absolue dépasse un seuil $\tau$ :

$$M_t(x,y) = \mathbf{1}\!\left[D_t(x,y) > \tau\right]$$

### 4.2 Propriétés et limites

La différence entre images est extrêmement rapide et n'a pratiquement aucun coût mémoire, puisqu'une seule image précédente doit être conservée. Cependant, elle souffre d'un problème structurel fondamental appelé **effet de double bord** ou **problème d'ouverture des images de différence**. Considérons un rectangle uniforme en mouvement : seuls ses bords avant et arrière changent d'une image à l'autre, tandis que son intérieur, s'il est uniforme, produit une différence nulle. Le masque obtenu est alors une enveloppe creuse du contour de l'objet, et non une silhouette remplie.

Mathématiquement, le masque $M_t$ approxime le **gradient temporel** de la vidéo :

$$M_t \approx \mathbf{1}\!\left[\left|\frac{\partial I}{\partial t}(x,y,t)\right| > \tau\right]$$

Cette quantité n'est grande qu'aux endroits où l'intensité *change*, et pas nécessairement là où un objet *se trouve*. Pour des objets texturés, on obtient des masques plus remplis. Pour des objets lisses ou uniformes, l'intérieur reste non détecté.

Une seconde limite est la sensibilité aux secousses rapides de la caméra ou aux variations globales d'éclairage : toute variation globale de luminosité affecte simultanément tous les pixels, produisant un masque presque entièrement faux positif. Cette méthode ne convient donc qu'aux caméras fixes et aux scènes à éclairage stable.

Le **Difference threshold** $\tau$ (plage du curseur 1–255) contrôle la sensibilité. Une faible valeur permet de détecter des mouvements subtils mais augmente les faux positifs liés au bruit ; une valeur élevée exige un fort contraste sur les bords en mouvement.

---

## 5. Méthode 2 — Moyenne glissante

### 5.1 Modèle de fond par moyenne exponentielle

Au lieu d'utiliser seulement l'image immédiatement précédente, la méthode Running Average maintient une **estimation continue et adaptative** du fond en calculant une moyenne exponentiellement pondérée de toutes les images passées :

$$B_t(x,y) = (1 - \alpha)\, B_{t-1}(x,y) + \alpha\, G_t(x,y)$$

où $\alpha \in (0, 1)$ est le **learning rate**, un hyperparamètre crucial. En résolvant cette récurrence, on obtient l'expression fermée :

$$B_t(x,y) = \alpha \sum_{k=0}^{t} (1-\alpha)^{t-k}\, G_k(x,y)$$

Il s'agit d'un **filtre passe-bas à réponse impulsionnelle infinie (IIR)** appliqué le long de l'axe temporel. Chaque image passée contribue avec un poids décroissant exponentiellement, de sorte que les images récentes influencent davantage l'estimation du fond que les anciennes. La mémoire temporelle effective de ce filtre, c'est-à-dire le nombre d'images qui contribuent de manière significative, est de l'ordre de $1/\alpha$.

### 5.2 Détection du premier plan

Une fois $B_t$ calculé, l'étape de détection est analogue à celle de la différence entre images :

$$D_t(x,y) = \left| G_t(x,y) - B_t(x,y) \right|, \qquad M_t(x,y) = \mathbf{1}\!\left[D_t(x,y) > \tau\right]$$

### 5.3 Effet du learning rate

Le learning rate $\alpha$ gouverne le compromis entre **adaptabilité** et **stabilité** :

- **$\alpha$ élevé** (par exemple 0.5) : le modèle de fond s'adapte rapidement aux changements, ce qui est utile dans les scènes où l'éclairage évolue progressivement. En revanche, un objet de premier plan qui reste immobile suffisamment longtemps sera absorbé dans le fond et finira par disparaître du masque.
- **$\alpha$ faible** (par exemple 0.001) : le fond est très stable, et un objet arrêté reste détecté comme premier plan pendant longtemps. L'inconvénient est une adaptation lente aux véritables changements du fond, par exemple lorsqu'une lumière s'allume.

Cette tension fondamentale entre **persistance** et **plasticité** est un problème universel dans la modélisation adaptative du fond. Dans le domaine fréquentiel, la moyenne glissante correspond à un filtre passe-bas du premier ordre, avec une fréquence de coupure à 3 dB donnée par $f_c \approx \alpha / (2\pi)$ images$^{-1}$.

### 5.4 Comparaison avec la différence entre images

Contrairement à la différence entre images, le modèle de fond par moyenne glissante intègre l'information sur de nombreuses images. Son image $D_t$ correspond davantage au **déplacement** de l'objet par rapport à sa position moyenne à long terme qu'au simple changement instantané entre deux images consécutives. Cela donne de meilleurs masques remplis pour les objets se déplaçant lentement ou s'arrêtant brièvement, au prix d'une phase d'initialisation : les premières images de la vidéo produisent une estimation de fond peu fiable, car $B_t$ n'a pas encore convergé.

---

## 6. Méthode 3 — MOG2 : mélange de gaussiennes

### 6.1 Motivation : modélisation distributionnelle par pixel

Les deux méthodes précédentes représentent le fond en chaque pixel par **une seule valeur**. Cette représentation est fragile en présence d'un comportement de fond *multimodal*. Par exemple, un pixel peut alterner entre un ciel lumineux et les feuilles d'un arbre agité par le vent. Aucune valeur unique ne représente correctement cet état de fond.

La famille des méthodes par mélange de gaussiennes (MoG), et en particulier MOG2 (Zivkovic, 2004 ; Zivkovic & van der Heijden, 2006), modélise l'intensité de chaque pixel comme un **mélange de $K$ distributions gaussiennes** :

$$p(x_{t}) = \sum_{k=1}^{K} w_{k,t} \cdot \mathcal{N}(x_t;\, \mu_{k,t},\, \sigma_{k,t}^2)$$

où $w_{k,t}$ est le poids (coefficient de mélange) de la $k$-ième gaussienne à l'instant $t$, $\mu_{k,t}$ sa moyenne, et $\sigma_{k,t}^2$ sa variance. Notons que dans l'implémentation d'OpenCV, une variance scalaire (isotrope) est utilisée par composante pour des raisons d'efficacité, plutôt qu'une matrice de covariance complète.

### 6.2 Classification fond / premier plan

Les $K$ composantes ne représentent pas toutes le fond. À chaque instant, les composantes sont triées selon le rapport $w_{k}/\sigma_{k}$, qui classe heuristiquement les composantes selon leur caractère "stable et fréquent" : une gaussienne étroite et fortement pondérée correspond à un état de fond stable et récurrent. Les premières $B$ composantes dont le poids cumulé dépasse un seuil représentent le fond :

$$B = \arg\min_b \left\{ \sum_{k=1}^{b} w_k > T_{bg} \right\}$$

Une nouvelle observation $x_t$ au pixel $(x,y)$ est classée comme **fond** si elle se trouve à moins de $2.5\sigma$ d'au moins une des composantes de fond, au sens de la distance de Mahalanobis :

$$\text{foreground}(x_t) = \mathbf{1}\!\left[\, \nexists\, k \leq B : \frac{(x_t - \mu_k)^2}{\sigma_k^2} < \lambda \,\right]$$

où $\lambda$ est le **variance threshold** correspondant au paramètre de seuil de variance de MOG2. Augmenter $\lambda$ rend le classifieur plus tolérant vis-à-vis des écarts au modèle de fond, ce qui réduit les faux positifs au prix d'une moindre sensibilité.

### 6.3 Mise à jour en ligne du modèle

Le modèle MOG2 est mis à jour en ligne à chaque nouvelle image. Pour la composante appariée $k^*$, les équations de mise à jour sont :

$$w_{k^*,t} \leftarrow (1 - \alpha)\, w_{k^*,t} + \alpha$$

$$\mu_{k^*,t} \leftarrow (1 - \rho)\, \mu_{k^*,t} + \rho\, x_t$$

$$\sigma_{k^*,t}^2 \leftarrow (1 - \rho)\, \sigma_{k^*,t}^2 + \rho\, (x_t - \mu_{k^*,t})^2$$

où $\rho = \alpha \cdot \mathcal{N}(x_t; \mu_{k^*}, \sigma_{k^*}^2)$ est un learning rate par composante, pondéré par la qualité de l'ajustement entre l'observation et la composante. Tous les autres poids diminuent selon : $w_{k \neq k^*} \leftarrow (1-\alpha)\, w_{k}$. Si aucune composante ne correspond, la composante la moins probable est remplacée par une nouvelle composante initialisée en $x_t$.

Le **learning rate** $\alpha$ est équivalent à $1/H$ où $H$ est le paramètre **History**. Il contrôle donc le nombre d'images passées qui contribuent effectivement au modèle. Une valeur de 150 pour l'historique signifie que le modèle possède une mémoire temporelle d'environ 150 images.

### 6.4 Avantages de MOG2

MOG2 gère efficacement les fonds complexes grâce à sa représentation multimodale : chaque pixel peut conserver simultanément plusieurs états de fond distincts. Il gère naturellement les variations progressives d'éclairage, par déplacement lent des moyennes $\mu_k$, ainsi que les mouvements périodiques du fond tels que le feuillage, grâce à une composante dédiée pour chaque état. Son principal coût est un surcoût en calcul et en mémoire : il faut maintenir $K$ gaussiennes pour chaque pixel, avec en général $K$ compris entre 3 et 5.

---

## 7. Méthode 4 — KNN : soustraction de fond par k plus proches voisins

### 7.1 Modélisation non paramétrique du fond

La soustraction de fond par KNN (Zivkovic & van der Heijden, 2006) adopte une approche fondamentalement différente, **non paramétrique**. Au lieu d'ajuster des distributions gaussiennes, elle maintient pour chaque pixel un **ensemble d'échantillons** $\mathcal{S}_{t} = \{s_1, s_2, \ldots, s_N\}$ constitué des $N$ intensités les plus récemment observées à ce pixel dans des conditions de fond.

Le modèle de fond n'est pas une densité de probabilité sous forme fermée, mais une distribution empirique représentée directement par les échantillons. La classification d'une nouvelle observation $x_t$ repose sur ses **plus proches voisins** dans $\mathcal{S}_t$ : si au moins $k$ parmi les $N$ échantillons stockés sont plus proches de $x_t$ qu'un seuil de distance $d^2$, alors le pixel est classé comme fond :

$$\text{background}(x_t) = \mathbf{1}\!\left[\left|\left\{s_i \in \mathcal{S}_t : (x_t - s_i)^2 \leq d^2\right\}\right| \geq k\right]$$

Le **KNN distance threshold** $d^2$ est la distance euclidienne au carré dans l'espace des intensités. Un pixel est classé comme premier plan si moins de $k$ échantillons de fond sont situés à une distance inférieure ou égale à $d$ de l'observation courante.

### 7.2 Comparaison avec MOG2

| Propriété | MOG2 | KNN |
|---|---|---|
| Représentation du fond | Paramétrique (mélange de gaussiennes) | Non paramétrique (ensemble d'échantillons) |
| Support multimodal | Oui ($K$ composantes) | Oui (implicitement, via la diversité des échantillons) |
| Complexité par pixel | $O(K)$ | $O(N)$ |
| Adaptation à de nouvelles distributions | Progressive par mise à jour des poids | Immédiate lors du remplacement d'un échantillon |
| Paramètre principal | Variance threshold | Distance threshold $d^2$ |

KNN tend à mieux fonctionner que MOG2 dans les scènes présentant une dynamique de fond très complexe et non gaussienne, par exemple les surfaces d'eau, les flammes ou les écrans, car il ne fait aucune hypothèse de distribution. MOG2 est généralement plus rapide pour de petites valeurs de $K$ et peut être plus stable numériquement.

### 7.3 Mise à jour des échantillons

L'ensemble d'échantillons est mis à jour en remplaçant d'anciens échantillons par de nouvelles observations de fond, à un rythme gouverné par le learning rate. Comme seuls les pixels confirmés comme fond contribuent à l'ajout de nouveaux échantillons, le modèle se renforce lui-même : une fois le fond initial appris, les événements parasites de premier plan ne corrompent pas le modèle de fond.

---

## 8. Détection des ombres

MOG2 et KNN incluent tous deux une étape optionnelle de **détection des ombres**, activée par la case *Detect shadows*. Cette étape corrige un mode d'échec fréquent : l'ombre portée par un objet en mouvement est plus sombre que le fond, mais ne fait pas partie de l'objet lui-même. Sans détection des ombres, celles-ci sont intégrées au masque de premier plan, ce qui rend les boîtes englobantes beaucoup plus grandes que l'objet réel.

### 8.1 Modèle d'ombre

La détection d'ombre utilise un modèle couleur simple mais efficace (Prati et al., 2003). Un pixel $(x,y)$ est classé comme **ombre** et non comme premier plan si :

$$\alpha_{shadow} \leq \frac{I_t(x,y)}{B_t(x,y)} \leq \beta_{shadow} \qquad \text{and} \qquad \left|\arg(I_t) - \arg(B_t)\right| < \tau_{hue}$$

où $\alpha_{shadow}$ et $\beta_{shadow}$ bornent le rapport acceptable entre l'intensité courante et l'intensité de fond, une ombre assombrit l'image mais ne modifie pas fortement la couleur, et où la différence angulaire dans le canal de teinte doit rester faible. Dans l'espace HSV, une ombre réduit principalement le canal $V$ tandis que le canal $H$ reste presque inchangé.

### 8.2 Encodage de sortie

Lorsque la détection des ombres est activée, le masque brut retourné par le soustracteur utilise un **encodage à trois valeurs** :
- $0$ : fond
- $127$ : ombre
- $255$ : premier plan

Après le soustracteur, un seuil est appliqué à la valeur 200 afin de binariser le masque, ce qui élimine les pixels d'ombre (127) et conserve seulement le véritable premier plan (255). Lorsque la détection des ombres est désactivée, le seuil est fixé à 1, toute valeur non nulle étant alors considérée comme du premier plan.

---

## 9. Post-traitement : morphologie mathématique

Le masque binaire brut $M_t$ produit par l'une quelconque des quatre méthodes contient généralement du **bruit** : pixels isolés dus au bruit du capteur ou à la compression, ainsi que petits trous ou lacunes à l'intérieur des objets détectés, notamment dans les zones peu texturées. Une morphologie mathématique est appliquée pour nettoyer le masque avant la détection des contours.

### 9.1 Élément structurant

Toutes les opérations morphologiques utilisent un **élément structurant elliptique** $\mathcal{B}$ de taille $k \times k$ :

$$\mathcal{B} = \{(u,v) : (2u/k)^2 + (2v/k)^2 \leq 1\}$$

L'ellipse est préférée au carré car elle est plus isotrope : elle évite d'introduire des artefacts carrés artificiels dans la géométrie du masque et agit de manière moins agressive dans les coins des structures diagonales.

### 9.2 Érosion et dilatation

Les deux opérations morphologiques élémentaires sont :

**Érosion** — un pixel $(x,y)$ de l'image érodée vaut 1 seulement si *tous* les pixels du voisinage $\mathcal{B}$ centré en $(x,y)$ valent 1 dans le masque d'origine :

$$(\text{Erosion: } M \ominus \mathcal{B})(x,y) = \min_{(u,v) \in \mathcal{B}} M(x+u, y+v)$$

**Dilatation** — un pixel $(x,y)$ de l'image dilatée vaut 1 si *au moins un* pixel du voisinage vaut 1 :

$$(\text{Dilation: } M \oplus \mathcal{B})(x,y) = \max_{(u,v) \in \mathcal{B}} M(x+u, y+v)$$

### 9.3 Ouverture (suppression du bruit)

L'**ouverture morphologique** est définie comme une érosion suivie d'une dilatation :

$$M_{\text{open}} = (M \ominus \mathcal{B}) \oplus \mathcal{B}$$

Géométriquement, l'ouverture supprime toutes les structures de premier plan **plus petites que l'élément structurant** : pixels isolés, filaments fins et petites taches sont effacés. Les grandes régions connexes sont conservées, mais leurs contours sont légèrement lissés. Il s'agit de l'opération principale pour supprimer les faux positifs dus au bruit du capteur.

Le paramètre **Opening kernel size** contrôle la taille de $\mathcal{B}$. Une valeur de 1 désactive cette opération. L'augmenter est une action agressive : les régions de premier plan plus petites que le rayon du noyau sont entièrement supprimées.

### 9.4 Fermeture (comblement des lacunes)

La **fermeture morphologique** est une dilatation suivie d'une érosion :

$$M_{\text{close}} = (M \oplus \mathcal{B}) \ominus \mathcal{B}$$

La fermeture comble les **petits trous et petites coupures** à l'intérieur des régions de premier plan : les taches sombres à l'intérieur d'un objet en mouvement, causées par des zones uniformes produisant de faibles différences entre images, sont remplies. Elle fusionne également des fragments proches de premier plan en une seule région cohérente.

Le paramètre **Closing kernel size** contrôle l'ampleur de ce comblement. Un noyau de fermeture trop grand fusionnera des objets proches mais distincts en une seule détection.

**Ordre de traitement** : l'ouverture est appliquée d'abord, puis la fermeture. C'est la séquence standard : on supprime d'abord le bruit, puis on comble les lacunes dans le masque nettoyé. Inverser cet ordre reviendrait à remplir d'abord les trous, puis potentiellement à les rouvrir.

---

## 10. Composantes connexes et filtrage par aire

Après le post-traitement morphologique, le masque binaire $M_t$ peut encore contenir de petites taches connexes résiduelles qui ont traversé le filtre morphologique. Une seconde étape de filtrage repose alors sur une **analyse en composantes connexes**.

### 10.1 Composantes connexes

Une **composante connexe** dans une image binaire est un ensemble maximal de pixels de premier plan tels que chacun peut être rejoint depuis les autres par un chemin de pixels adjacents de premier plan. Ici, on utilise une **8-connexité**, ce qui signifie que chaque pixel possède jusqu'à 8 voisins, horizontaux, verticaux et diagonaux.

L'algorithme (Suzuki, 1985) attribue une étiquette entière unique $\ell$ à chaque composante connexe et retourne, pour chacune d'elles, un ensemble de statistiques comprenant :
- $A_\ell$ : l'aire en pixels
- $(x_\ell, y_\ell, w_\ell, h_\ell)$ : la boîte englobante

### 10.2 Filtrage par aire

Chaque composante n'est conservée que si son aire dépasse le seuil **minimum object area** $A_{min}$ :

$$M_t^{\text{filtered}}(x,y) = \mathbf{1}\!\left[\ell(x,y) \neq 0 \;\wedge\; A_{\ell(x,y)} \geq A_{min}\right]$$

Cette étape élimine les petites taches parasites qui ont survécu à l'ouverture morphologique. Choisir correctement $A_{min}$ suppose de connaître la taille minimale attendue des objets d'intérêt en nombre de pixels, ce qui dépend de la résolution et de la distance des objets à la caméra.

---

## 11. Détection de contours et boîtes englobantes

La dernière étape transforme le masque binaire en une sortie interprétable : des **annotations visuelles** superposées à l'image d'origine.

### 11.1 Extraction des contours

Les **contours** sont les frontières des régions de premier plan. La fonction `findContours` d'OpenCV implémente l'algorithme de Suzuki & Abe (1985), qui suit la frontière extérieure de chaque composante connexe comme une suite ordonnée de coordonnées de pixels. Le mode `RETR_EXTERNAL` ne conserve que le contour le plus externe de chaque région, en ignorant les trous intérieurs. L'option `CHAIN_APPROX_SIMPLE` compresse les segments horizontaux, verticaux et diagonaux en ne conservant que leurs extrémités.

### 11.2 Rectangles englobants

Pour chaque contour, on calcule le **rectangle englobant aligné sur les axes** :

$$(x_c, y_c, w_c, h_c) = \arg\min_{x,y,w,h} \{wh \;:\; \text{contour} \subseteq [x, x+w] \times [y, y+h]\}$$

Il s'agit du plus petit rectangle droit contenant tous les points du contour. C'est une approximation grossière mais peu coûteuse de l'étendue de l'objet. Des descripteurs de forme plus précis, comme le rectangle orienté d'aire minimale, l'enveloppe convexe ou l'ajustement elliptique, pourraient être utilisés à coût plus élevé.

### 11.3 Composition des superpositions

Deux vidéos de sortie sont produites :

**Foreground mask video** — le masque binaire $M_t$, en niveaux de gris puis converti en BGR pour l'écriture vidéo, est sauvegardé directement. Cela donne une vue claire et non ambiguë de ce que le détecteur classe comme premier plan à chaque image.

**Overlay video** — l'image couleur d'origine est mélangée à une mise en évidence du masque sur le canal vert à l'aide d'une somme pondérée :

$$I_{\text{overlay}} = \alpha_1 \cdot I_{\text{frame}} + \alpha_2 \cdot I_{\text{green}}$$

avec $\alpha_1 = 1.0$ et $\alpha_2 = 0.45$, où $I_{\text{green}}$ est une image à trois canaux nulle partout sauf sur le canal vert, où elle vaut $M_t$. Le rectangle englobant de chaque objet détecté est ensuite tracé en rouge par-dessus.

---

## 12. Guide des paramètres

Le tableau ci-dessous résume tous les paramètres visibles par l'utilisateur, leur rôle mathématique et des conseils pratiques de réglage.

| Paramètre | Méthode(s) | Rôle mathématique | Conseil de réglage |
|---|---|---|---|
| **History** | MOG2, KNN | Mémoire temporelle $H \approx 1/\alpha$ ; nombre d'images utilisées pour construire le modèle | Augmenter pour des fonds évoluant lentement ; diminuer pour des changements rapides |
| **MOG2 variance threshold** $\lambda$ | MOG2 | Seuil de distance de Mahalanobis pour la classification fond / premier plan | Augmenter pour réduire les faux positifs ; diminuer pour améliorer la sensibilité |
| **KNN distance threshold** $d^2$ | KNN | Distance d'intensité au carré pour l'appariement par plus proches voisins | Plus grand = moins sensible ; plus petit = plus de faux positifs dus au bruit |
| **Detect shadows** | MOG2, KNN | Active une discrimination ombre / premier plan basée sur HSV | À activer lorsque les ombres provoquent des détections trop grandes |
| **Learning rate** $\alpha$ | MOG2, KNN, Running Avg | Poids de l'image courante dans la mise à jour du modèle de fond | Élevé : adaptation rapide mais disparition possible des objets immobiles ; faible : fond stable mais adaptation lente |
| **Difference threshold** $\tau$ | Frame Diff, Running Avg | Seuil de binarisation sur l'image de différence absolue | À ajuster selon le contraste attendu entre objets en mouvement et fond |
| **Blur kernel size** $k$ | All | Taille du noyau gaussien de pré-lissage, doit être impaire | Augmenter pour des vidéos bruitées ; garder faible pour préserver les petits détails de mouvement |
| **Opening kernel size** | All | Élément structurant pour érosion + dilatation, suppression du bruit | Augmenter pour éliminer les petites détections parasites |
| **Closing kernel size** | All | Élément structurant pour dilatation + érosion, comblement des lacunes | Augmenter pour obtenir des masques d'objets plus pleins et compacts |
| **Minimum object area** $A_{min}$ | All | Seuil de filtrage par aire des composantes connexes, en pixels | À fixer selon la taille minimale attendue des objets en pixels |
| **Maximum output dimension** | All | Redimensionnement de la plus grande dimension de la vidéo avant traitement | À réduire pour accélérer le traitement des vidéos haute résolution |
| **Box thickness** | All | Épaisseur visuelle des rectangles englobants dans la superposition | Purement esthétique |
| **Max frames** | All | Limite le nombre d'images traitées | À réduire pour des aperçus rapides |

---