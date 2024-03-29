# CNN_person_id
Implémentation de l'article "Improving Person Re-identification by Attribute and Identity Learning". Le but du modèle proposé est de déterminer l'identité d'une personne à partir de features extraites par CNN et d'attributs tels que la couleur des vêtements ou le sexe.

## Pré-requis
Ce projet est réalisé à l'aide des librairies suivantes :
- keras==2.3.1
- tensorflow==1.13.1
- opencv-python==4.1.2
- scikit-learn==0.21.2
- numpy==1.16.4
- matplotlib==3.1.0
- json==2.0.9

## Lancement du projet
Le modèle est entraîné sur la base de données Market-1501 composée de :
- L'ensemble d'images Market-1501
- Le fichier market_attributes.json

Les données sont à placer dans le même dossier que le code.
L'entraînement se lance en exécutant la commande "python train.py", ou "python main.py" (la deuxième commande permet après l'entraînement de tester le modèle, ou de l'utiliser sans l'entraîner si le fichier .h5 du modèle est présent). 

## Architecture
Le projet est constitué des 4 fichiers suivants :
- reid_model.py : fonctions permettant de créer le modèle et ses callbacks
- data.py : fonctions permettant d'extraire et pré-traiter les données du dataset Market-1501
- train.py : lance l'entraînement du modèle, augmente les données, et fournit ses performances sur les métriques mAP, ainsi que rank-1, rank-5, rank-10
- main.py : entraîne le modèle, puis l'applique à la prédiction d'attributs et d'ID pour une image donnée, ainsi qu'à la comparaison de features entre deux images pour ré-identifier une personne

## Modifications apportées aux données
L'architecture et la méthode d'entraînement du modèle sont celles décrites dans l'article cité plus haut (module de repondération compris). La principale modification apportée est l'adaptation des attributs à un problème de classification multi-labels, afin de faciliter l'apprentissage.
La séparation Train/Test est également différente, et correspond ici à 80% du dataset global pour le train et 20% pour le test.

Attributs initiaux :
[2 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 2 2 1 1 1]

Attributs modifiés :
[0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 0]

## Performances sur la classification de l'image parmi les 1501 classes
Train set: 
- 'mAP': 0.9613158840471077
- 'rank_1': 0.9887662691146633
- 'rank_5': 0.9986143415648043
- 'rank_10': 0.9993566585836591

Test set:
- 'mAP': 0.8046109612447331
- 'rank_1': 0.8477830562153602
- 'rank_5': 0.9524940617577197
- 'rank_10': 0.9716943784639747

Evolution de la fonction de coût et de la précision du modèle :
![Fonction de coût pour le modèle complet, la prédiction d'attributs, et la classification de l'image](./loss.png)

Fonction de coût pour le modèle complet, la prédiction d'attributs, et la classification de l'image. La fonction de coût totale du modèle est calculée de la manière suivante : L_totale = 0.9*L_id + 0.1*L_attributs.

![Précision pour le modèle complet, la prédiction d'attributs, et la classification de l'image](./acc.png)

Précision du modèle pour la prédiction d'attributs, et la classification de l'image (de haut en bas).

### Contributeurs
Projet réalisé dans le cadre de la majeure Intelligence Artificielle (ESME Sudria 2019/2020) par Alexandre Lacour et Hippolyte Foussat.
