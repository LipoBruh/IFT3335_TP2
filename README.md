# IFT3355_Devoir3
Text Classification with scikit-learn


## Contributors :

- Emanuel Rollin - 20106951
- Anne Sophie Rozefort - 20189221





## References
- [Scikit-learn.org](https://scikit-learn.org/stable/)
- [Documentation](https://scikit-learn.org/0.21/documentation.html)
- [2h Scikit Tutorial](https://www.youtube.com/watch?v=0B5eIE_1vpU)





## Prerequisites 

Make sure you have python installed globally. Else not much will work lol.
Pip is usefull, worth having too.


###### Linux : 
```bash
sudo apt update
sudo apt install python3
#You usually need those too : 
sudo apt install python3-venv python3-full
```

###### Windows :
[Releases & Installers](https://www.python.org/downloads/windows/)





## Dependencies

How to install the dependencies from requirement.txt :

Create a virtual environment in your project directory:

```bash
python -m venv venv
#then start a terminal in your virtual environment 
```

```bash
#Activate your virtual environment

#Windows :
.\venv\Scripts\activate
#Linux : 
source venv/bin/activate
```

Make sure the environment is selected. Use the shortcut Ctrl+Shift+P and pick the Python: Select Interpreter option and select the venv interpreter. Then you can install the `requirements.txt` without impacting your global dependencies.


And then install them with :

```bash
pip install -r requirements.txt
```

That's it !




## Part 1 - Classification 

#### Notes sur Bag of Words
- [1 minute introduction](https://www.youtube.com/watch?v=OGK9SHt8SWg)
- [20 minute introduction](https://www.youtube.com/watch?v=QbPDjzk2oCA)

Count the frequency of words from a dataset and compare frequencies of words / attributes within an output / category. 

La fréquence des mots peut compter comme une **feature selection method**. Bag-of-Words se concentre sur l'occurence des mots dans une classe de texte. Si une classe favorise certains mots une autre favorise de différents mots, ils seronts faciles à différencier à partir des mots qui ont peu d'overlap / intersection dans leur vocabulaire commun. 

**Binary BOW** : Chaque vecteur représente un document. Chaque index représente un mot présent. Un *document term matrix* représente l'ensemble de ces vecteurs / des documents. Ne s'attarde pas sur la fréquence. Les mots des vecteurs sont triés et facilitent l'indexation. Le vecteur peut être interprété comme un point dans un espace multidimensionel (vector space model). La distance et la direction entre deux textes / points encode la similarité. On peut utiliser le dot product pour simplifier la métrique. Les vecteurs orthogonaux sont égal à zéro et considérés peu similaires. Les points ayant un dot product positif ont une certaine ressemblance.
 
**Frequency BOW**: Similaire. Implémentation 1 : on peut incrémenter leur valeur dans le vecteur. Si on utilise le dot product, on normalise ensuite pour diminuer l'impact de la fréquence sur cette métrique. Équivalent à mesurer la cosine similarity, tq deux points sont similaires avec une métrique égale à 1, et différents si la métrique égal zéro. Le domaine est borné entre [0,1] car aucune fréquence négative. Si l'intersection est vide, la métrique sera zéro. 

Limitations : 
- Synonymes non détectés
- Pluriel considéré différent
- Temps de verbes affectent la similarité aussi
- Ne considère pas l'ordre des mots. Ne s'intéresse pas au contexte.

Considérant ces limitations, un texte plus long permet de mitiger les faiblesses du *Frequency BOW*. Les n-grams peuvent aussi aider à capturer un peu de contexte. Les n-grams peuvent être mélangés aux unigrammes avec une certaine implémentation.

#### Notes sur TF-IDF
- [8 minutes intro](https://www.youtube.com/watch?v=zLMEnNbdh4Q)
- [5 minutes intro](https://www.youtube.com/watch?v=x1u5TotQ0G0)

TF: Bag-like. Ne préserve pas l'ordre des mots. Se sert aussi de la fréquence des mots de chaque entrée / document. 

IDF : Inverse document frequency. Se sert de la formule `log(N/(d w/ t))`, tq N = nb de documents du corpus, d = nombre de documents ayant le mot t. Permet de donner davantage d'importance aux mots rares. Low score = mot commun, high score = mot rare.

TF-IDF : Multiplication naive de chaque TD et IDF d'un document pour produire le vecteur final. Aide à extraire les mots reliés à un contexte / sujet spécifique dans un texte. Ignore aussi les synonymes et l'ordre des mots.


### Pre traitement
- Téléchargement du dataset .csv sur [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?phase=FinishSSORegistration&returnUrl=/datasets/uciml/sms-spam-collection-dataset/versions/1)
- Extraire les mots via `parser`, retirer les caractères spéciaux et convertir en minuscule via `
-

#### Vectorizers
- [10 mins tutorial](https://www.youtube.com/watch?v=W2YJ_IOSWd0)



### Q1.1 
> Quelle méthode (BoW ou TF-IDF) donne les meilleurs résultats ? Pourquoi pensez-vous que c’est le cas ?

### Q1.2
> Que se passe-t-il pour les métriques si vous diminuez `max_features` ? (Montrez un graphique)
