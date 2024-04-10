Directions théorique:

 - [ ] creuser pour modifier la méthode pour obtenir la complétude que n'offre pas BCOP. (Si on arrive a exprimer toutes les conv orthogonale a support fixe en un temps raisonnable, ça permettra de répondre enfin à la question "l'orthogonalité est elle une propriété désirable pour l'entrainement de réseaux lip ?"

Directions empiriques: (xp pour montrer l'intérêt de layers orthogonale au delà du 1-lip)
robustesse/classif:

 - [ ] re-ordonner les layers de Lipschitz-layers compared ( ça donnerait un message type "implementation matter in provable robustness")
 - [ ] reproduire le sota robustness pour moins cher
 - [ ] tenter un imagenet robuste (j'ai des résultats en classif pure, mais je suis pas très a jour sur les tricks robustesse) ça pourrait faire un tour de chauffe pour le large scale

stabilité des réseaux de neurones

 - [x] entrainer un réseau très profond sans skip connection, dropout, ni batch norm. (pas objectif de sota, mais donner des pistes sur la stabilité, un peu à la manière des SNN)
 - [ ] entrainer un RNN sur des séquences d'images (seule la partie RNN étant orthogonale) il y a une carte a jouer sur la performance, vu que le coût devient linéaire a la taille de la séquence.
 - [ ] remplacer la layer de patch extraction dans les VIT (qui est une conv2d stridée) par un conv orthogonale puis montrer qu'on gagne en stabilité de l'entrainement (certains papiers pointent cette layer comme étant importante pour la stabilité.)

élargir les applications:

 - [ ] segmentation/détection robuste: pour mettre en avant qu'on débloque les U-nets grace au conv transpose stridées efficientes.
 - [ ] robotique (lidar 2 map): problème qui fait intervenir des conv2d et ou on peut garantir une robustesse au bruit L2 sur les mesure Lidar.
 - [ ] tester de la diffusion ? (j'ai pas trouvé de papier pour motiver théoriquement l'intérêt mais il me semble que quelqu'un en avait parlé)
 - [ ] GAN/VAE: Sur le gans la préservation du rank empêche le mode collapse. Sur les VAE on a une layer facilement inversible & les logdet de la jacobienne vaut 1, donc c'est potentiellement plus efficace.


# Purpose of this library :

todo

build your environment using:
```
make prepare-dev
```

install me using:
```
pip install -e .[dev]
```

# Status of the repository :

todo

# library repository template

todo

# pre-commit : Conventional Commits 1.0.0

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

```

The commit contains the following structural elements, to communicate intent to the consumers of your library:

- fix: a commit of the type fix patches a bug in your codebase (this correlates with PATCH in Semantic Versioning).

- feat: a commit of the type feat introduces a new feature to the codebase (this correlates with MINOR in Semantic Versioning).

- BREAKING CHANGE: a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope, introduces a breaking API change (correlating with MAJOR in Semantic Versioning). A BREAKING CHANGE can be part of commits of any type.

- types other than fix: and feat: are allowed, for example @commitlint/config-conventional (based on the the Angular convention) recommends *build:, chore:, ci:, docs:, style:, refactor:, perf:, test:*, and [others](https://delicious-insights.com/fr/articles/git-hooks-et-commitlint/).

- footers other than BREAKING CHANGE: <description> may be provided and follow a convention similar to git trailer format.

- Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., feat(parser): add ability to parse arrays.

# README sections

The following should be used as a template for the README of your library. Of course, depending on what you are doing not all sections are necessary but try to keep the order of the sections.

<!-- Banner section -->
<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/banner_dark.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/banner_light.png">
                <img alt="Library Banner" src="./docs/assets/banner_light.png">
        </picture>
</div>
<br>

<!-- Badge section -->
<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

<!-- Short description of your library -->
<p align="center">
  <b>Libname</b> is a Python toolkit dedicated to make people happy and fun.

  <!-- Link to the documentation -->
  <br>
  <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"><strong>Explore Libname docs »</strong></a>
  <br>

</p>

## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [👀 See Also](#-see-also)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍🎓 Creator](#-creator)
- [🗞️ Citation](#-citation)
- [📝 License](#-license)

## 🔥 Tutorials

We propose some tutorials to get familiar with the library and its api:

- [Getting started](https://colab.research.google.com/drive/1XproaVxXjO9nrBSyyy7BuKJ1vy21iHs2) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deel-ai/<libname>/blob/master/docs/notebooks/demo_fake.ipynb) </sub>

You do not necessarily need to register the notebooks on the GitHub. Notebooks can be hosted on a specific [drive](https://drive.google.com/drive/folders/1DOI1CsL-m9jGjkWM1hyDZ1vKmSU1t-be).

## 🚀 Quick Start

Libname requires some stuff and several libraries including Numpy. Installation can be done using Pypi:

```python
pip install libname
```

Now that Libname is installed, here are some basic examples of what you can do with the available modules.

### Print Hello World

Let's start with a simple example:

```python
from libname.fake import hello_world

hello_world()
```

### Make addition

In order to add `a` to `b` you can use:

```python
from libname.fake import addition

a = 1
b = 2
c = addition(a, b)
```

## 📦 What's Included

A list or table of methods available

## 👍 Contributing

Feel free to propose your ideas or come and contribute with us on the Libname toolbox! We have a specific document where we describe in a simple way how to make your first pull request: [just here](CONTRIBUTING.md).

## 👀 See Also

This library is one approach of many...

Other tools to explain your model include:

- [Random](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

More from the DEEL project:

- [Xplique](https://github.com/deel-ai/xplique) a Python library exclusively dedicated to explaining neural networks.
- [deel-lip](https://github.com/deel-ai/deel-lip) a Python library for training k-Lipschitz neural networks on TF.
- [Influenciae](https://github.com/deel-ai/influenciae) Python toolkit dedicated to computing influence values for the discovery of potentially problematic samples in a dataset.
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) a Python library for training k-Lipschitz neural networks on PyTorch.
- [DEEL White paper](https://arxiv.org/abs/2103.10529) a summary of the DEEL team on the challenges of certifiable AI and the role of data quality, representativity and explainability for this purpose.

## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍🎓 Creators

If you want to highlights the main contributors


## 🗞️ Citation

If you use Libname as part of your workflow in a scientific publication, please consider citing the 🗞️ [our paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ):

```
@article{rickroll,
  title={Rickrolling},
  author={Some Internet Trolls},
  journal={Best Memes},
  year={ND}
}
```

## 📝 License

The package is released under [MIT license](LICENSE).
