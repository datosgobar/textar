# Analizador de Textos

[![Coverage Status](https://coveralls.io/repos/github/datosgobar/textar/badge.svg?branch=master)](https://coveralls.io/github/datosgobar/textar?branch=master)
[![Build Status](https://travis-ci.org/datosgobar/textar.svg?branch=master)](https://travis-ci.org/datosgobar/textar)
[![PyPI](https://badge.fury.io/py/textar.svg)](http://badge.fury.io/py/textar)
[![Stories in Ready](https://badge.waffle.io/datosgobar/textar.png?label=ready&title=Ready)](https://waffle.io/datosgobar/textar)
[![Documentation Status](http://readthedocs.org/projects/textar/badge/?version=latest)](http://textar.readthedocs.org/en/latest/?badge=latest)

Paquete en python para análisis, clasificación y recuperación de textos, utilizado por el equipo de Datos Argentina.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Instalación](#instalaci%C3%B3n)
  - [Dependencias](#dependencias)
  - [Desde pypi](#desde-pypi)
  - [Para desarrollo](#para-desarrollo)
- [Uso](#uso)
  - [Búsqueda de textos similares](#b%C3%BAsqueda-de-textos-similares)
  - [Clasificación de textos](#clasificaci%C3%B3n-de-textos)
- [Tests](#tests)
- [Créditos](#cr%C3%A9ditos)
- [Contacto](#contacto)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

* Licencia: MIT license

## Instalación

### Dependencias

`textar` usa `pandas`, `numpy`, `scikit-learn` y `scipy`. Para que funcionen, se requiere instalar algunas dependencias no pythonicas:

* En Ubuntu: `sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran`

### Desde pypi

`pip install textar`

### Para desarrollo

```
git clone https://www.github.com/datosgobar/textar.git
cd path/to/textar
pip install -e .
```

Cualquier cambio en el código está disponible en el entorno virtual donde fue instalado de esta manera.

## Uso

### Búsqueda de textos similares

```python
from textar import TextClassifier

tc = TextClassifier(
    texts=[
        "El árbol del edificio moderno tiene manzanas",
        "El árbol más chico tiene muchas mandarinas naranjas, y está cerca del monumento antiguo",
        "El edificio más antiguo tiene muchos cuadros caros porque era de un multimillonario",
        "El edificio más moderno tiene muchas programadoras que comen manzanas durante el almuerzo grupal"
    ],
    ids=map(str, range(4))
)

ids, distancias, palabras_comunes = tc.get_similar(
    example="Me encontré muchas manzanas en el edificio", 
    max_similars=4
)

print ids
['0', '3', '2', '1']

print distancias
[0.92781458944579009, 1.0595805639371083, 1.1756638126839645, 1.3206413200640157]

print palabras_comunes
[[u'edificio', u'manzanas'], [u'edificio', u'muchas', u'manzanas'], [u'edificio', u'muchas'], [u'muchas']]
```

### Clasificación de textos

```python
from textar import TextClassifier

tc = TextClassifier(
    texts=[
        "Para hacer una pizza hace falta harina, tomate, queso y jamón",
        "Para hacer unas empanadas necesitamos tapas de empanadas, tomate, jamón y queso",
        "Para hacer un daiquiri necesitamos ron, una fruta y un poco de limón",
        "Para hacer un cuba libre necesitamos coca, ron y un poco de limón",
        "Para hacer una torta de naranja se necesita harina, huevos, leche, ralladura de naranja y polvo de hornear",
        "Para hacer un lemon pie se necesita crema, ralladura de limón, huevos, leche y harina"
    ],
    ids=map(str, range(6))
)

# entrena un clasificador
tc.make_classifier(
    name="recetas_classifier",
    ids=map(str, range(6)),
    labels=["Comida", "Comida", "Trago", "Trago", "Postre", "Postre"]
)

labels_considerados, puntajes = tc.classify(
    classifier_name="recetas_classifier", 
    examples=[
        "Para hacer un bizcochuelo de chocolate se necesita harina, huevos, leche y chocolate negro",
        "Para hacer un sanguche de miga necesitamos pan, jamón y queso"
    ]
)

print labels_considerados
array(['Comida', 'Postre', 'Trago'], dtype='|S6')

print puntajes
array([[-3.52493526,  5.85536809, -6.05497008],
       [ 2.801027  , -6.55619473, -3.39598721]])

# el primer ejemplo es un postre
print sorted(zip(puntajes[0], labels_considerados), reverse=True)
[(5.8553680868184079, 'Postre'),
 (-3.5249352611212568, 'Comida'),
 (-6.0549700786502845, 'Trago')]

# el segundo ejemplo es una comida
print sorted(zip(puntajes[1], labels_considerados), reverse=True)
[(2.8010269985828997, 'Comida'),
 (-3.3959872063363505, 'Trago'),
 (-6.5561947275785393, 'Postre')]
```

## Tests

Los tests sólo se pueden correr habiendo clonado el repo. Luego instalar las dependencias de desarrollo:

`pip install -r requirements_dev.txt`

y correr los tests:

`nosetests`

## Créditos

* [Victor Lavrenko](http://homepages.inf.ed.ac.uk/vlavrenk/) nos ayudó a entender el problema con sus explicaciones en youtube: https://www.youtube.com/user/victorlavrenko

## Contacto

Te invitamos a [crearnos un issue](https://github.com/datosgobar/textar/issues/new?title=Encontré un bug en textar) en caso de que encuentres algún bug o tengas feedback de alguna parte de `textar`.

Para todo lo demás, podés mandarnos tu comentario o consulta a [datos@modernizacion.gob.ar](mailto:datos@modernizacion.gob.ar).
