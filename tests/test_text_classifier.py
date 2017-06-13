#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests del modulo text_classifier."""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

import unittest
import sys
import os
import codecs
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
sys.path.insert(0, os.path.abspath('..'))
from textar import TextClassifier


# @unittest.skip("skip")
class TextClassifierTestCase(unittest.TestCase):
    """Tests unitarios del objeto TextClassifier."""

    def test_get_similar(self):
        """Testea la búsqueda de textos similares."""

        tc = TextClassifier(
            texts=[
                "El árbol del edificio moderno tiene manzanas",
                "El árbol más chico tiene muchas mandarinas naranjas, y está cerca del monumento antiguo",
                "El edificio más antiguo tiene muchas cuadros caros porque era de un multimillonario",
                "El edificio más moderno tiene muchas programadoras que comen manzanas durante el almuerzo grupal"
            ],
            ids=list(map(str, range(4)))
        )

        ids, distancias, palabras_comunes = tc.get_similar(
            example="Me encontré muchas manzanas en el edificio",
            max_similars=4
        )

        self.assertEqual(ids, ['0', '3', '2', '1'])
        self.assertEqual(
            [
                sorted(palabras)
                for palabras in palabras_comunes
            ]
            ,
            [
                [u'edificio', u'manzanas'],
                [u'edificio', u'manzanas', u'muchas'],
                [u'edificio', u'muchas'], [u'muchas']
            ]
        )

    def test_classify(self):
        tc = TextClassifier(
            texts=[
                "Para hacer una pizza hace falta harina, tomate, queso y jamón",
                "Para hacer unas empanadas necesitamos tapas de empanadas, tomate, jamón y queso",
                "Para hacer un daiquiri necesitamos ron, una fruta y un poco de limón",
                "Para hacer un cuba libre necesitamos coca, ron y un poco de limón",
                "Para hacer una torta de naranja se necesita harina, huevos, leche, ralladura de naranja y polvo de hornear",
                "Para hacer un lemon pie se necesita crema, ralladura de limón, huevos, leche y harina"
            ],
            ids=list(map(str, range(6)))
        )

        # entrena un clasificador
        tc.make_classifier(
            name="recetas_classifier",
            ids=list(map(str, range(6))),
            labels=["Comida", "Comida", "Trago", "Trago", "Postre", "Postre"]
        )

        labels_considerados, puntajes = tc.classify(
            classifier_name="recetas_classifier",
            examples=[
                "Para hacer un bizcochuelo de chocolate se necesita harina, huevos, leche y chocolate negro",
                "Para hacer un sanguche de miga necesitamos pan, jamón y queso"
            ]
        )

        sorted_tuples = sorted(zip(puntajes[0], labels_considerados),
                               reverse=True)
        self.assertEqual(sorted_tuples[0][1], "Postre")

        sorted_tuples = sorted(zip(puntajes[1], labels_considerados),
                               reverse=True)
        self.assertEqual(sorted_tuples[0][1], "Comida")


# @unittest.skip("skip")
class TextClassifierPerformanceTestCase(unittest.TestCase):

    """Clase de tests del objeto TextClassifier."""

    def setUp(self):
        """Carga de los datos de prueba (20 Newsgroups corpus)."""
        newsdata = fetch_20newsgroups(data_home="./data/")
        self.ids = [str(i) for i in range(len(newsdata.target))]
        self.texts = newsdata.data
        self.labels = [newsdata.target_names[idx] for idx in newsdata.target]
        self.tc = TextClassifier(self.texts, self.ids)

    def test_reload_texts(self):
        # TODO aca van los tests!
        pass

    def test_classifier_performance(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.ids, self.labels, test_size=0.33, random_state=42)
        self.tc.make_classifier("prueba", X_train, y_train)
        clasificador = getattr(self.tc, "prueba")
        indices = np.searchsorted(self.tc.ids, X_test)
        my_score = clasificador.score(self.tc.tfidf_mat[indices, :], y_test)
        self.assertGreater(my_score, 0.8)


if __name__ == '__main__':
    unittest.main()
