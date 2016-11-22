"""Tests del modulo text_classifier."""
import unittest
import sys
import os
import codecs
import numpy as np
from sklearn.cross_validation import train_test_split
sys.path.insert(0, os.path.abspath('..'))
from text_classifier import TextClassifier


class TestTextClassifier(unittest.TestCase):

    """Clase de tests del objeto TextClassifier."""

    def setUp(self):
        """Carga de los datos de prueba (20 Newsgroups corpus)."""
        TEST_DIR = "/home/mec/testData/textos/20_newsgroups/"
        cats = os.listdir(TEST_DIR)
        cats_totales = []
        contenidos = []
        for cat in cats:
            temp_list = os.listdir(os.path.join(TEST_DIR, cat))
            cats_totales = cats_totales + [cat for i in range(len(temp_list))]
            temp_list = map(lambda x: os.path.join(TEST_DIR, cat, x),
                            temp_list)
            for filename in temp_list:
                with codecs.open(
                        filename, encoding='latin1', mode='r') as content_file:
                    lines = content_file.readlines()
                    filtered_lines = [line for line in lines[11:] if line[0] not in ['$', '>']]
                    content = ''.join(filtered_lines)
                contenidos.append(content)
        # sacar los textos muy cortos
        largo = map(len, contenidos)
        contenidos = [c for c, l in zip(contenidos, largo) if l > 100]
        labels = cats_totales
        labels = [lab for lab, l in zip(cats_totales, largo) if l > 100]
        self.ids = [str(i) for i in range(len(contenidos))]
        self.texts = contenidos
        self.labels = labels
        self.tc = TextClassifier(self.texts, self.ids, encoding='latin1')

    def test_reload_texts(self):
        # TODO aca van los tests!
        self.assertEqual(12, 12)

    def test_classifier_performance(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.tc.tfidf_mat, self.labels, test_size=0.33, random_state=42)
        self.tc.make_classifier("prueba", X_train, y_train)
        clasificador = getattr(self.tc, "prueba")
        my_score = clasificador.score(X_test, y_test)
        self.assertGreater(my_score, 0.8)

if __name__ == '__main__':
    unittest.main()
