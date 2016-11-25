"""Tests del modulo text_classifier."""
import unittest
import sys
import os
import codecs
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_20newsgroups
sys.path.insert(0, os.path.abspath('..'))
from textar import TextClassifier


class TestTextClassifier(unittest.TestCase):

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
