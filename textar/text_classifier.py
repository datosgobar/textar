# -*- coding: utf-8 -*-

u"""Módulo de clasificación de textos.

Este módulo contiene a los objetos que permiten entrenar un clasificador
automático de textos y pedir sugerencias de textos similares.
"""

from __future__ import unicode_literals

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import os
import warnings


class TextClassifier():

    u"""Clasificador automático de textos.

    Usa TF-IDF para vectorizar.
    Usa SVM para clasificar.
    """

    def __init__(self, texts, ids, vocabulary=None, encoding='utf-8'):
        """Definido en la declaracion de la clase.

        Attributes:
            texts (list of str): Textos a clasificar.
            ids (list of str): Identificadores únicos para cada texto (debe
                tener la misma longitud que `texts`).
            vocabulary (list): Opcional. Vocabulario a tener en cuenta para la
                vectorización de los textos. Default: usa todas las palabras
                presentes en los textos, salvo los ES_stopwords.txt.
            encoding (str): Codificación de los textos en `texts` y en `ids`.
        """
        this_dir, this_filename = os.path.split(__file__)
        es_stopwords = pd.read_csv(os.path.join(this_dir, 'ES_stopwords.txt'),
                                   header=None, encoding='utf-8')
        es_stopwords = list(np.squeeze(es_stopwords.values))
        self._check_id_length(ids)
        self.vectorizer = CountVectorizer(
            input='content', encoding=encoding, decode_error='strict',
            strip_accents='ascii', lowercase=True, preprocessor=None,
            tokenizer=None, stop_words=es_stopwords, ngram_range=(1, 1),
            analyzer='word', max_df=0.8, min_df=1, max_features=None,
            vocabulary=vocabulary, binary=False)

        self.transformer = TfidfTransformer()
        self.ids = None  # Matiene una lista ordenada de ids de textos.
        self.term_mat = None  # Matriz que cuenta los terminos en un texto.
        self.tfidf_mat = None  # Matriz de relevancia de los terminos.
        self.reload_texts(texts, ids)

    def __str__(self):
        """Representacion en str del objeto."""
        base_string = """Clasificador de textos con {:d} textos almacenados"""
        return base_string.format(len(self.ids))

    def make_classifier(self, name, ids, labels):
        """Entrenar un clasificador SVM sobre los textos cargados.

        Crea un clasificador que se guarda en el objeto bajo el nombre `name`.

        Args:
            name (str): Nombre para el clasidicador.
            ids (list): Se espera una lista de N ids de textos ya almacenados
                en el TextClassifier.
            labels (list): Se espera una lista de N etiquetas. Una por cada id
                de texto presente en ids.
        Nota:
            Usa el clasificador de `Scikit-learn <http://scikit-learn.org/>`_
        """
        if not all(np.in1d(ids, self.ids)):
            raise ValueError("Hay ids de textos que no se encuentran \
                              almacenados.")
        setattr(self, name, SGDClassifier())
        classifier = getattr(self, name)
        indices = np.searchsorted(self.ids, ids)
        classifier.fit(self.tfidf_mat[indices, :], labels)

    def retrain(self, name, ids, labels):
        """Reentrenar parcialmente un clasificador SVM.

        Args:
            name (str): Nombre para el clasidicador.
            ids (list): Se espera una lista de N ids de textos ya almacenados
                en el TextClassifier.
            labels (list): Se espera una lista de N etiquetas. Una por cada id
                de texto presente en ids.
        Nota:
            Usa el clasificador de `Scikit-learn <http://scikit-learn.org/>`_
        """
        if not all(np.in1d(ids, self.ids)):
            raise ValueError("Hay ids de textos que no se encuentran \
                              almacenados.")
        try:
            classifier = getattr(self, name)
        except AttributeError:
            raise AttributeError("No hay ningun clasificador con ese nombre.")
        indices = np.in1d(self.ids, ids)
        if isinstance(labels, str):
            labels = [labels]
        classifier.partial_fit(self.tfidf_mat[indices, :], labels)

    def classify(self, classifier_name, examples, max_labels=None,
                 goodness_of_fit=False):
        """Usar un clasificador SVM para etiquetar textos nuevos.

        Args:
            classifier_name (str): Nombre del clasidicador a usar.
            examples (list or str): Se espera un ejemplo o una lista de
                ejemplos a clasificar en texto plano o en ids.
            max_labels (int, optional): Cantidad de etiquetas a devolver para
                cada ejemplo. Si se devuelve mas de una el orden corresponde a
                la plausibilidad de cada etiqueta. Si es None devuelve todas
                las etiquetas posibles.
            goodness_of_fit (bool, optional): Indica si devuelve o no una
                medida de cuan buenas son las etiquetas.
        Nota:
            Usa el clasificador de `Scikit-learn <http://scikit-learn.org/>`_

        Returns:
            tuple (array, array): (labels_considerados, puntajes)
                labels_considerados: Las etiquetas que se consideraron para
                    clasificar.
                puntajes: Cuanto más alto el puntaje, más probable es que la
                    etiqueta considerada sea la adecuada.
        """
        classifier = getattr(self, classifier_name)
        texts_vectors = self._make_text_vectors(examples)
        return classifier.classes_, classifier.decision_function(texts_vectors)

    def _make_text_vectors(self, examples):
        """Funcion para generar los vectores tf-idf de una lista de textos.

        Args:
            examples (list or str): Se espera un ejemplo o una lista de:
                o bien ids, o bien textos.
        Returns:
            textvec (sparse matrix): Devuelve una matriz sparse que contiene
                los vectores TF-IDF para los ejemplos que se pasan de entrada.
                El tamaño de la matriz es de (N, T) donde N es la cantidad de
                ejemplos y T es la cantidad de términos en el vocabulario.
        """
        if isinstance(examples, str):
            if examples in self.ids:
                textvec = self.tfidf_mat[self.ids == examples, :]
            else:
                textvec = self.vectorizer.transform([examples])
                textvec = self.transformer.transform(textvec)
        elif type(examples) is list:
            if all(np.in1d(examples, self.ids)):
                textvec = self.tfidf_mat[np.in1d(self.ids, examples)]
            elif not any(np.in1d(examples, self.ids)):
                textvec = self.vectorizer.transform(examples)
                textvec = self.transformer.transform(textvec)
            else:
                raise ValueError("Las listas de ejemplos deben ser todos ids\
                                  de textos almacenados o todos textos planos")
        else:
            raise TypeError("Los ejemplos no son del tipo de dato adecuado.")

        return textvec

    def get_similar(self, example, max_similars=3, similarity_cutoff=None,
                    term_diff_max_rank=10, filter_list=None,
                    term_diff_cutoff=None):
        """Devuelve textos similares al ejemplo dentro de los textos entrenados.

        Nota:
            Usa la distancia de coseno del vector de features TF-IDF

        Args:
            example (str): Se espera un id de texto o un texto a partir del
                cual se buscaran otros textos similares.
            max_similars (int, optional): Cantidad de textos similares a
                devolver.
            similarity_cutoff (float, optional): Valor umbral de similaridad
                para definir que dos textos son similares entre si.
            term_diff_max_rank (int, optional): Este valor sirve para controlar
                el umbral con el que los terminos son considerados importantes
                a la hora de recuperar textos (no afecta el funcionamiento de
                que textos se consideran cercanos, solo la cantidad de terminos
                que se devuelven en best_words).
            filter_list (list): Lista de ids de textos en la cual buscar textos
                similares.
            term_diff_cutoff (float): Deprecado. Se quitara en el futuro.

        Returns:
            tuple (list, list, list): (text_ids, sorted_dist, best_words)
                text_ids (list of str): Devuelve los ids de los textos
                    sugeridos.
                sorted_dist (list of float): Devuelve la distancia entre las
                    opciones sugeridas y el ejemplo dado como entrada.
                best_words (list of list): Para cada sugerencia devuelve las
                    palabras mas relevantes que se usaron para seleccionar esa
                    sugerencia.
        """

        if term_diff_cutoff:
            warnings.warn('Deprecado. Quedo sin uso. Se quitara en el futuro.',
                          DeprecationWarning)
        if filter_list:
            if max_similars > len(filter_list):
                raise ValueError("No se pueden pedir mas sugerencias que la \
                                  cantidad de textos en `filter_list`.")
            else:
                filt_idx = np.in1d(self.ids, filter_list)

        elif max_similars > self.term_mat.shape[0]:
            raise ValueError("No se pueden pedir mas sugerencias que la \
                              cantidad de textos que hay almacenados.")
        else:
            filt_idx = np.ones(len(self.ids), dtype=bool)
        # Saco los textos compuestos solo por stop_words
        good_ids = np.array(np.sum(self.term_mat, 1) > 0).squeeze()
        filt_idx = filt_idx & good_ids
        filt_idx_to_general_idx = np.flatnonzero(filt_idx)
        if example in self.ids:
            index = self.ids == example
            exmpl_vec = self.tfidf_mat[index, :]
            distances = np.squeeze(pairwise_distances(self.tfidf_mat[filt_idx],
                                                      exmpl_vec))
            # Pongo la distancia a si mismo como inf, par que no se devuelva a
            # si mismo como una opcion
            if filter_list and example in filter_list:
                distances[filter_list.index(example)] = np.inf
            elif not filter_list:
                idx_example = np.searchsorted(self.ids, example)
                filt_idx_example = np.searchsorted(np.flatnonzero(filt_idx),
                                                   idx_example)
                distances[filt_idx_example] = np.inf
        else:
            exmpl_vec = self.vectorizer.transform([example])  # contar terminos
            exmpl_vec = self.transformer.transform(exmpl_vec)  # calcular tfidf
            distances = np.squeeze(pairwise_distances(self.tfidf_mat[filt_idx],
                                                      exmpl_vec))
        if np.sum(exmpl_vec) == 0:
            return [], [], []
        sorted_indices = np.argsort(distances)
        closest_n = sorted_indices[:max_similars]
        sorted_dist = distances[closest_n]
        if similarity_cutoff:
            closest_n = closest_n[sorted_dist < similarity_cutoff]
            sorted_dist = sorted_dist[sorted_dist < similarity_cutoff]
        best_words = []

        # Calculo palabras relevantes para cada sugerencia
        best_example = np.squeeze(exmpl_vec.toarray())
        sorted_example_weights = np.flipud(np.argsort(best_example))
        truncated_max_rank = min(term_diff_max_rank, np.sum(best_example > 0))
        best_example_words = sorted_example_weights[:truncated_max_rank]
        for suggested in closest_n:
            suggested_idx = filt_idx_to_general_idx[suggested]
            test_vec = np.squeeze(self.tfidf_mat[suggested_idx, :].toarray())
            sorted_test_weights = np.flipud(np.argsort(test_vec))
            truncated_max_rank = min(term_diff_max_rank,
                                     np.sum(test_vec > 0))
            best_test = sorted_test_weights[:truncated_max_rank]
            best_words_ids = np.intersect1d(best_example_words, best_test)
            best_words.append([k for k, v in
                               self.vectorizer.vocabulary_.items()
                               if v in best_words_ids])

        # Filtro dentro de las buscadas
        if filter_list:
            text_ids = self.ids[filt_idx_to_general_idx[closest_n]]
        else:
            text_ids = self.ids[closest_n]
        return list(text_ids), list(sorted_dist), best_words

    def reload_texts(self, texts, ids, vocabulary=None):
        """Calcula los vectores de terminos de textos y los almacena.

        A diferencia de :func:`~TextClassifier.TextClassifier.store_text` esta
        funcion borra cualquier informacion almacenada y comienza el conteo
        desde cero. Se usa para redefinir el vocabulario sobre el que se
        construyen los vectores.

        Args:
            texts (list): Una lista de N textos a incorporar.
            ids (list): Una lista de N ids alfanumericos para los textos.
        """
        self._check_id_length(ids)
        self.ids = np.array(sorted(ids))
        if vocabulary:
            self.vectorizer.vocabulary = vocabulary
        sorted_texts = [x for (y, x) in sorted(zip(ids, texts))]
        self.term_mat = self.vectorizer.fit_transform(sorted_texts)
        self._update_tfidf()

    # NO ENCUENTRO UNA MANERA EFICIENTE DE HACER ESTO POR AHORA NO HACE FALTA
    # def store_text(self, texts, ids, replace_texts=False):
    #     """Calcula los vectores de terminos de un texto y los almacena.
    #         NOT IMPLEMENTED.
    #     Nota:
    #         Esta funcion usa el vocabulario que ya esta almacenado, es decir,
    #         que no se incorporan nuevos terminos. Si se quiere cambiar el
    #         vocabulario deben recargarse todos los textos con
    #         :func:`~TextClassifier.TextClassifier.reload_texts`
    #     Args:
    #         texts (list): Una lista de N textos a incorporar.
    #         ids (list of str): Una lista de N ids alfanumericos para los textos
    #         replace_texts (bool, optional): Indica si deben reemplazarse los
    #             textos cuyo id ya este almacenado. Si es False y algun id ya se
    #             encuentra almacenado se considera un error.
    #     """
    #     self._check_id_length(ids)
    #     if not replace_texts and any(np.in1d(ids, self.ids)):
    #         raise ValueError("Alguno de los ids provistos ya esta en el \
    #                           indice")
    #     else:
    #         ids = np.array(ids)
    #         partial_mat = self.vectorizer.transform(texts)
    #         # Si no hay ids ya guardados solo concateno y los agrego al
    #         # array self.ids
    #         if not any(np.in1d(ids, self.ids)):
    #             self.ids = np.r_[self.ids, ids]
    #             self.term_mat = sparse.vstack((self.term_mat,
    #                                            partial_mat))
    #         # Si los hay,
    #         else:
    #             oldrows = np.in1d(self.ids, ids)
    #             oldpartial = np.in1d(ids, self.ids)
    #             # Actualizo las filas que ya estaban
    #             self.term_mat[oldrows, :] = partial_mat[oldpartial, :]
    #             # y agrego las que no
    #             partial_mat = partial_mat[~oldpartial, :]
    #             self.term_mat = sparse.vstack((self.term_mat, partial_mat))
    #             # concateno los viejos ids y los nuevos
    #             self.ids = np.r_[self.ids, ids[~oldpartial]]
    #     self._update_tfidf()

    def _update_tfidf(self):
        self.tfidf_mat = self.transformer.fit_transform(self.term_mat)

    def _check_id_length(self, ids):
        if any(map(lambda x: len(x) > 10, ids)):
            warnings.warn("Hay ids que son muy largos. Es posible que se hayan \
            ingresado textos planos en lugar de ids.")

    def _check_repeated_ids(self, ids):
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Hay ids repetidos.")
