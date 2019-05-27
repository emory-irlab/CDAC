import sys
import numpy as np
from scipy import spatial
import pickle as pickle
# from importlib import reload

# reload(sys)

# sys.setdefaultencoding('UTF-8')


class Word2Vec:
    def __init__(self, word2vec_filename, vocabulary):
        if word2vec_filename is None:
            return
        
        self.filename = word2vec_filename
        self.vocabulary = vocabulary
        self.word_vectors = None
        self._load_base_file()
        self._add_unknown_words()
        # self._modify_entity_vector()

    def _load_base_file(self):
        word_vectors = dict()
        print ("opening file: %s" % self.filename)

        with open(self.filename, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            print("vocab size: %d, layer1_size: %d" % (vocab_size, layer1_size))

            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word)
                        break
                    if ch != b'\n':
                        word.append(ch)

                # I made every word lower case
                word_vectors[word.lower()] = np.fromstring(f.read(binary_len), dtype='float32')

        print("num words already in word2vec: %d" % len(word_vectors))
        print("vocabulary size: %d" % len(self.vocabulary))
        self.word_vectors = word_vectors

    def get_word_vector(self, word):
        try:
            return self.word_vectors[word]
        except KeyError:
            return None

    def _add_unknown_words(self):
        not_present = 0
        for word in self.vocabulary:
            if word not in self.word_vectors:
                self.word_vectors[word] = np.random.uniform(-0.25, 0.25, 300)
                not_present += 1
        print ('randomized words: %d out of %d' % (not_present, len(self.vocabulary)))


    def _modify_entity_vector(self):
        complex_entity = 0
        for word in self.vocabulary:
            tokens = word.split('_')
            if len(tokens) > 1:
                complex_entity += 1
                self.word_vectors[word] = np.zeros(300)
                for token in tokens:
                    self.word_vectors[word] += self.word_vectors[token]

        print ('complex entities: %d out of %d' % (complex_entity, len(self.vocabulary)))

    def get_similar_words(self, word, n=5):
        similarity_list = []

        if type(word) is not np.ndarray:
            word = self.word_vectors[word]

        for idx, w in enumerate(self.word_vectors):
            similarity_list.append((w, 1 - spatial.distance.cosine(self.word_vectors[w], word)))

        return (sorted(similarity_list, key=lambda x: x[1], reverse=True))[:n]

    def get_dissimilar_words(self, word, n=5):
        similarity_list = []

        if type(word) is not np.ndarray:
            word = self.word_vectors[word]

        for idx, w in enumerate(self.word_vectors):
            similarity_list.append((w, 1 - spatial.distance.cosine(self.word_vectors[w], word)))

        return (sorted(similarity_list, key=lambda x: x[1]))[:n]

    def load_model(self, file_path):
        pickle_obj = pickle.load(open(file_path))
        print ('loaded %d vectors from %s' % (len(pickle_obj[1]), file_path))
        self.vocabulary = pickle_obj[0]
        self.word_vectors = pickle_obj[1]

    def save_model(self, file_path):
        print ('dumping %d vectors into %s' % (len(self.word_vectors), file_path))
        pickle.dump([self.vocabulary, self.word_vectors], open(file_path, 'w'), protocol=2)