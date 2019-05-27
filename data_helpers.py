from __future__ import division
import numpy as np
import re, json
import itertools
from collections import Counter
from reader import TSVReader
import cPickle
from embedding import Word2Vec
# from entity_util import GetEntities
from time import time
from copy import deepcopy
from collections import defaultdict
import time, json
from collections import Counter
import collections, pickle

################################################################################################
def clean_str(string):
    # string = "{C And then } the same in Vietnam, [ you would, + you wouldn't ] handle Vietnam the same way, you would handle, {F uh, } Saddam Houssein."
    cleaned_text = re.sub(r'<[^<]+?>', '', string)
    cleaned_text = re.sub(r'[a-z]*[:.]+\S+', '', cleaned_text)
    cleaned_text = " ".join(re.sub("[^A-Za-z0-9, ']+", ' ', cleaned_text).split())
    cleaned_text = cleaned_text.replace(',', '').lower()
    cleaned_text = ' '.join([w for w in cleaned_text.split() if len(w)>1 or w == 'i' or w == 'a'])

    return cleaned_text

################################################################################################
def extracting_structral_features(text, unpolished_text, label):
    num_words = len(text.split())
    num_sentence = len(unpolished_text.split('.'))
    num_question =  1 if len(unpolished_text.split('?')) > 1 else 0
    num_sharp = len(unpolished_text.split('#'))
    num_chars = len(text)
    char_rate = num_chars/(num_words+1)

    # wh_list = ["what", "what", "who", "why", "when","where", "which", "how long", "how far"]
    # wh_bool = 0
    # for tx in wh_list:
    #     if tx in text and num_question == 1:
    #         # print unpolished_text, '  and label is :  ', label
    #         wh_bool = 1
    #
    # do_list = ["do ", "does"]
    # do_bool = 0
    # for tx in do_list:
    #     if tx in text and num_question == 1:
    #         # print unpolished_text, '  and label is :  ', label
    #         do_bool = 1

    might_qw = 0
    # if num_question == 1 and wh_bool == 1:
    #     might_qw = 1
    #
    # b_list = ["right", "yeah", 'yes']
    # b_bool = 0
    # for tx in b_list:
    #     if tx in text:
    #         print unpolished_text, '  and label is :  ', label
    #         b_bool = 1
    #
    # might_bh = 0
    # if num_question > 0 and b_bool == 1:
    #    might_bh = 1

    # info = inteInfo.inteInfo(text)
    # polarity_subjectivity = info["sentiment"]
    features = [num_words, num_sentence, num_question, num_chars, char_rate]

    return features

################################################################################################
def load_data_and_labels(train_file, test_file):
    # with open(train_file, 'r') as handel:
    #     dialogue_dataset = json.load(handel)

    dialogue_dataset = []
    dialogue_dataset_train = open(train_file).read().split('\n')
    print len(dialogue_dataset_train)

    dialogue_dataset.extend(dialogue_dataset_train)
    dialogue_dataset_test = open(test_file).read().split('\n')
    print len(dialogue_dataset_test)
    dialogue_dataset.extend(dialogue_dataset_test)

    #preparing data for training
    num_classes = 42
    state_size = 43
    speakerID_size = 609

    label_pred = []

    st2_features = []
    st1_features = []

    ib2_features = []
    ib1_features = []

    speakerid3_features = []
    speakerid2_features = []
    speakerid1_features = []
    speakerid_pred_features = []

    cl2_features = []
    cl1_features = []
    cl0_features = []

    data_pred = []
    data_utt3 = []
    data_utt2 = []
    data_utt1 = []

    data_features_pred = []
    data_features_utt1 = []
    data_features_utt2 = []
    data_features_utt3 = []

    data_ib1 = []
    data_ib2 = []


    data_char_pred = []
    data_char1 = []
    data_char2 = []

    data_pos1 = []
    data_pos2 = []
    data_pos3 = []
    data_pos_pred = []

    data_hub1 = []
    data_hub2 = []
    data_hub_pred = []

    data_topic1 = []
    data_topic2 = []
    data_topic3 = []
    data_topic_pred = []

    class_label = dict()
    classl = -1

    speakerID_label = dict()
    spi = -1

    ib_label = dict()
    ibl = -1

    with open('/Users/aliahmadvand/Desktop/Dialogue_Act/Contextual_DA/swda-master/label_list.pickle', 'rb') as f:
        class_lbl = pickle.load(f)

    class_lbl.remove('+')
    with open('/Users/aliahmadvand/Desktop/Dialogue_Act/Contextual_DA/swda-master/pos_list.pickle', 'rb') as f:
        pos_lbl = pickle.load(f)

    pos_dict = {'CC': 1, 'CD': 2, 'DT': 3, 'EX': 4, 'FW': 5, 'IN': 6, 'JJ': 7, 'JJR': 8, 'JJS': 9, \
                'LS': 10, 'MD': 11, 'NN': 12, 'NNS': 13, 'NNP': 14, 'NNPS': 15, 'PDT': 16, 'POS': 17, 'PRP': 18,
                'PRP$': 19, 'RB': 20, 'RBR': 21, 'RBS': 22, 'RP': 23, 'SYM': 24, 'TO': 25, 'UH': 26, 'VB': 27, 'VBD': 28, 'VBG': 29, \
                'VBN': 30, 'VBP': 31, 'VBZ': 32, 'WDT': 33, 'WP': 34, 'WP$': 35, 'WRB': 36, ',': 37}


    with open('/Users/aliahmadvand/Desktop/Dialogue_Act/Contextual_DA/swda-master/topic_list.pickle', 'rb') as f:
        topic_lbl = pickle.load(f)

    topic_dict = dict()
    for i, topic in enumerate(topic_lbl):
        topic_dict[topic] = i + 1


    cl_label = dict()
    cll = -1
    for i, label in enumerate(class_lbl):
        cl_label[label] = i

    vocabulary = set()
    all_states = []
    all_distinct_states = []

    train_dev_index = 0
    sample_count = 0
    num_conversation = 0
    for iter, sample1 in enumerate(dialogue_dataset):
        num_conversation += 1
        if iter == len(dialogue_dataset_train):
            train_dev_index = sample_count
            print "size of training set is:  {}".format(train_dev_index)


        if len(sample1) > 0:

            sample = json.loads(sample1)
            pre_label = (sample['label'][-1])
            label = pre_label


            if len(label) > 0 :

                    sample_count += 1
                    all_distinct_states.append(sample['label'][-1])
                    all_states.append(sample['label'][-1])
                    all_states.append(sample['label'][2])
                    all_states.append(sample['label'][1])
                    all_states.append(sample['label'][0])

                    utt_pred = clean_str(sample['utt'][-1].lower())
                    data_pred.append(utt_pred)
                    utt_1 = clean_str(sample['utt'][2].lower())
                    data_utt1.append(utt_1)
                    utt_2 = clean_str(sample['utt'][1].lower())
                    data_utt2.append(utt_2)
                    utt_3 = clean_str(sample['utt'][0].lower())
                    data_utt3.append(utt_3)


                    data_features_pred.append(extracting_structral_features(utt_pred, sample['utt'][-1].lower(), label))
                    data_features_utt1.append(extracting_structral_features(utt_1, sample['utt'][2].lower(),label))
                    data_features_utt2.append(extracting_structral_features(utt_2, sample['utt'][1].lower(), label))
                    data_features_utt3.append(extracting_structral_features(utt_3, sample['utt'][0].lower(), label))


                    vocabulary.update(utt_pred.lower().split())

                    data_ib1.append((sample['label'][1]))
                    data_ib2.append((sample['label'][0]))


                    data_char_pred.append(load_char_text(utt_pred, 550))
                    data_char1.append(load_char_text(utt_1, 550))
                    data_char2.append(load_char_text(utt_2, 550))

                    data_pos_pred.append(getting_extra_features(sample['pos'][-1], pos_dict))
                    data_pos1.append(getting_extra_features(sample['pos'][1], pos_dict))
                    data_pos2.append(getting_extra_features(sample['pos'][0], pos_dict))
                    data_pos3.append(getting_extra_features(sample['pos'][0], pos_dict))

                    data_topic_pred.append(getting_topic_features(sample['main_topics'][-1], topic_dict))
                    data_topic1.append(getting_topic_features(sample['main_topics'][2], topic_dict))
                    data_topic2.append(getting_topic_features(sample['main_topics'][1], topic_dict))
                    data_topic3.append(getting_topic_features(sample['main_topics'][0], topic_dict))

                    # data_hub_pred.append(char2float(sample['hub'][-1]))
                    # data_hub1.append(char2float(sample['hub'][1]))
                    # data_hub2.append(char2float(sample['hub'][0]))

                    data_hub_pred.append(np.zeros(128))
                    data_hub1.append(np.zeros(128))
                    data_hub2.append(np.zeros(128))


                    speakerid_pred = sample['speakerid'][-1]
                    speakerid1 = sample['speakerid'][2]
                    speakerid2 = sample['speakerid'][1]
                    speakerid3 = sample['speakerid'][0]

                    #######################################################
                    one_hot = np.zeros(num_classes)
                    if label not in class_label.keys():
                        classl = classl + 1
                        class_label[label] = classl

                    one_hot[class_label[label]] = 1
                    label_pred.append(one_hot)

                    #######################################################
                    one_hot = np.zeros(state_size)
                    if data_ib1[-1] not in ib_label.keys():
                        ibl = ibl + 1
                        ib_label[data_ib1[-1]] = ibl
                        # print ibl

                    one_hot[ib_label[data_ib1[-1]]] = 1
                    ib1_features.append(one_hot)

                    #######################################################
                    one_hot = np.zeros(state_size)
                    if data_ib2[-1] not in ib_label.keys():
                        ibl = ibl + 1
                        ib_label[data_ib2[-1]] = ibl
                        # print ibl

                    one_hot[ib_label[data_ib2[-1]]] = 1
                    ib2_features.append(one_hot)

                    #######################################################
                    one_hot = np.zeros(speakerID_size)
                    if speakerid_pred not in speakerID_label.keys():
                        spi = spi + 1
                        speakerID_label[speakerid_pred] = spi
                        # print spi

                    one_hot[speakerID_label[speakerid_pred]] = 1
                    speakerid_pred_features.append(one_hot)

                    #######################################################
                    one_hot = np.zeros(speakerID_size)
                    if speakerid1 not in speakerID_label.keys():
                        spi = spi + 1
                        speakerID_label[speakerid1] = spi
                        # print spi

                    one_hot[speakerID_label[speakerid1]] = 1
                    speakerid1_features.append(one_hot)

                    #######################################################
                    one_hot = np.zeros(speakerID_size)
                    if speakerid2 not in speakerID_label.keys():
                        spi = spi + 1
                        speakerID_label[speakerid2] = spi
                        # print spi

                    one_hot[speakerID_label[speakerid2]] = 1
                    speakerid2_features.append(one_hot)

                #######################################################
                    one_hot = np.zeros(speakerID_size)
                    if speakerid3 not in speakerID_label.keys():
                        spi = spi + 1
                        speakerID_label[speakerid3] = spi
                        # print spi

                    one_hot[speakerID_label[speakerid3]] = 1
                    speakerid3_features.append(one_hot)

            #######################################################
            else:
                print "...."

        else:
            print "...."


    print "train_dev_index: ", train_dev_index
    print sample_count,  "num conversations: ", num_conversation
    counter = collections.Counter(all_distinct_states)
    # print len(all_states)
    print(counter.values())
    print(counter.keys())
    # print(counter.most_common(len(label_list)))

    Data = [data_pred, data_utt1, data_utt2, data_utt3]
    features_Data = [data_features_pred, data_features_utt1, data_features_utt2, data_features_utt3]
    cl_Data = [data_pos_pred, data_pos1, data_pos2, data_pos3]
    topic_Data = [data_topic_pred, data_topic1, data_topic2, data_topic3]
    char_Data = [data_char_pred, data_char1, data_char2]
    ib_Data = [ib1_features, ib2_features]
    speakerID_Data = [speakerid_pred_features, speakerid1_features, speakerid2_features, speakerid3_features]
    hub_Data = [data_hub_pred, data_hub1, data_hub2]


    class_order = dict()
    for label in class_label:
        class_order[class_label[label]] = label

    print class_order
    file_classorder = open('./auxiliary_files/class_order.json', 'w')
    json.dump(class_order, file_classorder)


    handcraft_train = []
    for i, sample in enumerate(data_pred):
        G = np.ones(1)

        handcraft_train.append(G)


    with open('./auxiliary_files/vocabulary.pkl', 'wb') as handle:
        cPickle.dump(vocabulary, handle)
    #

    print 'write dict finishes.....'

    w2v = Word2Vec('./GoogleNews-vectors-negative300.bin', vocabulary)

    count = 0
    vocab = {}
    for item in vocabulary:
        vocab[item] = count
        count += 1


    print "Data: ", np.array(Data).shape

    return Data, char_Data, ib_Data, cl_Data, topic_Data, features_Data, speakerID_Data, hub_Data, np.array(label_pred), np.array(handcraft_train), vocab, w2v, ent_dict, train_dev_index-1, data_pred, class_label

################################################################################################
# Assign onehot vector for every word and entity in the dictionary
def get_entity_vector(orig_txt, new_text, entities):

    ent_vector = np.zeros(len(orig_txt.split()))
    for ent in entities.keys():
        for i, word in enumerate(new_text.split()):
            new_entity = '_'.join(ent.split())
            if word == new_entity:
                type = entities[ent][0]
                # print entities[ent]
                words = new_entity.split('_')
                for j, wd in enumerate(words):
                    ent_vector[i+j] = ent_dict[type]
                break

    return list(ent_vector)

################################################################################################
def get_onehot_entity_vector(orig_txt, new_text, entities):
        ent_vector = np.zeros(len(Types.keys()))
        for ent in entities.keys():
            for i, word in enumerate(new_text.split()):
                new_entity = '_'.join(ent.split())
                if word == new_entity:
                    for typ in entities[ent]:
                        ent_vector[ent_dict[typ]] = 1

        return ent_vector


################################################################################################
def generate_onehot_word_vector():
    features = defaultdict(list)
    for ent in entity_vector:
        for type in entity_vector[ent]:
            # print type
            features[ent].append(entity_vector[ent][type])

    return features

def generate_onehot_vector(dataset_onehot_dict):
    features = []
    for sample in dataset_onehot_dict:
        vector = []
        for type in sample:
            vector.append(sample[type])

        features.append(vector)

    return features


################################################################################################
def reset_type(types):
    for key in types.keys():
        types[key] = 0

    return types


################################################################################################
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



################################################################################################
def fit_transform(train_file, vocab_processor, vocab):
    count = 0
    for iter, line in enumerate(train_file):
        split_line = line.strip().split('\t')
        text = ' '.join(split_line[0:])
        for j, word in enumerate(text.split()):
            try:
                vocab_processor[iter][j] = vocab[word]
            except:
                pass

    return vocab_processor


def getting_extra_features(word_pos, pos_dict):
    vector_pos = []
    for pos in word_pos:
        try:
           vector_pos.append(pos_dict[pos])
        except:
            vector_pos.append(0)

    return vector_pos


def getting_topic_features(topic, topic_dict):
    one_hot = np.zeros(len(topic_dict))
    try:
       one_hot[topic_dict[topic]] = 1
    except:
        pass

    return one_hot


def fit_transform_pos(extra_pos_features, vocab_processor_pos):
    for iter1, line in enumerate(extra_pos_features):
        for j, sample in enumerate(line):
            try:
                vocab_processor_pos[iter1][j] = sample
            except:
                pass




    return vocab_processor_pos
