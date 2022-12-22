__author__ = "Matlatipov Sanatbek, Jaloliddin Rajabov"
__credits__ = ""
__license__ = ""
__version__ = ""
__maintainer__ = ""
__email__ = "{s.matlatipov, j.rajabov}@nuu.uz"

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
    import krippendorff as kp
    from sklearn.metrics import confusion_matrix, cohen_kappa_score
    import pandas as pd
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fd(counts):
    """Given a list of occurrences (e.g., [1,1,1,2]), return a dictionary of frequencies (e.g., {1:3, 2:1}.)"""
    d = {}
    for i in counts:
        d[i] = d[i] + 1 if i in d else 1
    return d


frequency_rank = lambda d: sorted(d, key=d.get, reverse=True)
'Given a map, return ranked the keys based on their values.'


class Category:
    """Category objects contain the term and polarity (i.e., pos, neg, neu, conflict) of the category (e.g., food,
    price, etc.) of a sentence. """

    def __init__(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity

    def create(self, element):
        self.term = element.attrib['category']
        self.polarity = element.attrib['polarity']
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Aspect:
    """Aspect objects contain the term (e.g., battery life) and polarity (i.e., positive, negative, neutral, conflict)
    of an aspect. """

    def __init__(self, term, polarity, offsets):
        self.term = term
        self.polarity = polarity
        self.offsets = offsets

    def create(self, element):
        self.term = element.attrib['term']
        self.polarity = element.attrib['polarity']
        self.offsets = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


def validate(filename):
    """Validate an XML file"""
    elements = ET.parse(filename).getroot().findall('sentence')
    aspects = []
    for e in elements:
        for eterms in e.findall('aspectTerms'):
            if eterms is not None:
                for a in eterms.findall('aspectTerm'):
                    aspects.append(Aspect('', '', []).create(a).term)
    return elements, aspects


fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'Simple fix for writing out text.'


class Instance:
    """An instance is a sentence, modeled out of XML (pre-specified format, based on the 4th task of SemEval 2014).
    It contains the text, the aspect terms, and any aspect categories."""

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('ID')
        self.aspect_terms = [Aspect('', '', offsets={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [Category(term='', polarity='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]

    def get_aspect_terms(self):
        return [a.term.lower() for a in self.aspect_terms]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_term(self, term, polarity='', offsets=None):
        if offsets is None:
            offsets = {'from': '', 'to': ''}
        a = Aspect(term, polarity, offsets)
        self.aspect_terms.append(a)

    def add_aspect_category(self, term, polarity=''):
        c = Category(term, polarity)
        self.aspect_categories.append(c)


class Corpus:
    """A corpus contains instances, and is useful for training algorithms or splitting to train/test files."""

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.get_aspect_terms()])
        self.top_aspect_terms = frequency_rank(self.aspect_terms_fd)
        self.texts = [t.text for t in self.corpus]

    def echo(self):
        print('%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms)))
        print('Top aspect terms: %s' % (', '.join(self.top_aspect_terms[:10])))

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []


class Evaluate:
    """Manual evaluation of subtask"""

    def __init__(self, correct, predicted):
        self.value_domains_str = None
        self.size = len(correct)
        self.correct = correct
        self.predicted = predicted
        self.reliability_aspect_terms_data = self.get_reliability_aspect_terms_data()
        self.reliability_aspect_terms_polarity = self.get_reliability_aspect_terms_polarity()
        self.reliability_aspect_categories_data = self.get_reliability_aspect_category_data()
        self.reliability_aspect_categoty_polarity = self.get_reliability_aspect_categories_polarities()

    def krippendorff_alpha_aspect_terms(self, krippendorff_metric_type):
        self.get_aspect_terms_value_domains_str()
        alpha = kp.alpha(reliability_data=self.reliability_aspect_terms_data, value_domain=list(self.value_domains_str),
                         level_of_measurement=krippendorff_metric_type)
        return alpha

    def krippendorff_alpha_aspect_terms_polarity(self, krippendorff_metric_type):
        self.get_aspect_terms_value_domains_str()
        alpha = kp.alpha(reliability_data=self.reliability_aspect_terms_polarity,
                         level_of_measurement=krippendorff_metric_type)
        return alpha

    def krippendorff_alpha_aspect_categories(self, krippendorff_metric_type):
        alpha = kp.alpha(reliability_data=self.reliability_aspect_categories_data,
                         value_domain=list(['ovqat', 'xizmat', 'muhit', 'narx', 'boshqalar']),
                         level_of_measurement=krippendorff_metric_type)
        # value_counts = kp._reliability_data_to_value_counts(reliability_data, value_domain)
        # alpha.
        return alpha

    def krippendorff_alpha_aspect_terms_polarity(self, krippendorff_metric_type):
        self.get_aspect_terms_value_domains_str()
        alpha = kp.alpha(reliability_data=self.reliability_aspect_terms_polarity,
                         level_of_measurement=krippendorff_metric_type)
        return alpha

    def get_aspect_terms_value_domains_str(self):
        self.value_domains_str = set(self.reliability_aspect_terms_data[0])
        self.value_domains_str.update(self.reliability_aspect_terms_data[1])

    def get_reliability_aspect_terms_data(self):
        new_gold = []
        new_test = []
        for i in range(self.size):
            gold = self.correct[i].get_aspect_terms()
            test = self.predicted[i].get_aspect_terms()
            self.get_reliability_data(gold, new_gold, new_test, test)
        return [new_gold, new_test]

    def get_reliability_aspect_terms_polarity(self):
        new_gold, new_test = [], []
        for i in range(self.size):
            cor_offsets, cor_polarities = [], []
            pre_offsets, pre_polarities = [], []
            for a in self.correct[i].aspect_terms:
                cor_offsets = list(a.offsets)
                cor_polarities = list(a.polarity)
            for a in self.predicted[i].aspect_terms:
                pre_offsets = list(a.offsets)
                pre_polarities = list(a.polarity)
            for cor_idx in range(len(cor_offsets)):
                for pre_idx in range(len(pre_offsets)):
                    if cor_offsets[cor_idx] != pre_offsets[pre_idx]:
                        new_gold.append(cor_polarities[cor_idx])
                        new_test.append(np.nan)
                        new_gold.append(np.nan)
                        new_test.append(pre_polarities[pre_idx])
                    else:
                        new_gold.append(cor_polarities[cor_idx])
                        new_test.append(pre_polarities[pre_idx])
        return [new_gold, new_test]

    def get_reliability_aspect_category_data(self):
        new_gold = []
        new_test = []
        for i in range(self.size):
            gold = self.correct[i].get_aspect_categories()
            test = self.predicted[i].get_aspect_categories()
            gold = sorted(gold)
            test = sorted(test)
            new_gold = new_gold + gold
            new_test = new_test + test
        return [new_gold, new_test]

    def get_reliability_aspect_categories_polarities(self):
        new_gold, new_test = [], []
        for i in range(self.size):
            cor_polarities = self.correct[i].aspect_categories
            pre_polarities = self.predicted[i].aspect_categories
            new_gold = new_gold + cor_polarities
            new_test = new_test + pre_polarities
        return [new_gold, new_test]

    @staticmethod
    def get_reliability_data(gold, new_gold, new_test, test):
        gold = sorted(gold)
        test = sorted(test)
        cnt = 0
        for j in range(max(len(gold), len(test))):
            try:
                goldJ = re.sub(r'[^\w]', ' ', gold[j])
                testJ = re.sub(r'[^\w]', ' ', test[j])
                if goldJ != testJ:
                    cnt = cnt + 1
                    new_gold.append(goldJ)
                    new_test.append(np.nan)
                    new_gold.append(np.nan)
                    new_test.append(testJ)
                else:
                    new_test.append(testJ)
                    new_gold.append(goldJ)
            except IndexError:
                if len(gold) < j:
                    new_gold.append(np.nan)
                if len(test) < j:
                    new_test.append(np.nan)

    def aspect_extraction(self, b=1):
        manual_common, manual_gold, manual_test = 0., 0., 0.
        for i in range(self.size):
            cor = [a.offsets for a in self.correct[i].aspect_terms]
            pre = [a.offsets for a in self.predicted[i].aspect_terms]
            manual_common += len([a for a in pre if a in cor])
            manual_test += len(pre)
            manual_gold += len(cor)
        p = manual_common / manual_test if manual_test > 0 else 0.
        r = manual_common / manual_gold
        f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, manual_common, manual_test, manual_gold

    def aspect_extraction_cohen_kappa(self, b=1):
        manual_gold, manual_test = [], []
        for i in range(self.size):
            temp_gold_list = []
            temp_test_list = []
            for a in self.correct[i].aspect_terms:
                temp_gold_list.append(a.term)
            for a in self.predicted[i].aspect_terms:
                temp_test_list.append(a.term)
            manual_gold = manual_gold + sorted(temp_gold_list)
            manual_test = manual_test + sorted(temp_test_list)

        return cohen_kappa_score(manual_gold, manual_test)

    def get_confusion_matrix_heatmap(self, manual_gold, manual_test, labels, title):
        confusion = confusion_matrix(manual_gold, manual_test, labels=labels)
        ax = plt.subplot()
        sns.heatmap(confusion, annot=True, fmt='g', ax=ax)
        # labels, title and ticks
        ax.set_xlabel('Test labels')
        ax.set_ylabel('Gold labels')
        ax.set_title(title);
        ax.xaxis.set_ticklabels(list(labels))
        ax.yaxis.set_ticklabels(list(labels))
        plt.show()
    # Aspect Category Detection
    def category_detection(self, b=1):
        manual_common, manual_gold, manual_test = 0., 0., 0.
        for i in range(self.size):
            cor = self.correct[i].get_aspect_categories()
            # Use set to avoid duplicates (i.e., two times the same category)
            pre = set(self.predicted[i].get_aspect_categories())
            manual_common += len([c for c in pre if c in cor])
            manual_test += len(pre)
            manual_gold += len(cor)
        p = manual_common / manual_test if manual_test > 0 else 0.
        r = manual_common / manual_gold
        f1 = (1 + b ** 2) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
        return p, r, f1, manual_common, manual_test, manual_gold

    def aspect_category_detection_cohen_kappa(self, b=1):
        manual_gold, manual_test = [], []
        for i in range(self.size):
            temp_gold_list = []
            temp_test_list = []
            for a in list(self.correct[i].aspect_categories):
                temp_gold_list.append(a.term)
            for a in list(self.predicted[i].aspect_categories):
                temp_test_list.append(a.term)
            manual_gold = manual_gold + sorted(temp_gold_list)
            manual_test = manual_test + sorted(temp_test_list)
            if len(self.correct[i].aspect_categories) != len(self.predicted[i].aspect_categories):
                print("ID missed = ", self.correct[i].id)
        labels = ['ovqat', 'xizmat', 'narx', 'muhit', 'boshqalar']
        self.get_confusion_matrix_heatmap(manual_gold, manual_test, labels, 'Aspect Category term Confusion Matrix')
        alpha = cohen_kappa_score(manual_gold, manual_test, labels=labels)
        return alpha

    def aspect_polarity_estimation(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.polarity for a in self.correct[i].aspect_terms]
            pre = [a.polarity for a in self.predicted[i].aspect_terms]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved

    def aspect_polarity_kappa_cohen_estimation(self, b=1):
        manual_gold, manual_test = [], []
        for i in range(self.size):
            for a in self.correct[i].aspect_terms:
                manual_gold.append(a.polarity)
            for a in self.predicted[i].aspect_terms:
                manual_test.append(a.polarity)
            if len(self.correct[i].aspect_terms) != len(self.predicted[i].aspect_terms):
                print("ID missed = ", self.correct[i].id)
        labels = ['positive', 'negative', 'neutral', 'conflict']
        self.get_confusion_matrix_heatmap(manual_gold, manual_test, labels, 'Aspect Terms Polarity Confusion Matrix')
        return cohen_kappa_score(manual_gold, manual_test, labels=labels)

    def aspect_category_polarity_estimation(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.polarity for a in self.correct[i].aspect_categories]
            pre = [a.polarity for a in self.predicted[i].aspect_categories]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved

    def aspect_category_polarity_kappa_cohen_estimation(self, b=1):
        manual_gold, manual_test = [], []
        for i in range(self.size):
            temp_gold_list = []
            temp_test_list = []
            for a in self.correct[i].aspect_categories:
                manual_gold.append(a.polarity)
            for a in self.predicted[i].aspect_categories:
                manual_test.append(a.polarity)
            if len(self.correct[i].aspect_categories) != len(self.predicted[i].aspect_categories):
                print("ID missed = ", self.correct[i].id)
        labels = ['positive', 'negative', 'neutral', 'conflict']
        self.get_confusion_matrix_heatmap(manual_gold, manual_test, labels, 'Aspect Category Terms Polarity Confusion Matrix')

        return cohen_kappa_score(manual_gold, manual_test)


def main(argv=None):
    # Parse the input
    opts, args = getopt.getopt(argv, "hg:dt:om:k:", ["help", "grammar", "train=", "task=", "test="])
    trainfile, testfile, task = None, None, 1
    use_msg = 'Use as:\n">>> python baselines.py --train file.xml --task -1|1|2|3|4"\n\nThis will parse a train ' \
              'set, examine whether is valid, test files, perform ABSA for task 1, 2, 3, or 4 , and write out a file ' \
              'with the predictions. '

    if len(opts) == 0:
        sys.exit(use_msg)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit(use_msg)
        elif opt in ('-t', "--train"):
            trainfile = arg
        elif opt in ('-m', "--task"):
            task = int(arg)

    # Examine if the file is in proper XML format for further use.
    print('Validating the file...')
    try:
        elements, aspects = validate(trainfile)
        print('PASSED! This corpus has: %d sentences, %d aspect term occurrences, and %d distinct aspect terms.' % (
            len(elements), len(aspects), len(list(set(aspects)))))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    # Get the corpus and split into train/test.
    manual_corpus_gold = Corpus(ET.parse(trainfile).getroot().findall('sentence'))
    manual_corpus_test = Corpus(ET.parse('rest-manual-test-cohen-kapp.xml').getroot().findall('sentence'))

    if task == 1:
        print('\n------- Aspect terms --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved-test: '
              '%d, #relevant-gold: %d)' % Evaluate(manual_corpus_gold.corpus,
                                                   manual_corpus_test.corpus).aspect_extraction())
        print('Cohen\'s kappa = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_extraction_cohen_kappa())
        print('Krippendorff nominal metric = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .krippendorff_alpha_aspect_terms("nominal"))

    if task == 2:
        print('\nAspect term polarity...')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_polarity_estimation())
        print('Cohen Kappa Accuracy = %f,' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_polarity_kappa_cohen_estimation())
        print('Krippendorff nominal metric = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .krippendorff_alpha_aspect_terms_polarity("nominal"))

    if task == 3:
        print('\n------- Aspect Categories --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: '
              '%d, #relevant: %d)' % Evaluate(manual_corpus_gold.corpus,
                                              manual_corpus_test.corpus).category_detection())
        print('Cohen\'s kappa = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_category_detection_cohen_kappa())
        print('Krippendorff nominal metric = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .krippendorff_alpha_aspect_categories("nominal"))

    if task == 4:
        print('\nEstimating aspect category polarity...')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_category_polarity_estimation())
        print('Cohen Kappa Accuracy = %f,' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_category_polarity_kappa_cohen_estimation())
        print('Krippendorff nominal metric = ', Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .krippendorff_alpha_aspect_terms_polarity("nominal"))


if __name__ == "__main__": main(sys.argv[1:])
