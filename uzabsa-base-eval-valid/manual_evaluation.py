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
        self.size = len(correct)
        self.correct = correct
        self.predicted = predicted

    def krippendorff_alpha(self, type):
        reliability_data = [
            ['ofitsiantning', 'his-tuyg\'ular', 'u yerga', 'XIZMAT', 'oilamiz', '**', 'oshxona'],
            ['ofitsiantning', 'his-tuyg\'ular', 'u yerga', 'XIZMAT', '**', 'Stol', 'oshxona']
        ]

        alpha = kp.alpha(reliability_data=reliability_data, level_of_measurement=type)
        print("Krippendorff's alpha for nominal metric: ", alpha)
        return alpha

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

        print(manual_gold)
        # print(manual_test)
        print(confusion_matrix(manual_gold, manual_test))
        return cohen_kappa_score(manual_gold, manual_test)

    def aspect_category_polarity_estimation(self, b=1):
        common, relevant, retrieved = 0., 0., 0.
        for i in range(self.size):
            cor = [a.polarity for a in self.correct[i].aspect_categories]
            pre = [a.polarity for a in self.predicted[i].aspect_categories]
            common += sum([1 for j in range(len(pre)) if pre[j] == cor[j]])
            retrieved += len(pre)
        acc = common / retrieved
        return acc, common, retrieved


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
    manual_corpus_test = Corpus(ET.parse('rest-manual-test.xml').getroot().findall('sentence'))

    if task == -1:
        print('\n------- Aspect terms --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: '
              '%d, #relevant: %d)' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus).aspect_extraction())
        print('\n------- Aspect Categories --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: '
              '%d, #relevant: %d)' % Evaluate(manual_corpus_gold.corpus,
                                              manual_corpus_test.corpus).category_detection())
        print('\n------- Aspect term polarity --------')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_polarity_estimation())
        print('\n-------- Aspect Category polarity  --------')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_category_polarity_estimation())

    if task == 1:
        print('------- Aspect terms --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: '
              '%d, #relevant: %d)' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus).aspect_extraction())

    if task == 2:
        print('------- Aspect Categories --------')
        print('P = %f -- R = %f -- F1 = %f (#correct: %d, #retrieved: '
              '%d, #relevant: %d)' % Evaluate(manual_corpus_gold.corpus,
                                              manual_corpus_test.corpus).category_detection())
    if task == 3:
        print('Aspect term polarity...')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_polarity_estimation())
        print('Cohen Kappa Accuracy = %f,' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_polarity_kappa_cohen_estimation())
    if task == 4:
        print('Estimating aspect category polarity...')
        print('Accuracy = %f, #Correct/#All: %d/%d' % Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus)
              .aspect_category_polarity_estimation())
    if task == 5:
        Evaluate(manual_corpus_gold.corpus, manual_corpus_test.corpus).krippendorff_alpha("nominal")


if __name__ == "__main__": main(sys.argv[1:])
