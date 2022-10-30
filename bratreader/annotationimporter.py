import os

from io import open
from collections import OrderedDict, defaultdict
from bratreader.annotation import Annotation
from bratreader.sentence import Sentence


def importann(pathtofile):
    """
     .ANN va .TXT fayllarini jilddan import qiling.

     :param pathtofile: (string) ikkala faylni o'z ichiga olgan jildga yo'l
     .ann va .txt fayllari.
     :qaytish: izohlar lug'ati va satrni o'z ichiga olgan kortej,
     hujjat matnini ifodalaydi
     """
    annotations = readannfile(pathtofile)
    path, extension = os.path.splitext(pathtofile)

    sentences = []

    char_index = 0

    for sent_index, line in enumerate(open(path + ".txt", encoding='utf-8')):
        sentences.append(Sentence(sent_index, line, char_index))
        # print('importann ', line, ' ', sent_index, ' sentences ', sentences )
        char_index += len(line)

    _join(annotations.values(), sentences)

    # print('sentences = ', annotations.values())
    return sentences


def _join(annotations, sentences):
    """
     jumlalar ro'yxati bilan izohlar ro'yxatiga qo'shiling.

     :param annotations: izohlar ro'yxati
     :param jumlalar:
     :qaytish:
     """
    for ann in annotations:
        for span in ann.spans:

            begin, end = span

            for s in sentences:
                words = s.getwordsinspan(begin, end)
                ann.words.extend(words)
                for w in words:
                    w.annotations.append(ann)


def _createannotationobjects(annotations):
    """
     Har bir "T" izohi uchun Annotatsiya sinfining namunalarini yarating.

     Kirish faqat "T" izohlari sifatida qabul qilinadi.

     :param annotations: (dict) "T" izohlari lug'ati.
     :return: (OrderedDict) Annotatsiyalar ob'ektlarining tartiblangan lug'ati.
     Ushbu lug'atning uzunligi kirish lug'atiga teng bo'lishi kerak.
     """
    targets = OrderedDict()

    for key, t in annotations.items():
        splitted = t.split("\t")
        t = splitted[0]
        repr = u" ".join(splitted[1:])

        split = t.split()
        label = split[0]

        spans = [[int(span.split()[0]), int(span.split()[1])]
                 for span in u" ".join(split[1:]).split(";")]
        targets[key] = Annotation(key, repr, spans, [label])
        # print("target [ ", key, "] --> ", targets[key])

    return targets


def _find_t(e, annotations):
    """
     .ann faylidan "E" izohi berilgan bo'lsa, "T" izohini toping.

     "E" izohlarini joylashtirish mumkinligi sababli, qidiruv chuqurroq amalga oshirilishi kerak
     darajalari.

     :param e: (string) biz maqsadni topmoqchi bo'lgan "E" izohi.
     :param annotations: (dict) annotatsiyalar dict.
     :qaytish: bu e annotatsiya ko'rsatadigan "T" izohlarining tugmalari.
     """
    e = e.split()
    keys = []

    if len(e) > 1:

        targetkeys = [y for y in [x.split(":")[1] for x in e[1:]]]

        for key in targetkeys:
            if key[0] == "E":
                keys.append(annotations['E'][key[1:]].split()[0].split(":")[1])

            if key[0] == "T":
                keys.append(key)

    return keys


def _evaluate_annotations(annotations):
    """
        .ann fayli uchun barcha izohlarni baholang.

        Izohlarning har bir toifasi (masalan, "T", "E", "A", "R", "N") ko'rib chiqiladi.
        alohida. Birinchidan, barcha "T" izohlari Annotatsiya ob'ektlariga qayta yoziladi,
        chunki bu barcha ifodalarning yakuniy maqsadlari.

        Keyin ifodalar uchun valentliklarni o'z ichiga olgan "A" izohlari
        va maqsadlar baholanadi. Uchinchidan, voqea bo'lgan "E" izohlari
        ifodalar (“A” dan valentlik olishi mumkin) baholanadi.

        Va nihoyat, boshqalardan alohida bo'lgan "R" va "N" izohlari,
        baholanadi.

        :param annotations: (dict of dict) lug'atlar lug'ati,
        birinchi lug'atda har bir annotatsiya toifasi uchun kalit mavjud
        (ya'ni, "T", "E", "A", "R", "N"). Ikkinchisida ikkinchi raqam mavjud
        izohlarni farqlash. Bularning barchasi .ann fayliga asoslangan.
        Barcha tugmachalar, hatto raqamli tugmalar ham mosligini kafolatlash uchun satrlardir
        boshqa versiyalar.

        Misol: ann faylida bizda "T14" izohi bor. Bunga qo'shiladi
        "T" lug'ati "14" kaliti sifatida.

        :return: Annotatsiya ob'ektlari lug'ati.
        """

    # Create the annotation objects
    annotationobjects = _createannotationobjects(annotations["T"])
    # "A" annotations
    # print('annotations = ', annotations)
    for a in annotations["A"].values():
        try:
            # Triple format (e.g. Sentiment T14 Positive)
            label, key, valency = a.split()
            # print('label = ', label, ' key = ', key, 'valency = ', valency)
        except ValueError:
            # Only a label, no valency (e.g. Target T14)
            label, key = a.split()
            valency = ""

        # Type of target (e.g. "T")
        type = key[0]
        id = key[1:]

        # print(type)
        if type == "E":
            tempe = annotations["E"][id]
            key2 = tempe.split()[0].split(":")[1][1:]
            annotationobjects[key2].labels[label].append(valency)

        elif type == "T":
            annotationobjects[id].labels[label].append(valency)

    # "E" annotations
    for e in annotations["E"].values():
        # function returns the id of T.
        targetkeys = _find_t(e, annotations)
        origintype, originkey = e.split()[0].split(":")
        originkey = originkey[1:]

        targets = [x[1:] for x in targetkeys]

        for x in targets:
            t = annotationobjects[originkey]
            annotationobjects[x].links[origintype].append(t)

    # "R" annotations
    for r in annotations["R"].values():
        r = r.split()

        if len(r) > 1:

            origintype = r[0]
            originkey = r[1].split(":")[1][1:]
            targets = [y for y in [x.split(":")[1][1:] for x in r[2:]]]

            for x in targets:
                t = annotationobjects[originkey]
                annotationobjects[x].links[origintype].append(t)

    return annotationobjects


def readannfile(filename):
    """
     .ann faylini o'qing va lug'atlarni o'z ichiga olgan lug'atni qaytaradi.

     :param fayl nomi: (string) .ann faylining fayl nomi.
     :qaytish: (dict of dict) ni ifodalovchi lug‘atlar lug‘ati
     izohlar.
     """
    anndict = defaultdict(dict)
    with open(filename, encoding='utf-8') as f:
        for index, line in enumerate(f):

            begin = line.rstrip().split("\t")[0]
            rest = line.rstrip().split("\t")[1:]

            try:
                anndict[begin[0]][begin[1:]] = u"\t".join(rest)
            except IndexError:
                continue

    return _evaluate_annotations(anndict)
