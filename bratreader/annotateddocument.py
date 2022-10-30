from itertools import chain
from lxml import etree


class AnnotatedDocument(object):
    """Brat Korpusidagi Hujjatni ANN2XML ko'rinishida taqdim etish."""

    def __init__(self, key, sentences):
        """
         ANN2XML da Brat hujjatini yaratish.

         :param kaliti: (string) Hujjatning kaliti.
         Odatda kengaytmasiz fayl nomi
         (masalan, “022.ann” 022 ga aylanadi)
         :param jumlalari: so'zlarni o'z ichiga olgan lug'atlar ro'yxati.
         Ko'rib chiqish matnini so'zma-so'z ifodalaydi.
         :qaytish: Yo'q
         """
        self.sentences = sentences
        annotations = [chain.from_iterable([w.annotations for w in set(x.words)])
                       for x in sentences]

        self.annotations = list(chain.from_iterable(annotations))

        self.key = key

        self.text = u"\n".join([u" ".join([x.form for x in s.words])
                                for s in sentences])

    def export_xml(self, pathtofile):
        """
         Joriy hujjatni belgilangan joyda XML fayliga eksport qilish.

         :param pathtofile: .XML faylni saqlash kerak boʻlgan yoʻl.
         :qaytish: Yo'q
         """
        document = etree.Element("sentences")
        sentence = etree.Element("sentence", ID=self.key)
        text = etree.Element("text")
        text.text = self.text
        sentence.append(text)

        aspectTerms = etree.Element("aspectTerms")
        aspect_polarities = dict()
        for annotation in self.annotations:
            for label, valency in annotation.labels.items():
                isLabelFound = 0
                if label == 'AspectPolarity':
                    if aspect_polarities.get(str(annotation.repr), None) is not None:
                        continue
                    aspect_polarities[str(annotation.repr)] = valency[0]

                    aspectTerms.append(etree.Element("aspectTerm",
                                                     term=str(annotation.repr)))
                    aspectTerm = aspectTerms.getchildren()[-1]
                    aspectTerm.set('polarity', "|".join(valency))
                    aspectTerm.set("from", " ".join(str(x[0]) for x in annotation.spans))
                    aspectTerm.set("to", " ".join(str(x[1]) for x in annotation.spans))
                    sentence.append(aspectTerms)

        aspectCategories = etree.Element("aspectCategories")
        category_types = dict()
        for annotation in self.annotations:
            aspectCategory = etree.Element("aspectCategory")
            isLabelFound = 0
            # print("annotation = ", annotation)
            category_type = ""
            category_polarity = ""
            for label, valency in annotation.labels.items():
                if label == 'CategoryType':
                    # print('label = ', label, ' valency = ', valency)
                    isLabelFound = 1
                    category_type = valency[0]
                    aspectCategory.set('category', category_type)
                    continue
                if label == 'CategoryPolarity':
                    # print('label = ', label, ' valency = ', valency)
                    isLabelFound = 1
                    category_polarity = valency[0]
                    aspectCategory.set('polarity', category_polarity)
                    continue
            if category_type in category_types:
                continue
            category_types[category_type] = category_polarity
            if isLabelFound:
                aspectCategories.append(aspectCategory)
                sentence.append(aspectCategories)
                document.append(sentence)

        with open(pathtofile, 'wb') as f:
            etree.ElementTree(document).write(f, pretty_print=True)
