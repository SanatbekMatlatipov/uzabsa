from collections import defaultdict


class Annotation(object):
    """This class represents an annotation."""

    def __init__(self, id, representation, spans, labels=()):
        """
                 Annotatsiya ob'ektini yaratish.

                 :param id: (string) Joriy izohning identifikatori.
                 :param ko'rinishi: (string) ning satr ko'rinishi
                 izoh. Izohlar bo'lishi mumkinligini hisobga olmaydi
                 uzluksiz.
                 :param spans: (ints ro'yxati) Ints ro'yxati
                 har qanday uchun boshlang'ich va yakuniy nuqtalarni belgilar bilan ifodalaydi
                 izohdagi so'zlar.
                 :param teglari: (satrlar ro'yxati) uchun boshlang'ich teglar ro'yxati
                 izohlash obyekti. Ular hech qachon boshlang'ich qiymatga ega bo'lmaydi.
                 :qaytish: Yo'q
        """
        self.id = id
        self.links = defaultdict(list)
        self.labels = defaultdict(list)
        for label in labels:
            self.labels[label] = []
        self.repr = representation
        self.spans = spans
        self.realspan = (spans[0][0], spans[-1][1])
        self.words = []

    def __repr__(self):
        """Izohning ifodasi."""
        return "Annotation: {0}".format(self.repr.encode("utf-8"))
