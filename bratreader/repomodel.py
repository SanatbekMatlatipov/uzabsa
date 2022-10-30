import sys
from bratreader.annotateddocument import AnnotatedDocument
from bratreader.annotationimporter import importann
from glob import iglob

import os


class RepoModel(object):
    """
    BRAT bilan izohlangan mahalliy omborni modellashtirish uchun sinf.

     Brat bilan izohlangan korpus korpusdagi har bir hujjat uchun 2 ta fayldan foydalanadi:
     Brat Standoff formatidagi izohlarni o'z ichiga olgan .ann fayli
     (http://brat.nlplab.org/standoff.html) va .txt fayli
     haqiqiy matn. Ushbu vosita ushbu fayllarning juftlarini o'z ichiga olgan papkani oladi
     kiritadi va RepoModel ob'ektini yaratadi. Bu RepoModel ob'ekti bo'lishi mumkin
     XML formatida eksport qilinadi yoki xotirada ishlaydi.

     Hozirda dastur Eslatmalar yoki # ta izohlarni e'tiborsiz qoldirmoqda.
    """

    def __init__(self, pathtorepo):
        """
         RepoModel obyektini yaratish.

         :param pathtorepo: (string) mahalliy omborga yo'l, qaysi
         .ann va .txt fayllari juftlarini o'z ichiga oladi. Kafolat uchun hech qanday tekshiruv o'tkazilmaydi
         omborning izchil ekanligi.
         :qaytish: Yo'q
         """
        # Har bir hujjat matn birligi sifatida saqlanadi.
        self.documents = {}

        if os.path.isdir(pathtorepo):
            for path in iglob("{0}/*.ann".format(pathtorepo)):
                try:
                    # Har bir hujjatning kaliti hujjat nomisiz
                    # qo'shimcha (ya'ni, "001.ann" "001" ga aylanadi)
                    key = os.path.splitext(path)[0]
                    key = os.path.split(key)[-1]
                    context = importann(path)
                    self.documents[key] = AnnotatedDocument(key, context)
                except KeyError as e:
                    print("Parse error for document {}: {}, {} \n".format(str(path),
                                                                          str(e),
                                                                          sys.exc_info()[0])
                          )

        else:
            raise IOError(u"{0} is not a valid directory".format(pathtorepo))

    def save_xml(self, pathtofolder):
        """
         RepoModelni XML sifatida belgilangan papkaga eksport qiling.

         Agar papka mavjud bo'lmasa, u yaratilgan.
         :param pathtofolder: (string) XML joylashgan papkaga yo'l
         eksport qilinishi kerak.
         """
        if not os.path.isdir(pathtofolder):
            os.mkdir(pathtofolder)

        for document in self.documents.values():
            path = os.path.join(pathtofolder,
                                "{0}.xml".format(str(document.key)))
            document.export_xml(path)
