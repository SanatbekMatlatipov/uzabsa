class Word(object):

    def __init__(self, key, sentkey, form, start, end):
        """
         So'z ob'ektini aniqlang.

         :param kaliti: Bu tegishli hujjatning kaliti.
         :param sentkey: Ushbu so'z tegishli bo'lgan gapning kaliti.
         :param formasi: Bu so‘zning qator shakli.
         :param start: Ushbu so'zning boshlang'ich indeksi.
         :param end: Ushbu so'zning yakuniy indeksi.
         """
        self.key = key
        self.sentkey = sentkey
        self.form = form
        self.start = start
        self.end = end
        self.annotations = []
