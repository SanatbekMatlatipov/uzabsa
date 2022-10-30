from bratreader.word import Word


class Sentence(object):

    def __init__(self, key, line, start):
        """
         Gap obyekti.

         :param kaliti: Bu gap tegishli kalit.
         :param line: Bu gap kelgan qator.
         :param start: Ushbu qatorning belgilar bilan boshlang'ich indeksi.
         """
        self.key = key
        self.words = []
        self.start = start
        self.end = start + len(line)

        for windex, w in enumerate(line.split()):

            start = start
            end = start+len(w)
            self.words.append(Word(key=windex,
                                   sentkey=self.key,
                                   form=w,
                                   start=start,
                                   end=end))
            start = end+1

    def getwordsinspan(self, start, end):
        """
         Belgilangan belgilar oralig'idagi barcha so'zlarni oling.

         :param start: Belgilardagi boshlanish indeksi.
         :param end: Belgilardagi yakuniy indeks.
         : oraliq ichiga tushadigan so'zlar ro'yxatini qaytaring.
         """
        return [word for word in self.words if
                (word.start <= start < word.end)
                or (word.start < end <= word.end)
                or (start < word.start < end and start < word.end < end)]
