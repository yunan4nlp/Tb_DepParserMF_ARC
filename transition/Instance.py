class Instance:
    def __init__(self, sentences, vocab):
        self.words = []
        self.heads = []
        self.rels = []
        self.tags = []
        root = sentences[0]
        assert root.id == 0 and root.head == 0 and \
               root.rel == vocab._root and root.tag == vocab._root
        for idx in range(1, len(sentences)):
            w = sentences[idx]
            self.words.append(w.form)
            head = w.head - 1
            assert head >= -1 and head < len(sentences) - 1
            self.heads.append(w.head - 1)
            self.rels.append(w.rel)
            self.tags.append(w.tag)

