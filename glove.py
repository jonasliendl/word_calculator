import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

class Calculator:
    def __init__(self, glove_path: str, topn: int = 5):
        """
        Lädt GloVe-Embeddings und bereitet den NearestNeighbors-Suchindex vor.
        
        Params
        -------
            glove_path: Pfad zur GloVe-Datei (z.B. glove.6B.100d.txt)
            topn: Anzahl der zurückzugebenden nächsten Nachbarn
        
        Returns
        -------
            None
        """
        self.glove_path = Path(glove_path)
        self.topn = topn
        
        print(f"Lade GloVe-Datei '{self.glove_path}'...")
        self.word2idx, self.idx2word, self.matrix = self._load_glove()
        print(f"Geladene Vokabulargröße: {len(self.word2idx)} Wörter.")
        
        self.nn = NearestNeighbors(n_neighbors=self.topn, metric='cosine')
        self.nn.fit(self.matrix)
        print("Fertig.\n")
    
    def _load_glove(self) -> Tuple[dict, List[str], np.ndarray]:
        """
        Liest Glove Embeddings und gibt ein Wörterbuch, eine Liste und die Matrix zurück.
        
        Returns (als Liste)
        -------
            w2i: Wörterbuch von Wörtern zu Indizes
            i2w: Liste von Wörtern
            vecs: Matrix der Vektoren
        """
        w2i = {} # wort zu index
        i2w = [] # index zu wort
        vecs = [] # vektorliste
        with self.glove_path.open('r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.rstrip().split(' ')
                word = parts[0]
                vals = np.array(parts[1:], dtype=np.float32)
                w2i[word] = idx
                i2w.append(word)
                vecs.append(vals)
        return w2i, i2w, np.vstack(vecs)
    
    def compute(self, expr: str) -> List[Tuple[str, float]]:
        """
        Führt eine Analogie-Rechnung der Form "w1 - w2 + w3 - w4 ..." aus
        und gibt die Top-N ähnlichsten Wörter zurück.
        
        Params
        -------
            expr: String wie "king - man + woman"
            
        Returns
        -------
            Liste von Tuples (Wort, cosine_distance)
        """
        tokens = expr.strip().split()
        pos, neg = [], [] # vektoren für wörter mit positivem und negativem vorzeichen
        sign = '+'
        for tok in tokens:
            if tok in ('+', '-'):
                sign = tok
            else:
                if sign == '+':
                    pos.append(tok.lower())
                else:
                    neg.append(tok.lower())
        
        vec = np.zeros(self.matrix.shape[1], dtype=np.float32)
        
        for w in pos:
            if w not in self.word2idx:
                raise ValueError(f"Wort '{w}' nicht im Vokabular.")
            vec += self.matrix[self.word2idx[w]]
            
        for w in neg:
            if w not in self.word2idx:
                raise ValueError(f"Wort '{w}' nicht im Vokabular.")
            vec -= self.matrix[self.word2idx[w]]
        
        dists, idxs = self.nn.kneighbors([vec], n_neighbors=self.topn + len(pos) + len(neg))
        results = []
        
        for dist, idx in zip(dists[0], idxs[0]):
            word = self.idx2word[idx]

            if word in pos or word in neg:
                continue
            
            results.append((word, float(dist)))
            
            if len(results) >= self.topn:
                break
            
        return results

def main():
    parser = argparse.ArgumentParser(description="GloVe Word-Analogy Calculator")
    parser.add_argument('glove', help="Pfad zur GloVe-Datei (z.B. glove.6B.100d.txt)")
    parser.add_argument('--topn', type=int, default=5, help="Anzahl der Top-Ergebnisse")
    args = parser.parse_args()
    
    calc = Calculator(args.glove, topn=args.topn)
    print("Gib Analogie-Ausdrücke ein (z.B. king - man + woman). 'exit' zum Beenden.\n")
    
    while True:
        expr = input(">>> ").strip()
        if expr.lower() in ('exit', 'quit'):
            break
        if not expr:
            continue
        try:
            res = calc.compute(expr)
            print(f"\nTop {args.topn} Ergebnisse für '{expr}':")
            for w, d in res:
                print(f"  {w:<15} (cosine distance: {d:.4f})")
            print()
        except ValueError as e:
            print("Fehler:", e, "\n")

if __name__ == '__main__':
    main()
