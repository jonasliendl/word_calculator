import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


class Calculator:
    def __init__(self, model_name: str, vocab_path: str, topn: int = 5):
        """
        Lädt das Sentence-Transformer-Modell und berechnet Embeddings für das Vokabular.

        Params
        ------
            model_name: Name des sentence-transformers Modells (z.B. 'all-MiniLM-L6-v2')
            vocab_path: Pfad zur Vokabular-Datei (ein Wort pro Zeile)
            topn: Anzahl der Top-Ergebnisse
            
        Returns
        -------
            None
        """
        print(f"Lade Transformer-Modell '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        self.topn = topn

        print(f"Lade Vokabular aus '{vocab_path}' …")
        self.vocab = [w.strip() for w in Path(vocab_path).read_text(encoding='utf-8').splitlines() if w.strip()]
        print(f"Vokabular geladen: {len(self.vocab)} Wörter.")

        print("Berechne Embeddings für alle Vokabeln …")
        self.embeddings = self.model.encode(
            self.vocab,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self.nn = NearestNeighbors(n_neighbors=self.topn + 10, metric='cosine')
        self.nn.fit(self.embeddings)
        print("Index bereit.\n")

    def compute(self, expr: str):
        """
        Führt eine Analogie-Rechnung der Form "w1 - w2 + w3 ..." aus
        und gibt die Top-n ähnlichsten Wörter zurück.

        Params
        ------
            expr: String wie "king - man + woman"
            
        Returns
        ------
            Liste von (Wort, cosine_distance)
        """
        tokens = expr.strip().split()
        pos, neg = [], []
        sign = '+'
        for tok in tokens:
            if tok in ('+', '-'):
                sign = tok
            else:
                if sign == '+':
                    pos.append(tok.lower())
                else:
                    neg.append(tok.lower())

        vecs_pos = self.model.encode(
            pos, convert_to_numpy=True, normalize_embeddings=True
        ) if pos else np.zeros(self.embeddings.shape[1], dtype=np.float32)
        
        vecs_neg = self.model.encode(
            neg, convert_to_numpy=True, normalize_embeddings=True
        ) if neg else np.zeros(self.embeddings.shape[1], dtype=np.float32)

        vec = np.sum(vecs_pos, axis=0) - np.sum(vecs_neg, axis=0)

        dists, idxs = self.nn.kneighbors([vec], n_neighbors=self.topn + len(pos) + len(neg))
        results = []
        
        for dist, idx in zip(dists[0], idxs[0]):
            w = self.vocab[idx]
            if w in pos + neg:
                continue
            
            results.append((w, float(dist)))
            
            if len(results) >= self.topn:
                break

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Word-Analogy Calculator mit Transformers"
    )
    parser.add_argument(
        'vocab', help="Pfad zur Vokabular-Datei (ein Wort pro Zeile)"
    )
    parser.add_argument(
        '--model', default='all-MiniLM-L6-v2',
        help="Sentence-Transformer Modellname"
    )
    parser.add_argument(
        '--topn', type=int, default=5,
        help="Anzahl der Top-Ergebnisse"
    )
    args = parser.parse_args()

    calc = Calculator(
        model_name=args.model,
        vocab_path=args.vocab,
        topn=args.topn
    )
    print("Gib Analogie-Ausdrücke ein (z.B. king - man + woman). 'exit' zum Beenden.\n")
    while True:
        expr = input('>>> ').strip()
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
        except Exception as e:
            print("Fehler:", e, "\n")


if __name__ == '__main__':
    main()
