.PHONY: glove
glove:
	poetry run python glove.py glove/glove.6B.200d.txt --topn 5

.PHONY: transformer
transformer:
	poetry run python transformer.py vocab.txt --topn 5

.PHONY: load_glove
load_glove:
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip

.PHONY: generate_vocab
generate_vocab:
	cut -d ' ' -f1 glove/glove.6B.200d.txt > vocab.txt
