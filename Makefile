merge:
	git mergetool --tool nbdime -- *.ipynb

flower:
	cd adversarial
	pip install -e .
	