dependencies: jupyter flower

flower:
	pip install -e ./adversarial

jupyter:
	pip3 install --user -r requirements.txt;

merge:
	git mergetool --tool nbdime -- *.ipynb


