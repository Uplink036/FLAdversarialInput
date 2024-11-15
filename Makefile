dependencies: jupyter flower

flower:
	pip install -e ./adversarial
	echo "COUNT=0" > adversarial/.env

jupyter:
	pip3 install --user -r requirements.txt;

merge:
	git mergetool --tool nbdime -- *.ipynb


