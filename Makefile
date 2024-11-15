dependencies: jupyter flower

flower:
	pip install -e ./del2
	echo "COUNT=0" > del2/.env

jupyter:
	pip3 install --user -r requirements.txt;

merge:
	git mergetool --tool nbdime -- *.ipynb


