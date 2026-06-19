.PHONY: install train test report api dashboard clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

test:
	pytest -q

api:
	uvicorn api.main:app --reload

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -f data/*.csv models/*.json models/*.joblib reports/figures/*.png
