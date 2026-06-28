.PHONY: install data train predict streamlit api test clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/make_synthetic_dataset.py --output data/synthetic --train-size 120 --val-size 30

train:
	python -m defect_detection.engine --config configs/faster_rcnn.yaml

predict:
	python -m defect_detection.predict --weights runs/defect_faster_rcnn/best.pt --image data/synthetic/val/images/val_0000.png --output outputs/prediction.png

streamlit:
	streamlit run app/streamlit_app.py -- --weights runs/defect_faster_rcnn/best.pt

api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

clean:
	rm -rf runs outputs data/synthetic .pytest_cache
