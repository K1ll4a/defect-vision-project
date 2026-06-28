.PHONY: install data train plot-loss predict streamlit api test clean

install:
	pip install -r requirements.txt
	pip install -e .

data:
	python3 -m defect_detection.data.make_synthetic_dataset --output data/synthetic --train-size 120 --val-size 30 --test-size 30

train:
	python3 -m defect_detection.training.engine --config configs/faster_rcnn.yaml

plot-loss:
	python3 -m defect_detection.training.plot_metrics --metrics runs/defect_faster_rcnn/metrics.jsonl --output-dir assets

predict:
	python3 -m defect_detection.inference.predict --weights runs/defect_faster_rcnn/best.pt --image data/synthetic/val/images/val_0000.png --output outputs/prediction.png

streamlit:
	streamlit run src/defect_detection/serving/streamlit_app.py -- --weights runs/defect_faster_rcnn/best.pt

api:
	uvicorn defect_detection.serving.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

clean:
	rm -rf runs outputs data/synthetic .pytest_cache
