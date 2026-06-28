# Как описать проект в резюме

**DefectVision** — система object detection для поиска визуальных дефектов на изображениях.

Что написать в CV/GitHub README:

- Built a PyTorch Faster R-CNN pipeline for visual defect detection.
- Implemented custom COCO-style Dataset, training loop, mAP@0.5 evaluation and inference visualization.
- Added Streamlit demo, FastAPI endpoint, Dockerfile and synthetic dataset generator for reproducibility.
- Improved model quality through transfer learning and detection-specific augmentations.

## Следующие задачи

1. Подключить реальный датасет дефектов.
2. Провести baseline vs improved model сравнение.
3. Добавить WandB/MLflow logging.
4. Добавить model card: ограничения, failure cases, latency.
