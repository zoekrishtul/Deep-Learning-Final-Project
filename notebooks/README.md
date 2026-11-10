# Notebooks

- **kaggle_pipeline.ipynb**: primary integrated entry point; dependency installation and prepared-input/direct-download support. The [root README](../README.md) gives 15-example settings.
- **Clip+ MLP.ipynb**: Yu-Jye's original grounding exploration, retained as development evidence. src/grounding.py documents differences in its adaptation.
- **kaggle_data_setup.ipynb**: earlier workflow for preparing test data as a reusable Kaggle input.
- **kaggle_prompting.ipynb**: earlier standalone prompting runner. Use the full pipeline notebook for the integrated study.

These notebooks document distinct contributions/workflows. Keep reusable logic in src/. Do not publish tokens, large embedded datasets, or model weights.
