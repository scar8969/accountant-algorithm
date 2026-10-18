PY ?= python

.PHONY: install test benchmark sweep report serve seed demo clean

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

benchmark:
	$(PY) scripts/run_benchmark.py --config configs/experiments/h1_structured_vs_baselines.yaml

sweep:
	$(PY) scripts/sweep_seeds.py --seeds 5

report:
	$(PY) scripts/make_report.py

serve:
	$(PY) -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8775

seed:
	$(PY) scripts/seed_demo.py --n 100

clean:
	rm -rf .pytest_cache **/__pycache__ results/results.json results/sweep.json results/report.md
