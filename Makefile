.PHONY: install test example clean

install:
	python -m pip install -e ".[test]"

test:
	python -m pytest

example:
	python scripts/02_compute_store_scores.py \
		--input data/example/reviews_with_dimensions.csv \
		--output outputs/tables/example_store_scores.csv

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('outputs').rglob('*') if p.is_file() and p.name != '.gitkeep']"

