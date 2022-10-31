VERSION = $(file < VERSION)

docs_push:
	poetry run mike deploy --push --update-aliases $(VERSION) latest

test:
	poetry run pytest --doctest-modules --hypothesis-show-statistics

coverage_run:
	poetry run coverage run -m pytest --doctest-modules -q

coverage_report:
	poetry run coverage report

coverage: coverage_run coverage_report
