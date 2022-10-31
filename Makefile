VERSION = $(file < VERSION)

docs_push:
	poetry run mike deploy --push --update-aliases $(VERSION) latest

test:
	poetry run coverage run -m pytest --doctest-modules

coverage:
	poetry run coverage html && xdg-open htmlcov/index.html
