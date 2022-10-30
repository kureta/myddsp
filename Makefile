VERSION = $(file < VERSION)

docs_push:
	poetry run mike deploy --push --update-aliases $(VERSION) latest

test:
	pytest --doctest-modules
