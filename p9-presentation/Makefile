# Makefile for automating publishing tasks

# Publishes the content
publish: render git-add git-push

# Runs quarto render to build the project
render:
	@echo "Running quarto render..."
	quarto render

# Adds entire publish directory to Git
git-add:
	@echo "Adding publish directory to git..."
	git add publish/*

# Pushes changes to the remote repository
git-push:
	@echo "Pushing changes to remote..."
	git commit -m "Publish new changes"
	git push

.PHONY: publish render git-add git-push