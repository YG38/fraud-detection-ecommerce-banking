@echo off
echo Initializing Git repository...
git init

echo Adding files to staging area...
git add README.md .gitignore analysis/

echo Making initial commit...
git commit -m "Initial commit: Project setup and interim-1 submission"

echo Adding remote repository...
git remote add origin https://github.com/YG38/fraud-detection-ecommerce-banking.git

echo Pushing to GitHub...
git push -u origin master

echo Done! Your repository has been initialized and pushed to GitHub.
