#!/bin/bash
# setup_git.sh
# Run this from inside the MiniVaultDB-Analytics folder
# Usage: bash setup_git.sh <your-github-username>

set -e

USERNAME=${1:-"your-username"}
REPO="MiniVaultDB-Analytics"

echo "══════════════════════════════════════════"
echo " MiniVaultDB Analytics — Git Setup Script"
echo "══════════════════════════════════════════"

# 1. Initialize git
echo ""
echo "► Initializing git repository..."
git init
git branch -M main

# 2. Initial commit — project structure
echo ""
echo "► Commit 1: initial project structure"
git add .gitignore requirements.txt
git commit -m "feat: initial project structure and .gitignore"

# 3. Commit adapter
echo ""
echo "► Commit 2: MiniVaultDB adapter"
git add minivaultdb/
git commit -m "feat: MiniVaultDB adapter with JSON serialization"

# 4. Commit ingestion
echo ""
echo "► Commit 3: ingestion pipeline"
git add pipeline/ingest.py
git commit -m "feat: data ingestion pipeline (CSV → key-value → DB)"

# 5. Commit retrieval
echo ""
echo "► Commit 4: retrieval pipeline"
git add pipeline/retrieve.py
git commit -m "feat: retrieval pipeline (DB → pandas DataFrame)"

# 6. Commit preprocessing
echo ""
echo "► Commit 5: preprocessing"
git add pipeline/preprocess.py pipeline/__init__.py
git commit -m "feat: preprocessing pipeline (impute, encode, scale)"

# 7. Commit training
echo ""
echo "► Commit 6: model training"
git add pipeline/train.py
git commit -m "feat: model training and evaluation (RF + LR)"

# 8. Commit prediction
echo ""
echo "► Commit 7: prediction pipeline"
git add pipeline/predict.py
git commit -m "feat: end-to-end prediction pipeline"

# 9. Commit notebook
echo ""
echo "► Commit 8: EDA notebook"
git add notebooks/
git commit -m "feat: EDA and model training notebook"

# 10. Commit README
echo ""
echo "► Commit 9: README"
git add README.md
git commit -m "docs: README with architecture and usage"

# 11. Add remote and push
echo ""
echo "► Setting up GitHub remote..."
echo ""
echo "  Make sure you have created the repo on GitHub first:"
echo "  https://github.com/new  →  name it: $REPO"
echo ""
read -p "  Press Enter once the GitHub repo is created..."

git remote add origin "https://github.com/$USERNAME/$REPO.git"

echo ""
echo "► Pushing to GitHub..."
git push -u origin main

echo ""
echo "══════════════════════════════════════════"
echo " ✓ All done! Your repo is live at:"
echo "   https://github.com/$USERNAME/$REPO"
echo "══════════════════════════════════════════"
