#!/bin/bash

# Setup script for invest-product-rag project

set -e  # Exit on error

echo "========================================"
echo "Invest Product RAG - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install NLTK data
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab')"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your MISTRAL_API_KEY"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your MISTRAL_API_KEY"
echo "2. Place your PDF files in data/raw_pdfs/"
echo "3. Run the ingestion pipeline:"
echo "   python scripts/run_ingestion.py"
echo ""
