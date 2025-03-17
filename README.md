# KPMG Datathon Project

This repository contains code and resources for the HEC Data Challenge Montréal (March 24-30, 2025), focused on machine learning and artificial intelligence.

## Project Structure

```
.
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── external/              # External data sources
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/           # Exploratory data analysis
│   └── modeling/              # Modeling notebooks
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   ├── features/              # Feature engineering
│   ├── models/                # Model definitions
│   ├── visualization/         # Visualization utilities
│   └── utils/                 # Utility functions
├── models/                    # Saved models
├── reports/                   # Generated reports
│   └── figures/               # Generated figures
├── tests/                     # Test files
├── .gitignore                 # Git ignore file
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

### Local Environment

1. Clone the repository:

   ```
   git clone https://github.com/PierreEmmanuelGoffi/kpmg_datathon.git
   cd kpmg_datathon
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```
   pip install --prefer-binary -r requirements.txt
   ```

### Docker Environment

1. Build the Docker image:

   ```
   docker build -t kpmg-datathon .
   ```

2. Run the container:
   ```
   docker run -p 8888:8888 -v $(pwd):/app kpmg-datathon
   ```

## Usage

1. For exploratory data analysis, see the notebooks in `notebooks/exploratory/`.
2. For model development, see the notebooks in `notebooks/modeling/`.
3. Use the source code in `src/` for production-ready implementations.
