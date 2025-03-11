# Traffic Accidents Analysis Dashboard

An interactive dashboard built with [Plotly Dash](https://dash.plotly.com/) for analyzing traffic accidents data using **Principal Component Analysis (PCA)** and **K-Means clustering**. This project demonstrates key data science and visualization techniques including:

- **PCA Scree Plot** to select the number of principal components
- **K-Means Elbow Plot** to determine the optimal number of clusters
- **Interactive Biplot** showing cluster assignments and feature loadings
- **Table of Top PCA Features** based on loadings
- **Scatterplot Matrix** for the top features

## Features

1. **Scree Plot**

   - Displays the explained variance ratio for each principal component.
   - Click on a bar to set the number of components used (intrinsic dimensionality).

2. **Elbow Plot**

   - Shows the inertia (sum of squared distances) vs. the number of clusters (k).
   - Click on a point to select k for K-Means.

3. **Biplot**

   - Plots the data in 2D (PC1 vs. PC2).
   - Points are colored by cluster assignment (if k > 1).
   - Top feature vectors are overlaid to show their contribution to the principal components.
   - Slider to choose how many feature vectors to display.

4. **Top PCA Features**

   - A table listing the most important features (highest PCA loadings) based on the selected number of components.

5. **Scatterplot Matrix**
   - Displays pairwise relationships among the top features.
   - Points are colored by the chosen cluster assignment.

## Installation

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/Arbiter09/traffic-accidents-pca-kmeans-dashboard.git
   ```
2. **Install dependencies**. If you have a `requirements.txt`, use:

   ```bash
   pip install -r requirements.txt
   ```

   Otherwise, ensure you have:

   - `pandas`
   - `numpy`
   - `plotly`
   - `dash`
   - `scikit-learn`

## Usage

1. **Run the Dash app**:
   ```bash
   python app.py
   ```
   or:
   ```bash
   python your_script_name.py
   ```
2. **Open your browser** and go to:
   ```
   http://127.0.0.1:8050/
   ```
3. **Explore the Dashboard**:
   - Select the intrinsic dimensionality from the Scree Plot.
   - Select the number of clusters from the Elbow Plot.
   - View the updated Biplot, Top Features table, and Scatterplot Matrix.

## File Overview

- **app.py** (or similar name): Main Dash application file containing:

  - Data loading & preprocessing
  - PCA and K-Means computations
  - Dash layout & callbacks

- **traffic_accidents_dict new.csv**: Example CSV dataset used by the app.

## Project Structure

```
.
├── app.py                     # Main Dash application code
├── traffic_accidents_dict new.csv  # CSV data file
├── README.md                  # Project documentation
└── requirements.txt           # (Optional) Python dependencies
```

## Contributing

Contributions and suggestions are welcome!

- Fork this repo
- Create a feature branch
- Submit a pull request

## License

This project is provided under an open-source license. Feel free to modify and distribute it in accordance with the license terms.

---

**Enjoy exploring the Traffic Accidents Analysis Dashboard!**

```

```
