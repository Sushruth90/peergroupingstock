# peergroupingstock
**Stock Returns Hierarchical Clustering**
This Python script performs hierarchical clustering on stock returns data to identify clusters of stocks based on their correlation distance. It filters the data based on NAICS codes starting with specific digits, then calculates the correlation matrix, converts it into a distance matrix, and performs hierarchical clustering using Ward's method.

**Features**
Filters stock data based on NAICS codes.
Calculates correlation matrix and distance matrix.
Performs hierarchical clustering.
Identifies stocks in the same cluster as NFG (example ticker).
Outputs the tickers of stocks in the same cluster.

**Requirements**
Python 3.x
pandas
numpy
scipy

**Usage**
Clone the repository.
Place your stock returns data file in the specified directory.
Update the file path in the script if necessary.
Run the script.
