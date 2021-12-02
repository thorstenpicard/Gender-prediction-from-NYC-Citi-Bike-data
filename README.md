# Citi-Bike
Repository for analysing NYC's Citi Bike system data.

* The project is based on the monthly Citi Bike datasets in 2019.

File explanation:
* CB-2019-dataset-creation: removes and adds features, filters data (e.g., exclude age>=100), corrects the data types, and saves each processed dataset as a .parquet file to maintain the data types. Meanwhile, it keeps track of the monthly differences in trip count between the original dataset and the filtered dataset. This script results in 13 files: 12 .parquet files representing the monthly datasets and 1 .csv file that contains aggregated statistics per month. The parquet files serve as input for the classifiers.

