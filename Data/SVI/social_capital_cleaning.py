## This is a script that imputes the Social Capital values for missing counties using a K-NN imputer based on geographic proximity.
## The bit after the first chunk is to test a range of k values for the KNN imputer to see what the optimal number of neighbors should be.
## It does this by randomly masking 20% of the known values and then calculating the Mean Absolute Error (MAE) between the known and imputed values
## for each k value.
## After running some testing, it seems like the optimal k value is around 15, but the MAE is around 0.55, which is ok, not great. Going to
## save this script, and run the XGBoost model with the imputed Social Capital values to see how the feature importance ranking changes.

from sklearn.impute import KNNImputer
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import requests


data_df = pd.read_csv('Data/SVI/Raw Files/Social_Capital_2018_US_county.csv',)

# Step 1: Load GeoJSON data
url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
response = requests.get(url)
county_geo = json.loads(response.text)

# Step 2: Extract FIPS codes and coordinates
county_data = []
for feat in county_geo['features']:
    try:
        # Extract FIPS code from GEO_ID
        geo_id = feat['properties']['GEO_ID']
        fips = geo_id.split('US')[-1].zfill(5)
        
        # Get coordinates (handles both Polygon and MultiPolygon)
        geometry_type = feat['geometry']['type']
        coordinates = feat['geometry']['coordinates']
        
        # For Polygon: take first point of first ring
        if geometry_type == 'Polygon':
            point = coordinates[0][0]  # First ring, first point
        # For MultiPolygon: take first point of first ring of first polygon
        elif geometry_type == 'MultiPolygon':
            point = coordinates[0][0][0]  # First polygon, first ring, first point
        else:
            continue  # Skip other geometry types
            
        county_data.append({
            'FIPS': fips,
            'lon': point[0],
            'lat': point[1]
        })
    except (KeyError, IndexError, TypeError) as e:
        print(f"Skipping feature {fips} due to error: {e}")
        continue

county_df = pd.DataFrame(county_data)

## Got the error: "ValueError: You are trying to merge on int64 and object columns for key 'FIPS'. If you wish to proceed you should use pd.concat"
## Trying to correct
data_df['FIPS'] = data_df['FIPS'].astype(str).str.zfill(5)
county_df['FIPS'] = county_df['FIPS'].astype(str).str.zfill(5)

# Step 2: Merge centroids with your data
data_df = pd.merge(
    data_df,
    county_df,
    on='FIPS',
    how='left'
)

# Step 3: Prepare for imputation
data_df['2018 Social Capital'] = data_df['2018 Social Capital'].replace(0, np.nan)  # Treat 0s as missing

# Step 4: KNN Imputation using geographic proximity
imputer = KNNImputer(n_neighbors=15, weights='distance')
data_df['2018 Social Capital'] = imputer.fit_transform(
    data_df[['2018 Social Capital', 'lat', 'lon']]
)[:, 0]

print(data_df.head(10))

# Define the output path
output_path = 'Data/SVI/Final Files/Social Capital_final_rates_testing.csv'

# Save the result (optional)
data_df.to_csv(output_path, index=False)


#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################

# ##############################################
# ## I am now going to test a range of k values for the KNN imputer to see what the optimal number of neighbors should be
# ##############################################
# import numpy as np
# from sklearn.metrics import mean_absolute_error

# # Identify non-missing indices
# non_missing_indices = data_df[data_df['2018 Social Capital'].notna()].index

# # Randomly mask 20% of known values
# #np.random.seed(42)
# # mask = np.random.choice(non_missing_indices, size=int(0.2 * len(non_missing_indices)), replace=False)
# # true_values = data_df.loc[mask, '2018 Social Capital'].copy()
# # data_df.loc[mask, '2018 Social Capital'] = np.nan

# # k_values = np.arange(2,41)#[3, 5, 7, 9, 11, 15, 20]
# # mae_scores = []

# # for k in k_values:
# #     # Copy data to avoid contamination
# #     temp_df = data_df.copy()
    
# #     # Impute with current k
# #     imputer = KNNImputer(n_neighbors=k, weights='distance')
# #     temp_df['2018 Social Capital'] = imputer.fit_transform(temp_df[['2018 Social Capital', 'lat', 'lon']])[:, 0]
    
# #     # Calculate MAE for masked values
# #     imputed_values = temp_df.loc[mask, '2018 Social Capital']
# #     mae = mean_absolute_error(true_values, imputed_values)
# #     mae_scores.append(mae)
# #     print(f"k={k}: MAE = {mae:.4f}")
    
# # import matplotlib.pyplot as plt

# # plt.plot(k_values, mae_scores, marker='o')
# # plt.xlabel('Number of Neighbors (k)')
# # plt.ylabel('Mean Absolute Error (MAE)')
# # plt.title('KNN Imputation: Optimal k')
# # plt.grid(True)
# # plt.show()



# #############################################
# ## So the error in the above algorithm was ok but not great, around 0.55 MAE over the masked known values.
# ## Here I'm trying a cross-validation approach to test different masked sets to see if the MAE improves.
# #############################################

 
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# k_values = np.arange(2,101)#[3, 5, 7, 9, 11]
# cv_scores = {k: [] for k in k_values}

# for train_idx, test_idx in kf.split(non_missing_indices):
#     # Mask test indices
#     temp_df = data_df.copy()
#     temp_df.loc[non_missing_indices[test_idx], '2018 Social Capital'] = np.nan
    
#     for k in k_values:
#         imputer = KNNImputer(n_neighbors=k, weights='distance')
#         temp_df['2018 Social Capital'] = imputer.fit_transform(temp_df[['2018 Social Capital', 'lat', 'lon']])[:, 0]
#         mae = mean_absolute_error(
#             data_df.loc[non_missing_indices[test_idx], '2018 Social Capital'],
#             temp_df.loc[non_missing_indices[test_idx], '2018 Social Capital']
#         )
#         cv_scores[k].append(mae)

# # Average MAE for each k
# for k in k_values:
#     avg_mae = np.mean(cv_scores[k])
#     print(f"k={k}: Avg MAE = {avg_mae:.4f}")




##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
## These are some old functions I tried to use to impute the social capital values, but weren't working. Kept them for posterity.


# # Step 2: Extract FIPS codes and coordinates
# county_data = []
# for feat in county_geo['features']:
#     try:
#         # Extract FIPS code from GEO_ID (format: '0500000US01001' → '01001')
#         geo_id = feat['properties']['GEO_ID']
#         fips = geo_id.split('US')[-1].zfill(5)  # Ensures 5-digit FIPS
        
#         # Get first coordinate point (some counties have multiple polygons)
#         coord = feat['geometry']['coordinates'][0][0][0]
#         county_data.append({
#             'FIPS': fips,
#             'lon': coord[0],
#             'lat': coord[1]
#         })
#     except (KeyError, IndexError) as e:
#         print(f"Skipping feature due to error: {e}")
#         continue


# # Load county centroids (lat/lon)
# county_geo = gpd.read_file("https://public.opendatasoft.com/explore/dataset/us-county-boundaries/download/?format=geojson")
# county_geo['FIPS'] = county_geo['STATE'] + county_geo['COUNTY']
# county_geo = county_geo[['FIPS', 'geometry']].to_crs('EPSG:4326')
# county_geo['lon'] = county_geo.geometry.centroid.x
# county_geo['lat'] = county_geo.geometry.centroid.y

# # Merge with your data
# data_df = pd.merge(data_df, county_geo[['FIPS', 'lat', 'lon']], on='FIPS', how='left')

# # Impute using KNN (5 nearest counties)
# imputer = KNNImputer(n_neighbors=5, weights='distance')
# data_df['2018 Social Capital'] = imputer.fit_transform(data_df[['2018 Social Capital', 'lat', 'lon']])[:, 0]
# print(data_df.iloc[:10])

# # Step 1: Load county centroids (no GDAL required)
# county_centroids_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
# county_geo = pd.read_json(county_centroids_url)
# county_df = pd.DataFrame([
#     {
#         'FIPS': feat['properties']['fips'].zfill(5),  # Ensure 5-digit FIPS
#         'lat': feat['geometry']['coordinates'][1],
#         'lon': feat['geometry']['coordinates'][0]
#     }
#     for feat in county_geo['features']
# ])