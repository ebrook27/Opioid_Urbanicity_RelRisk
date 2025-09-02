import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import lognorm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
MORTALITY_PATH = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
MORTALITY_NAMES = ['FIPS'] + [f'{year} MR' for year in range(2010, 2023)]
PREDICTIONS_PATH = 'Autoencoder/Predictions/ae_mortality_predictions.csv'
PREDICTIONS_NAMES = [f'{year} Preds' for year in range(2011, 2023)]
ANOM_TYPE = 'Cold' 

def load_shapefile(shapefile_path):
    shape = gpd.read_file(shapefile_path)
    return shape

def load_mort_and_fips():
    mort_df = pd.DataFrame()
    mort_df = pd.read_csv(MORTALITY_PATH, header=0, names=MORTALITY_NAMES)
    mort_df['FIPS'] = mort_df['FIPS'].astype(str).apply(lambda x: x.zfill(5) if len(x) < 5 else x)
    mort_df[MORTALITY_NAMES[1:]] = mort_df[MORTALITY_NAMES[1:]].astype(float).clip(lower=0)
    mort_df = mort_df.sort_values(by='FIPS').reset_index(drop=True)
    fips_df = mort_df[['FIPS']]
    return mort_df, fips_df

def load_predictions(fips_df, preds_path=PREDICTIONS_PATH, preds_names=PREDICTIONS_NAMES):
    preds_df = pd.read_csv(preds_path, header=0, names=preds_names)
    preds_df[preds_names] = preds_df[preds_names].astype(float)

    # Initialize dictionaries to store the predicted means and standard deviations
    predicted_shapes = {}
    predicted_locations = {}
    predicted_scales = {}
    start_year = 2011

    # Extract the last three rows (mean and std) for each column
    for i, col in enumerate(preds_names):
        year = start_year + i
        predicted_shapes[year] = preds_df[col].iloc[-3]
        predicted_locations[year] = preds_df[col].iloc[-2]
        predicted_scales[year] = preds_df[col].iloc[-1]

    # Drop the last three rows from the DataFrame
    preds_df = preds_df.iloc[:-3].reset_index(drop=True)
    preds_df = pd.concat([fips_df, preds_df], axis=1)
    return preds_df, predicted_shapes, predicted_locations, predicted_scales

def shape_merges(shape, mort_df, preds_df):
    shape = shape.merge(mort_df, on='FIPS')
    shape = shape.merge(preds_df, on='FIPS')
    return shape

def construct_anomaly_accuracy_map(shape, predicted_shapes, predicted_locs, predicted_scales, year, anom_type=ANOM_TYPE):
    fig, main_ax = plt.subplots(figsize=(10, 5))
    title = f'{year} {anom_type} Anomaly Accuracy Map'
    plt.title(title, size=16, weight='bold')

    # Alaska and Hawaii insets
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4]) 
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])  
    
    # Plot state boundaries
    state_boundaries = shape.dissolve(by='STATEFP', as_index=False)
    state_boundaries.boundary.plot(ax=main_ax, edgecolor='black', linewidth=.5)

    alaska_state = state_boundaries[state_boundaries['STATEFP'] == '02']
    alaska_state.boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=.5)

    hawaii_state = state_boundaries[state_boundaries['STATEFP'] == '15']
    hawaii_state.boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=.5)

    # Define the insets for coloring
    shapes = [
        (shape[(shape['STATEFP'] != '02') & (shape['STATEFP'] != '15')], main_ax, 'continental'),
        (shape[shape['STATEFP'] == '02'], alaska_ax, 'alaska'),
        (shape[shape['STATEFP'] == '15'], hawaii_ax, 'hawaii') ]
    
    # Get thresholds for DATA anomalies
    mort_rates = shape[f'{year} MR'].values
    non_zero_mort_rates = mort_rates[mort_rates > 0]
    data_params = lognorm.fit(non_zero_mort_rates)
    data_shape, data_loc, data_scale = data_params

    data_upper_thresh = lognorm.ppf(.98, data_shape, data_loc, data_scale)
    data_lower_thresh = lognorm.ppf(.02, data_shape, data_loc, data_scale)

    # Get thresholds for PREDICTED anomalies
    pred_shape = predicted_shapes[year]
    pred_loc = predicted_locs[year]
    pred_scale = predicted_scales[year]

    pred_upper_thresh = lognorm.ppf(.98, pred_shape, pred_loc, pred_scale)
    pred_lower_thresh = lognorm.ppf(.02, pred_shape, pred_loc, pred_scale)

    for inset, ax, _ in shapes:
        for _, row in inset.iterrows():
            county = row['FIPS']
            pred_value = row[f'{year} Preds']
            data_value = row[f'{year} MR']

            if anom_type == 'Hot':

                if data_value > data_upper_thresh: # WAS hot
                    if pred_value > pred_upper_thresh: # hot MATCH
                        color = 'red'
                    elif pred_value <= pred_upper_thresh: 
                        color = 'black' # missed hot anomaly

                elif data_value <= data_upper_thresh: # NOT hot
                    if pred_value > pred_upper_thresh: # incorrectly predicted hot anomaly
                        color = 'grey'
                    elif pred_value <= pred_upper_thresh: 
                        color = 'lightgrey' 

            elif anom_type == 'Cold':

                if 0 < data_value < data_lower_thresh: # WAS cold
                    if 0 < pred_value < pred_lower_thresh: # cold MATCH
                        color = 'blue'
                    elif pred_value >= pred_lower_thresh: 
                        color = 'black' # missed cold anomaly

                elif data_value >= data_lower_thresh: # NOT cold
                    if 0 < pred_value < pred_lower_thresh: # incorrectly predicted cold anomaly
                        color = 'grey'
                    elif pred_value >= pred_lower_thresh: 
                        color = 'lightgrey' 
                
            inset[inset['FIPS'] == county].plot(ax=ax, color=color)

    # Adjust the viewing
    set_view_window(main_ax,alaska_ax,hawaii_ax)

    # Add the colorbar
    add_legend(main_ax, anom_type)

    outputh_path = f'Autoencoder/Efficacy/Anomaly Accuracy Maps/{anom_type}/{year}_{anom_type}_anom_acc_map'
    plt.savefig(outputh_path, bbox_inches=None, pad_inches=0, dpi=300)
    # plt.show()
    plt.close(fig)

def set_view_window(main_ax,alaska_ax,hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    # Fix window
    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def add_legend(main_ax, anom_type):
    if anom_type == 'Hot':
        correct_patch = mpatches.Patch(color='red', label='Correct')
    elif anom_type == 'Cold':
        correct_patch = mpatches.Patch(color='blue', label='Correct')
    missed_patch = mpatches.Patch(color='black', label='Missed')
    wrong_patch = mpatches.Patch(color='grey', label='Incorrect')
    main_ax.legend(handles=[correct_patch, missed_patch, wrong_patch], loc='lower right', bbox_to_anchor=(1, 0))

def main():
    shape = load_shapefile(SHAPE_PATH)
    mort_df, fips_df = load_mort_and_fips()
    preds_df, predicted_shapes, predicted_locs, predicted_scales = load_predictions(fips_df)
    shape = shape_merges(shape, mort_df, preds_df)

    for year in range(2011, 2023):
        construct_anomaly_accuracy_map(shape, predicted_shapes, predicted_locs, predicted_scales, year, anom_type=ANOM_TYPE)
        print(f'{ANOM_TYPE} acc map printed for {year}.')

if __name__ == "__main__":
    main()