### 6/18/25, EB: Adam suggested we look at the raw mortality rates within each category, so we can compare to our predictions, and see if our high error counties have any overlap with the high mortality counties.
### I think this will be a good visualization regardless.

### 6/23/25, EB: I've added a bit more functionality to this script. We have the first attempt, that had a variable color scale for each map, not super useful.
### Now we have a few other functions that do different things: One has two modes: one to producea consistent color scale for each category, so we can compare across years. 
### The other mode has a consistent color scale across the whole dataset, so we can compare across categories and years.
### Finally, we have a copy of the previous function with two modes, but only plots the Continental US.


import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIG ──────────────────────────────────────────────────────────────────
SHAPE_PATH     = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
MORTALITY_CSV  = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
URBANICITY_CSV = 'Data/SVI/NCHS_urban_v_rural.csv'
OUT_DIR        = 'County_Category_Maps/Mortality_By_Urbanicity_Maps_All_Years/LogNormal_per_Year/'
TARGET_YEARS   = range(2010, 2023)  # 2010 through 2022
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD BASE DATA ──────────────────────────────────────────────────────────
shape = gpd.read_file(SHAPE_PATH)
shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)

mort = pd.read_csv(MORTALITY_CSV, dtype={'FIPS': str})
mort['FIPS'] = mort['FIPS'].str.zfill(5)

urb  = pd.read_csv(URBANICITY_CSV, dtype={'FIPS': str})
urb['FIPS'] = urb['FIPS'].str.zfill(5)
urb = urb[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'county_class'})


# ── PRE-COMPUTE CATEGORY RANGES ────────────────────────────────────────────
### 6/23/25, EB: This is used to make the per-category color bar. This way we can compare across years, within each category.
cat_ranges = {}
for cat in urb['county_class'].dropna().unique():
    cat_rows = (
        mort.merge(urb, on='FIPS', how='left')
            .query("county_class == @cat")
            .set_index('FIPS')
    )
    vals = cat_rows[[f"{y} MR" for y in TARGET_YEARS if f"{y} MR" in mort.columns]].values.ravel()
    vals = pd.to_numeric(vals, errors='coerce')
    cat_ranges[str(cat)] = (np.nanmin(vals), np.nanmax(vals))

### 6/23/25, EB: This is used to make the whole-dataset color bar, so we can compare across categories and years.
all_vals = mort[[f"{y} MR" for y in TARGET_YEARS if f"{y} MR" in mort.columns]].values.ravel()
all_vals = pd.to_numeric(all_vals, errors='coerce')
V_MIN, V_MAX = np.nanmin(all_vals), np.nanmax(all_vals)
GLOBAL_NORM  = BoundaryNorm(np.linspace(V_MIN, V_MAX, 21), plt.get_cmap('Reds', 20).N)


# ── HELPER: plot for one year and one category ──────────────────────────────
def plot_category_for_year(year, cat_code):
    mort_col = f'{year} MR'
    if mort_col not in mort.columns:
        print(f"⚠️ Mortality column '{mort_col}' not found. Skipping.")
        return

    df = shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left') \
              .merge(urb, on='FIPS', how='left') \
              .rename(columns={mort_col: 'Mortality'})

    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')
    df['PlotValue'] = df.apply(
        lambda row: row['Mortality'] if str(row['county_class']) == str(cat_code) else np.nan,
        axis=1
    )

    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f"{year} Mortality – Urbanicity Category {cat_code}", fontsize=14, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_bounds = df.dissolve(by='STATEFP', as_index=False)
    state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

    cmap = plt.get_cmap('Reds', 20)

    for inset, ax in [
        (df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')], main_ax),
        (df[df['STATEFP'] == '02'], alaska_ax),
        (df[df['STATEFP'] == '15'], hawaii_ax)
    ]:
        inset.plot(
            ax=ax,
            column='PlotValue',
            cmap=cmap,
            edgecolor='black',
            linewidth=0.1,
            missing_kwds={'color': 'lightgrey'}
        )
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    vmax = df['PlotValue'].max(skipna=True)
    vmin = df['PlotValue'].min(skipna=True)
    bounds = np.linspace(vmin, vmax, 21)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Mortality Rate', fontsize=10, weight='bold')

    out_path = os.path.join(OUT_DIR, f'Urbanicity_{cat_code}_Mortality_{year}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_path}")

def plot_category_for_year_normalized(year, cat_code):
    mort_col = f'{year} MR'
    if mort_col not in mort.columns:
        print(f"⚠️ Mortality column '{mort_col}' not found. Skipping.")
        return

    df = shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left') \
              .merge(urb, on='FIPS', how='left') \
              .rename(columns={mort_col: 'Mortality'})

    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')

    # Normalize within this category only
    is_target = df['county_class'].astype(str) == str(cat_code)
    df['PlotValue'] = np.nan

    cat_vals = df.loc[is_target, 'Mortality']
    if cat_vals.isnull().all():
        print(f"⚠️ No mortality data for category {cat_code} in {year}. Skipping.")
        return

    min_val, max_val = cat_vals.min(), cat_vals.max()
    if min_val == max_val:
        norm_vals = np.ones_like(cat_vals)  # all same color
    else:
        norm_vals = (cat_vals - min_val) / (max_val - min_val)

    df.loc[is_target, 'PlotValue'] = norm_vals

    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f"{year} Normalized Mortality – Urbanicity Category {cat_code}", fontsize=14, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_bounds = df.dissolve(by='STATEFP', as_index=False)
    state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

    cmap = plt.get_cmap('Reds', 20)
    bounds = np.linspace(0, 1, 21)
    norm = BoundaryNorm(bounds, cmap.N)

    for inset, ax in [
        (df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')], main_ax),
        (df[df['STATEFP'] == '02'], alaska_ax),
        (df[df['STATEFP'] == '15'], hawaii_ax)
    ]:
        inset.plot(
            ax=ax,
            column='PlotValue',
            cmap=cmap,
            edgecolor='black',
            linewidth=0.1,
            norm=norm,
            missing_kwds={'color': 'lightgrey'}
        )
        ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Mortality Rate (within category)', fontsize=10, weight='bold')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(['Low', '', 'Mid', '', 'High'])

    out_path = os.path.join(OUT_DIR, f'Urbanicity_{cat_code}_Mortality_{year}_Normalized.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {out_path}")

def plot_category_for_year_fixed(year, cat_code, norm=None):
    """
    This one should have a consistent color scale within each category, so we can compare across years.
    """
    vmin, vmax = cat_ranges[str(cat_code)]   # ← consistent range
    mort_col   = f'{year} MR'
    if mort_col not in mort.columns:
        return

    df = (shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
               .merge(urb, on='FIPS', how='left')
               .rename(columns={mort_col: 'Mortality'}))
    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')
    df['PlotValue'] = df.apply(
        lambda r: r['Mortality'] if str(r['county_class']) == str(cat_code) else np.nan, axis=1
    )

    fig, main_ax = plt.subplots(figsize=(10, 5))
    plt.title(f"{year} Mortality – Urbanicity {cat_code}", fontsize=14)
    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    state_bounds = df.dissolve(by='STATEFP', as_index=False)
    state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

    cmap  = plt.get_cmap('Reds', 20)
    ### 6/23/25, EB: Added this if statement to allow for a global norm, if desired.
    if norm is None:
        norm  = BoundaryNorm(np.linspace(vmin, vmax, 21), cmap.N)

    ### 6/23/25, EB: Troubleshooting the outline of the states 
    # for inset, ax in [
    #     (df[(df['STATEFP']!='02') & (df['STATEFP']!='15')], main_ax),
    #     (df[df['STATEFP']=='02'], alaska_ax),
    #     (df[df['STATEFP']=='15'], hawaii_ax)
    # ]:
    #     inset.plot(ax=ax, column='PlotValue', cmap=cmap, norm=norm,
    #                edgecolor='black', linewidth=.1,
    #                missing_kwds={'color':'lightgrey'})
    #     ax.axis('off')
    
    # Plot counties first
    for inset, ax in [
        (df[(df['STATEFP']!='02') & (df['STATEFP']!='15')], main_ax),
        (df[df['STATEFP']=='02'], alaska_ax),
        (df[df['STATEFP']=='15'], hawaii_ax)
    ]:
        inset.plot(ax=ax, column='PlotValue', cmap=cmap, norm=norm,
                edgecolor='black', linewidth=.1,
                missing_kwds={'color':'lightgrey'})
        ax.axis('off')

    # THEN overlay state borders (important: do this AFTER the above)
    state_bounds = df.dissolve(by='STATEFP', as_index=False)
    state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

    main_ax.set_xlim([-125, -65]); main_ax.set_ylim([25, 50])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=main_ax, fraction=.046, pad=.04)
    cbar.set_label('Mortality Rate', fontsize=10)

    out = os.path.join(OUT_DIR, f'Global_Cat{cat_code}_{year}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()

# def plot_category_for_year_CONUS(year, cat_code, norm=None):
#     """
#     This version plots only the continental US, and has a consistent color scale across the whole dataset.
#     """
#     vmin, vmax = cat_ranges[str(cat_code)]  # consistent per-category range
#     mort_col   = f'{year} MR'
#     if mort_col not in mort.columns:
#         return

#     # Merge and prep
#     df = (
#         shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
#              .merge(urb, on='FIPS', how='left')
#              .rename(columns={mort_col: 'Mortality'})
#     )
#     df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')
#     df['PlotValue'] = df.apply(
#         lambda r: r['Mortality'] if str(r['county_class']) == str(cat_code) else np.nan, axis=1
#     )

#     # Keep only continental US (exclude AK and HI)
#     df = df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')]

#     # Plot setup
#     fig, ax = plt.subplots(figsize=(10, 5))
#     plt.title(f"{year} Mortality – Urbanicity {cat_code}", fontsize=14)

#     cmap = plt.get_cmap('Reds', 20)
#     if norm is None:
#         norm = BoundaryNorm(np.linspace(vmin, vmax, 21), cmap.N)

#     # Plot
#     df.plot(
#         ax=ax,
#         column='PlotValue',
#         cmap=cmap,
#         norm=norm,
#         edgecolor='black',
#         linewidth=0.1,
#         missing_kwds={'color': 'lightgrey'}
#     )

#     ax.axis('off')
#     ax.set_xlim([-125, -65])
#     ax.set_ylim([25, 50])

#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Mortality Rate', fontsize=10)

#     out = os.path.join(OUT_DIR, f'CONUS_Global_Cat{cat_code}_{year}.png')
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     plt.close()

def plot_category_for_year_CONUS(year, cat_code, norm=None):
    """
    This version plots only the continental US, and has a consistent color scale across the whole dataset.
    This one should also plot the state outlines, unlike the previous version.
    """
    vmin, vmax = cat_ranges[str(cat_code)]  # consistent per-category range
    mort_col   = f'{year} MR'
    if mort_col not in mort.columns:
        return

    # Merge and prep
    df = (
        shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
             .merge(urb, on='FIPS', how='left')
             .rename(columns={mort_col: 'Mortality'})
    )
    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')
    df['PlotValue'] = df.apply(
        lambda r: r['Mortality'] if str(r['county_class']) == str(cat_code) else np.nan, axis=1
    )

    # Keep only continental US (exclude AK and HI)
    df = df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')]

    # Create state boundaries from county shapes
    state_bounds = df.dissolve(by='STATEFP', as_index=False)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(f"{year} Mortality – Urbanicity {cat_code}", fontsize=14)

    cmap = plt.get_cmap('Reds', 20)
    if norm is None:
        norm = BoundaryNorm(np.linspace(vmin, vmax, 21), cmap.N)

    # Plot counties
    df.plot(
        ax=ax,
        column='PlotValue',
        cmap=cmap,
        norm=norm,
        edgecolor='black',
        linewidth=0.1,
        missing_kwds={'color': 'lightgrey'}
    )

    # Overlay state outlines
    state_bounds.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)

    # Clean up
    ax.axis('off')
    ax.set_xlim([-125, -65])
    ax.set_ylim([25, 50])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mortality Rate', fontsize=10)

    out = os.path.join(OUT_DIR, f'CONUS_WithinCat_Scale_Cat{cat_code}_{year}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

### 6/27/25, EB: Adam mentioned fitting a log-normal to each category, and coloring counties based off of the distribution.
### The following functions are a first attempt at that.
from scipy.stats import lognorm

epsilon = 0.01  # A small value, appropriate given your rate scale
def fit_lognormal_by_category(mort, urb, target_years):
    """
    Fit a log-normal distribution to mortality rates for each category over all target years.
    Returns a dict: {category_code: (shape, loc, scale)}.
    """
    dist_params = {}

    for cat in sorted(urb['county_class'].dropna().unique()):
        cat_mortality = []

        for year in target_years:
            mort_col = f'{year} MR'
            if mort_col not in mort.columns:
                continue
            merged = mort[['FIPS', mort_col]].merge(urb, on='FIPS', how='left')
            filtered = merged[merged['county_class'].astype(str) == str(cat)]
            # cat_mortality.extend(filtered[mort_col].dropna().astype(float).values)
            values = pd.to_numeric(filtered[mort_col], errors='coerce').dropna()

            # ✅ Add epsilon to make all values strictly positive
            # Replace only the zero values
            values = values.copy()
            values[values == 0] = epsilon
            cat_mortality.extend(values)

        if cat_mortality:
            shape, loc, scale = lognorm.fit(cat_mortality, floc=0)
            dist_params[str(cat)] = (shape, loc, scale)

    return dist_params

def plot_category_for_year_lognormal(year, cat_code, mort, shape, urb, dist_params, output_dir):
    """
    Plot counties in a category colored by percentile rank in the category's fitted log-normal.
    """
    mort_col = f'{year} MR'
    if mort_col not in mort.columns:
        return

    df = (
        shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
             .merge(urb, on='FIPS', how='left')
             .rename(columns={mort_col: 'Mortality'})
    )

    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')

    # Apply log-normal CDF if parameters available
    if str(cat_code) not in dist_params:
        print(f"⚠️ No log-normal fit available for category {cat_code}. Skipping.")
        return

    shape_p, loc_p, scale_p = dist_params[str(cat_code)]

    df['PlotValue'] = df.apply(
        lambda r: lognorm.cdf(r['Mortality'], shape_p, loc=loc_p, scale=scale_p)
        if str(r['county_class']) == str(cat_code) and not pd.isna(r['Mortality'])
        else np.nan,
        axis=1
    )
    df['PlotValue'] = df['PlotValue'].clip(0.01, 0.99)

    # Only continental US
    df = df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')]

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(np.linspace(0, 1, 21), cmap.N)

    df.plot(
        ax=ax, column='PlotValue', cmap=cmap, norm=norm,
        edgecolor='black', linewidth=0.1,
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
    )

    ax.set_title(f"{year} Mortality Percentile – Urbanicity {cat_code}", fontsize=13)
    ax.axis('off')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Log-normal Percentile')

    out_path = os.path.join(output_dir, f"Cat{cat_code}_{year}_lognormal_percentile.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


### 6/27/25, EB: Fitting a distribution to each category in the previous to functions worked, but now I want to try fitting a log-normal to each year,
### across the categories, and then plotting the percentiles for each county, within the categories like before. So the only thing that should
### change is the coloring.



#### imports and functions for GIF generation, didn't work how I wanted it to.
# import imageio.v2 as imageio
# from io import BytesIO

# def generate_gif_for_category(cat_code, years, out_path):
#     frames = []

#     for year in years:
#         mort_col = f'{year} MR'
#         if mort_col not in mort.columns:
#             continue

#         df = shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left') \
#                   .merge(urb, on='FIPS', how='left') \
#                   .rename(columns={mort_col: 'Mortality'})
#         df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')

#         # Normalize within this category
#         is_target = df['county_class'].astype(str) == str(cat_code)
#         df['PlotValue'] = np.nan
#         cat_vals = df.loc[is_target, 'Mortality']
#         if cat_vals.isnull().all():
#             continue

#         min_val, max_val = cat_vals.min(), cat_vals.max()
#         norm_vals = np.ones_like(cat_vals) if min_val == max_val else (cat_vals - min_val) / (max_val - min_val)
#         df.loc[is_target, 'PlotValue'] = norm_vals

#         # Plot
#         fig, main_ax = plt.subplots(figsize=(10, 5))
#         plt.title(f"{year} – Urbanicity Category {cat_code}", fontsize=14)

#         alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
#         hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])
#         cmap = plt.get_cmap('Reds', 20)
#         bounds = np.linspace(0, 1, 21)
#         norm = BoundaryNorm(bounds, cmap.N)

#         state_bounds = df.dissolve(by='STATEFP', as_index=False)
#         for ax, state in zip([main_ax, alaska_ax, hawaii_ax], ['main', '02', '15']):
#             state_bounds[state_bounds['STATEFP'] == state if state != 'main' else slice(None)]\
#                 .boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
#             ax.axis('off')

#         for inset, ax in [
#             (df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')], main_ax),
#             (df[df['STATEFP'] == '02'], alaska_ax),
#             (df[df['STATEFP'] == '15'], hawaii_ax)
#         ]:
#             inset.plot(
#                 ax=ax,
#                 column='PlotValue',
#                 cmap=cmap,
#                 edgecolor='black',
#                 linewidth=0.1,
#                 norm=norm,
#                 missing_kwds={'color': 'lightgrey'}
#             )

#         main_ax.set_xlim([-125, -65])
#         main_ax.set_ylim([25, 50])
#         sm = ScalarMappable(cmap=cmap, norm=norm)
#         cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
#         cbar.set_label('Normalized Mortality Rate', fontsize=10)
#         cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
#         cbar.ax.set_yticklabels(['Low', '', 'Mid', '', 'High'])

#         # Save current figure to memory
#         buf = BytesIO()
#         plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         buf.seek(0)
#         frames.append(imageio.imread(buf))

#     # Save GIF
#     imageio.mimsave(out_path, frames, duration=1.0)
#     print(f"✅ Saved GIF for category {cat_code}: {out_path}")

def fit_lognormal_by_year(mort, target_years, epsilon=0.01):
    """
    Fit a log-normal distribution to mortality rates for each year across all counties.
    Returns a dict: {year: (shape, loc, scale)}.
    """
    dist_params = {}

    for year in target_years:
        mort_col = f'{year} MR'
        if mort_col not in mort.columns:
            continue

        values = pd.to_numeric(mort[mort_col], errors='coerce').dropna().copy()
        values[values == 0] = epsilon  # Replace 0s with epsilon

        try:
            shape, loc, scale = lognorm.fit(values, floc=0)
            dist_params[year] = (shape, loc, scale)
        except Exception as e:
            print(f"⚠️ Could not fit year {year}: {e}")

    return dist_params

def plot_category_for_year_lognormal_peryear(year, cat_code, mort, shape, urb, dist_params, output_dir):
    """
    Plot counties in a given category for a given year,
    colored by percentile under that year's log-normal distribution.
    """
    mort_col = f'{year} MR'
    if mort_col not in mort.columns or year not in dist_params:
        print(f"⚠️ Missing data or distribution for year {year}.")
        return

    # Merge data
    df = (
        shape.merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
             .merge(urb, on='FIPS', how='left')
             .rename(columns={mort_col: 'Mortality'})
    )

    df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')

    shape_p, loc_p, scale_p = dist_params[year]

    # Apply log-normal CDF only for selected category
    df['PlotValue'] = df.apply(
        lambda r: lognorm.cdf(r['Mortality'], shape_p, loc=loc_p, scale=scale_p)
        if str(r['county_class']) == str(cat_code) and not pd.isna(r['Mortality'])
        else np.nan,
        axis=1
    )
    df['PlotValue'] = df['PlotValue'].clip(0.01, 0.99)

    # Exclude Alaska and Hawaii
    df = df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(np.linspace(0, 1, 21), cmap.N)

    df.plot(
        ax=ax, column='PlotValue', cmap=cmap, norm=norm,
        edgecolor='black', linewidth=0.1,
        missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
    )

    ax.set_title(f"{year} Mortality Percentile – Urbanicity {cat_code}", fontsize=13)
    ax.axis('off')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Log-normal Percentile (year-wide)')

    out_path = os.path.join(output_dir, f"Cat{cat_code}_{year}_lognormal_peryear.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()



def main():
    ### This script makes and saves the maps that has a standardized color scale for each category, so we can compare across years.
    # categories = sorted(urb['county_class'].dropna().unique())
    # for year in TARGET_YEARS:
    #     for cat in categories:
    #         plot_category_for_year_CONUS(year, cat, norm=None)
    # print("✅ All maps generated.")

    ### This script fits a log-normal distribution to each category, and plots the percentiles for each county.
    categories = sorted(urb['county_class'].dropna().unique())
    dist_params = fit_lognormal_by_year(mort, TARGET_YEARS, epsilon=0.01)

    for year in TARGET_YEARS:
        for cat in categories:
            plot_category_for_year_lognormal_peryear(
                year=year,
                cat_code=cat,
                mort=mort,
                shape=shape,
                urb=urb,
                dist_params=dist_params,
                output_dir=OUT_DIR
            )

    print("✅ All log-normal percentile maps generated.")


# ── MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

# ### GIF loop:
# if __name__ == "__main__":
#     shape = gpd.read_file(SHAPE_PATH)
#     shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)
#     mort = pd.read_csv('Data/Mortality/Final Files/Mortality_final_rates.csv', dtype={'FIPS': str})
#     mort['FIPS'] = mort['FIPS'].str.zfill(5)
#     urb = pd.read_csv('Data/SVI/NCHS_urban_v_rural.csv', dtype={'FIPS': str})
#     urb['FIPS'] = urb['FIPS'].str.zfill(5)

#     years = list(range(2010, 2023))
#     os.makedirs('County_Category_Maps\Mortality_By_Urbanicity_Maps_All_Years/Normalized within Category/Mortality_GIFs_By_Category', exist_ok=True)

#     for cat in range(1, 7):
#         gif_path = f'Mortality_GIFs_By_Category/Urbanicity_Category_{cat}.gif'
#         generate_gif_for_category(cat, years, gif_path)





# ##############################################################################################################################################
# ### Quick, unnormalized GIF attempt


# import os
# import pandas as pd
# import numpy as np
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from matplotlib.colors import BoundaryNorm, Normalize
# from matplotlib.cm import ScalarMappable
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# import imageio.v2 as imageio
# from io import BytesIO

# # ── CONFIG ──────────────────────────────────────────────────────────────────
# SHAPE_PATH     = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
# MORTALITY_CSV  = 'Data/Mortality/Final Files/Mortality_final_rates.csv'
# URBANICITY_CSV = 'Data/SVI/NCHS_urban_v_rural.csv'
# OUT_DIR        = 'County_Category_Maps\Mortality_By_Urbanicity_Maps_All_Years/Unnormalized_GIFs'
# TARGET_YEARS   = range(2010, 2023)   # 2010-2022
# os.makedirs(OUT_DIR, exist_ok=True)

# # ── LOAD BASE DATA ──────────────────────────────────────────────────────────
# shape = gpd.read_file(SHAPE_PATH)
# shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)

# mort = pd.read_csv(MORTALITY_CSV, dtype={'FIPS': str})
# mort['FIPS'] = mort['FIPS'].str.zfill(5)

# urb  = pd.read_csv(URBANICITY_CSV, dtype={'FIPS': str})
# urb['FIPS'] = urb['FIPS'].str.zfill(5)
# urb = urb[['FIPS', '2023 Code']].rename(columns={'2023 Code': 'county_class'})

# # ── GIF GENERATOR (RAW MORTALITY) ───────────────────────────────────────────
# def generate_gif_for_category(cat_code, years, out_path):
#     frames = []

#     # compute vmin/vmax *once* for consistent color scale inside this category
#     cat_rows = mort.merge(urb, on='FIPS', how='left')
#     cat_rows = cat_rows[cat_rows['county_class'].astype(str) == str(cat_code)]
#     vmin = cat_rows[[f'{y} MR' for y in years if f'{y} MR' in mort.columns]].min().min()
#     vmax = cat_rows[[f'{y} MR' for y in years if f'{y} MR' in mort.columns]].max().max()

#     cmap  = plt.get_cmap('Reds', 20)
#     norm  = Normalize(vmin=vmin, vmax=vmax)

#     for year in years:
#         mort_col = f'{year} MR'
#         if mort_col not in mort.columns:
#             continue

#         df = (
#             shape
#             .merge(mort[['FIPS', mort_col]], on='FIPS', how='left')
#             .merge(urb, on='FIPS', how='left')
#             .rename(columns={mort_col: 'Mortality'})
#         )
#         df['Mortality'] = pd.to_numeric(df['Mortality'], errors='coerce')

#         # keep raw mortality for this category; grey others
#         df['PlotValue'] = df.apply(
#             lambda r: r['Mortality'] if str(r['county_class']) == str(cat_code) else np.nan, axis=1
#         )

#         # ── plotting ──
#         fig, main_ax = plt.subplots(figsize=(10, 5))
#         plt.title(f"{year} – Urbanicity {cat_code}", fontsize=14)

#         alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
#         hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

#         state_bounds = df.dissolve(by='STATEFP', as_index=False)
#         state_bounds.boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
#         state_bounds[state_bounds['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
#         state_bounds[state_bounds['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

#         for inset, ax in [
#             (df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')], main_ax),
#             (df[df['STATEFP'] == '02'], alaska_ax),
#             (df[df['STATEFP'] == '15'], hawaii_ax)
#         ]:
#             inset.plot(
#                 ax=ax,
#                 column='PlotValue',
#                 cmap=cmap,
#                 edgecolor='black',
#                 linewidth=0.1,
#                 norm=norm,
#                 missing_kwds={'color': 'lightgrey'}
#             )
#             ax.axis('off')

#         main_ax.set_xlim([-125, -65])
#         main_ax.set_ylim([25, 50])

#         sm = ScalarMappable(norm=norm, cmap=cmap)
#         cbar = plt.colorbar(sm, ax=main_ax, orientation='vertical', fraction=0.046, pad=0.04)
#         cbar.set_label('Mortality Rate (raw)', fontsize=10)

#         buf = BytesIO()
#         plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
#         plt.close(fig)
#         buf.seek(0)
#         frames.append(imageio.imread(buf))

#     imageio.mimsave(out_path, frames, duration=1.0)
#     print(f"✅ Saved GIF for category {cat_code}: {out_path}")

# # ── MAIN ────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     years = list(TARGET_YEARS)
#     for cat in sorted(urb['county_class'].dropna().unique()):
#         gif_path = os.path.join(OUT_DIR, f'Urbanicity_{cat}_Mortality_Raw.gif')
#         generate_gif_for_category(cat, years, gif_path)


