### 5/27/25, EB: Here I am plotting just the coutny categories, not any model results.
### This code should make a map showing each county category in its own map, with all others greyed out,
### and one map showing all counties, each category colored a different color.

# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import warnings
# import numpy as np
# warnings.filterwarnings("ignore", category=UserWarning)

# # === CONFIG ===
# SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
# CLASS_PATH = 'Data/SVI/NCHS_urban_v_rural.csv'
# OUT_DIR = 'County Classification/County_Category_Maps'
# os.makedirs(OUT_DIR, exist_ok=True)

# # === Load shapefile and county classes ===
# def load_data():
#     shape = gpd.read_file(SHAPE_PATH)
#     shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)

#     class_df = pd.read_csv(CLASS_PATH, dtype={'FIPS': str})
#     class_df['FIPS'] = class_df['FIPS'].str.zfill(5)
#     class_df['county_class'] = class_df['2023 Code'].astype(str)

#     merged = shape.merge(class_df[['FIPS', 'county_class']], on='FIPS', how='left')
#     return merged

# # === Base plotting function ===
# def plot_base_us(ax, merged):
#     state_boundaries = merged.dissolve(by='STATEFP', as_index=False)
#     state_boundaries.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
#     ax.axis('off')
#     ax.set_xlim([-125, -65])
#     ax.set_ylim([25, 50])

# # === Plot each category map separately ===
# def plot_individual_category_maps(merged):
#     categories = sorted(merged['county_class'].dropna().unique())

#     for cat in categories:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plt.title(f'Urbanicity Category {cat}', fontsize=14, weight='bold')

#         merged['highlight'] = merged['county_class'].apply(lambda x: 1 if x == cat else np.nan)

#         merged.plot(ax=ax, column='highlight', cmap='tab10', edgecolor='black',
#                     linewidth=0.1, missing_kwds={"color": "lightgrey"})

#         plot_base_us(ax, merged)
#         plt.savefig(f"{OUT_DIR}/Urbanicity_Category_{cat}_Map.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ Saved: Urbanicity_Category_{cat}_Map.png")

# # === Plot combined county category map ===
# def plot_combined_category_map(merged):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.title("County Urbanicity Categories", fontsize=14, weight='bold')

#     merged.plot(ax=ax, column='county_class', cmap='RdYlBu', edgecolor='black', linewidth=0.1)

#     ### Adding a legend
#     import matplotlib.patches as mpatches

#     # Get unique categories and colormap
#     unique_classes = sorted(merged['county_class'].dropna().unique())
#     cmap = plt.get_cmap('RdYlBu', len(unique_classes))
#     colors = [cmap(i) for i in range(len(unique_classes))]

#     # Create legend patches
#     legend_patches = [
#         mpatches.Patch(color=colors[i], label=f'Category {unique_classes[i]}')
#         for i in range(len(unique_classes))
#     ]

#     # Add the legend to your main axis
#     ax.legend(
#         handles=legend_patches,
#         title='County Category',
#         loc='lower left',
#         bbox_to_anchor=(1.02, 0.5),
#         borderaxespad=0.,
#         frameon=True
#     )

#     plot_base_us(ax, merged)
#     plt.savefig(f"{OUT_DIR}/All_County_Categories_Map.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("✅ Saved: All_County_Categories_Map.png")

# # === Main ===
# def main():
#     merged = load_data()
#     plot_individual_category_maps(merged)
#     plot_combined_category_map(merged)

# if __name__ == "__main__":
#     main()


# import geopandas as gpd
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import warnings
# import numpy as np
# import matplotlib.patches as mpatches

# warnings.filterwarnings("ignore", category=UserWarning)

# # === CONFIG ===
# SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
# CLASS_PATH = 'Data/SVI/NCHS_urban_v_rural.csv'
# OUT_DIR = 'County Classification/County_Category_Maps'
# os.makedirs(OUT_DIR, exist_ok=True)

# def load_data():
#     shape = gpd.read_file(SHAPE_PATH)
#     shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)

#     class_df = pd.read_csv(CLASS_PATH, dtype={'FIPS': str})
#     class_df['FIPS'] = class_df['FIPS'].str.zfill(5)
#     class_df['county_class'] = class_df['2023 Code'].astype(str)

#     merged = shape.merge(class_df[['FIPS', 'county_class']], on='FIPS', how='left')
#     return merged

# # === Add Alaska and Hawaii as insets ===
# def add_insets(fig, merged, ax_main, column, cmap, legend=False):
#     # Alaska
#     ax_ak = fig.add_axes([0.02, -0.1, 0.2, 0.2])
#     ak = merged[merged['STATEFP'] == '02']
#     ak.plot(ax=ax_ak, column=column, cmap=cmap, edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})
#     ax_ak.axis('off')

#     # Hawaii
#     ax_hi = fig.add_axes([0.25, 0.03, 0.08, 0.08])
#     hi = merged[merged['STATEFP'] == '15']
#     hi.plot(ax=ax_hi, column=column, cmap=cmap, edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})
#     ax_hi.axis('off')

# def plot_base_us(ax, merged):
#     state_boundaries = merged.dissolve(by='STATEFP', as_index=False)
#     state_boundaries = state_boundaries[~state_boundaries['STATEFP'].isin(['02', '15'])]  # exclude AK and HI
#     state_boundaries.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
#     ax.axis('off')
#     ax.set_xlim([-125, -65])
#     ax.set_ylim([25, 50])

# def plot_individual_category_maps(merged):
#     categories = sorted(merged['county_class'].dropna().unique())

#     for cat in categories:
#         fig, ax = plt.subplots(figsize=(10, 6))
#         plt.title(f'Urbanicity Category {cat}', fontsize=14, weight='bold')

#         merged['highlight'] = merged['county_class'].apply(lambda x: 1 if x == cat else np.nan)
#         contig = merged[~merged['STATEFP'].isin(['02', '15'])]

#         contig.plot(ax=ax, column='highlight', cmap='Set2', edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})
#         plot_base_us(ax, merged)
#         add_insets(fig, merged, ax, column='highlight', cmap='Set2')

#         plt.savefig(f"{OUT_DIR}/Urbanicity_Category_{cat}_Map.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ Saved: Urbanicity_Category_{cat}_Map.png")

# def plot_combined_category_map(merged):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.title("County Urbanicity Categories", fontsize=14, weight='bold')

#     contig = merged[~merged['STATEFP'].isin(['02', '15'])]
#     contig.plot(ax=ax, column='county_class', cmap='Set2', edgecolor='black', linewidth=0.1)

#     plot_base_us(ax, merged)
#     add_insets(fig, merged, ax, column='county_class', cmap='Set2')

#     # Legend
#     unique_classes = sorted(merged['county_class'].dropna().unique())
#     cmap = plt.get_cmap('Set2', len(unique_classes))
#     colors = [cmap(i) for i in range(len(unique_classes))]
#     legend_patches = [mpatches.Patch(color=colors[i], label=f'Category {unique_classes[i]}') for i in range(len(unique_classes))]

#     ax.legend(
#         handles=legend_patches,
#         title='County Category',
#         loc='lower left',
#         bbox_to_anchor=(1.02, 0.5),
#         borderaxespad=0.,
#         frameon=True
#     )

#     plt.savefig(f"{OUT_DIR}/All_County_Categories_Map.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("✅ Saved: All_County_Categories_Map.png")

# def main():
#     merged = load_data()
#     plot_individual_category_maps(merged)
#     plot_combined_category_map(merged)

# if __name__ == "__main__":
#     main()


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIG ===
SHAPE_PATH = '2022 USA County Shapefile/Filtered Files/2022_filtered_shapefile.shp'
CLASS_PATH = 'Data/SVI/NCHS_urban_v_rural.csv'
OUT_DIR = 'County Classification/County_Category_Maps'
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    shape = gpd.read_file(SHAPE_PATH)
    shape['FIPS'] = shape['FIPS'].astype(str).str.zfill(5)

    class_df = pd.read_csv(CLASS_PATH, dtype={'FIPS': str})
    class_df['FIPS'] = class_df['FIPS'].str.zfill(5)
    class_df['county_class'] = class_df['2023 Code'].astype(str)

    merged = shape.merge(class_df[['FIPS', 'county_class']], on='FIPS', how='left')
    return merged

def set_view_window(main_ax, alaska_ax, hawaii_ax):
    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)
    alaska_ax.set_axis_off()
    hawaii_ax.set_axis_off()
    main_ax.axis('off')

    main_ax.set_xlim([-125, -65])
    main_ax.set_ylim([25, 50])

def plot_individual_category_maps(merged):
    categories = sorted(merged['county_class'].dropna().unique())

    for cat in categories:
        fig, main_ax = plt.subplots(figsize=(10, 6))
        plt.title(f'Urbanicity Category {cat}', fontsize=14, weight='bold')

        alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
        hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

        merged['highlight'] = merged['county_class'].apply(lambda x: 1 if x == cat else np.nan)

        contig = merged[(merged['STATEFP'] != '02') & (merged['STATEFP'] != '15')]
        contig.plot(ax=main_ax, column='highlight', cmap='tab10', edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})

        ak = merged[merged['STATEFP'] == '02']
        ak.plot(ax=alaska_ax, column='highlight', cmap='tab10', edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})

        hi = merged[merged['STATEFP'] == '15']
        hi.plot(ax=hawaii_ax, column='highlight', cmap='tab10', edgecolor='black', linewidth=0.1, missing_kwds={"color": "lightgrey"})

        state_boundaries = merged.dissolve(by='STATEFP', as_index=False)
        state_boundaries[state_boundaries['STATEFP'].isin(contig['STATEFP'])].boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
        state_boundaries[state_boundaries['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
        state_boundaries[state_boundaries['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

        set_view_window(main_ax, alaska_ax, hawaii_ax)

        plt.savefig(f"{OUT_DIR}/Urbanicity_Category_{cat}_Map.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: Urbanicity_Category_{cat}_Map.png")

def plot_combined_category_map(merged):
    fig, main_ax = plt.subplots(figsize=(10, 6))
    plt.title("County Urbanicity Categories", fontsize=14, weight='bold')

    alaska_ax = fig.add_axes([0, -0.5, 1.4, 1.4])
    hawaii_ax = fig.add_axes([0.24, 0.1, 0.15, 0.15])

    contig = merged[(merged['STATEFP'] != '02') & (merged['STATEFP'] != '15')]
    contig.plot(ax=main_ax, column='county_class', cmap='RdYlBu', edgecolor='black', linewidth=0.1)

    ak = merged[merged['STATEFP'] == '02']
    ak.plot(ax=alaska_ax, column='county_class', cmap='RdYlBu', edgecolor='black', linewidth=0.1)

    hi = merged[merged['STATEFP'] == '15']
    hi.plot(ax=hawaii_ax, column='county_class', cmap='RdYlBu', edgecolor='black', linewidth=0.1)

    state_boundaries = merged.dissolve(by='STATEFP', as_index=False)
    state_boundaries[state_boundaries['STATEFP'].isin(contig['STATEFP'])].boundary.plot(ax=main_ax, edgecolor='black', linewidth=0.5)
    state_boundaries[state_boundaries['STATEFP'] == '02'].boundary.plot(ax=alaska_ax, edgecolor='black', linewidth=0.5)
    state_boundaries[state_boundaries['STATEFP'] == '15'].boundary.plot(ax=hawaii_ax, edgecolor='black', linewidth=0.5)

    # Legend
    unique_classes = sorted(merged['county_class'].dropna().unique())
    cmap = plt.get_cmap('RdYlBu', len(unique_classes))
    colors = [cmap(i) for i in range(len(unique_classes))]
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Category {unique_classes[i]}') for i in range(len(unique_classes))]

    main_ax.legend(
        handles=legend_patches,
        title='County Category',
        loc='lower left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.,
        frameon=True
    )

    set_view_window(main_ax, alaska_ax, hawaii_ax)

    plt.savefig(f"{OUT_DIR}/All_County_Categories_Map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: All_County_Categories_Map.png")

def main():
    merged = load_data()
    plot_individual_category_maps(merged)
    plot_combined_category_map(merged)

if __name__ == "__main__":
    main()
