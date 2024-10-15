import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import json
import logging
import math

# ================== Configuration ==================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Path to your Excel file
EXCEL_FILE = 'movies_with_dvd_or_blu_ray.xlsx'

# Define the dimensionality reduction technique to visualize
dim_reduction = 'UMAP'  # Options: 'PCA', 'tSNE', 'UMAP'

# Output files
OUTPUT_INTERACTIVE_HTML = 'movies_interactive_plot.html'
OUTPUT_STATIC_PLOT = 'movies_static_plot.png'

# =====================================================

def parse_plot_embedding(embedding_str):
    """
    Parses the JSON-formatted plot_embedding string into a Python list.
    
    :param embedding_str: JSON string of plot_embedding
    :return: List of numerical values representing the embedding
    """
    try:
        return json.loads(embedding_str)
    except (json.JSONDecodeError, TypeError):
        return []

def replace_nan(obj):
    """
    Recursively replace NaN values in a nested dictionary or list with empty strings.
    
    :param obj: The object to process (dict, list, or other).
    :return: The processed object with NaNs replaced.
    """
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return ""
        else:
            return obj
    elif pd.isna(obj):
        return ""
    else:
        return obj

def flatten_nested_fields(record: dict) -> dict:
    """
    Converts nested lists and dictionaries into JSON-formatted strings for Excel compatibility.
    
    :param record: The movie record (a dictionary).
    :return: The flattened movie record with nested structures as JSON strings.
    """
    flattened_record = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flattened_record[key] = json.dumps(value, ensure_ascii=False)
        else:
            # Handle NaN values by replacing them with empty strings
            if pd.isna(value):
                flattened_record[key] = ""
            else:
                flattened_record[key] = value
    return flattened_record

def main():
    # Step 1: Load the dataset into pandas DataFrame
    try:
        logger.info(f"Loading dataset from '{EXCEL_FILE}'...")
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
        logger.info(f"Dataset loaded with {len(df)} records.")
    except FileNotFoundError:
        logger.error(f"File not found: {EXCEL_FILE}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading the dataset: {e}")
        exit(1)
    
    # Step 2: Parse plot_embedding if it's a JSON string
    if df['plot_embedding'].dtype == object:
        logger.info("Parsing 'plot_embedding' from JSON strings to lists...")
        df['plot_embedding'] = df['plot_embedding'].apply(parse_plot_embedding)
    
    # Step 3: Extract embeddings
    embeddings = df['plot_embedding'].tolist()
    embeddings_array = np.array(embeddings)
    logger.info(f"Embeddings array shape: {embeddings_array.shape}")
    
    # Step 4: Handle missing embeddings
    missing_embeddings = df['plot_embedding'].isna().sum()
    if missing_embeddings > 0:
        logger.warning(f"There are {missing_embeddings} records with missing 'plot_embedding'. They will be excluded from visualization.")
        df = df.dropna(subset=['plot_embedding']).reset_index(drop=True)
        embeddings = df['plot_embedding'].tolist()
        embeddings_array = np.array(embeddings)
        logger.info(f"Embeddings array shape after excluding missing: {embeddings_array.shape}")
    
    # Step 5: Perform Dimensionality Reduction
    if dim_reduction == 'PCA':
        logger.info("Performing PCA for dimensionality reduction...")
        pca = PCA(n_components=50, random_state=42)
        pca_result = pca.fit_transform(embeddings_array)
        logger.info("PCA completed.")
        
        # Further reduce to 2D for visualization
        pca_final = PCA(n_components=2, random_state=42)
        pca_final_result = pca_final.fit_transform(pca_result)
        df['PCA_1'] = pca_final_result[:, 0]
        df['PCA_2'] = pca_final_result[:, 1]
        logger.info("PCA dimensionality reduction to 2D completed.")
        
        x_col = 'PCA_1'
        y_col = 'PCA_2'
        title = 'Movies Visualization using PCA'
    elif dim_reduction == 'tSNE':
        logger.info("Performing t-SNE for dimensionality reduction...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
        tsne_result = tsne.fit_transform(embeddings_array)
        df['tSNE_1'] = tsne_result[:, 0]
        df['tSNE_2'] = tsne_result[:, 1]
        logger.info("t-SNE dimensionality reduction to 2D completed.")
        
        x_col = 'tSNE_1'
        y_col = 'tSNE_2'
        title = 'Movies Visualization using t-SNE'
    elif dim_reduction == 'UMAP':
        logger.info("Performing UMAP for dimensionality reduction...")
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        umap_result = umap_reducer.fit_transform(embeddings_array)
        df['UMAP_1'] = umap_result[:, 0]
        df['UMAP_2'] = umap_result[:, 1]
        logger.info("UMAP dimensionality reduction to 2D completed.")
        
        x_col = 'UMAP_1'
        y_col = 'UMAP_2'
        title = 'Movies Visualization using UMAP'
    else:
        logger.error("Invalid dimensionality reduction technique selected. Choose from 'PCA', 'tSNE', 'UMAP'.")
        exit(1)
    
    # Step 6: Visualize with Matplotlib and Seaborn
    plt.figure(figsize=(10, 8))
    if 'cluster' in df.columns:
        sns.scatterplot(
            x=x_col, y=y_col,
            hue='cluster',
            palette='tab10',
            data=df,
            legend='full',
            alpha=0.6
        )
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(
            x=x_col, y=y_col,
            data=df,
            alpha=0.6
        )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(OUTPUT_STATIC_PLOT)
    logger.info(f"Static plot saved to '{OUTPUT_STATIC_PLOT}'.")
    plt.show()
    
    # Step 7: Interactive Visualization with Plotly
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color='cluster' if 'cluster' in df.columns else None,
        hover_data=['title', 'genres', 'directors', 'cast'],
        title=title,
        opacity=0.7,
        width=1200,
        height=800
    )
    
    fig.update_layout(legend_title_text='Cluster' if 'cluster' in df.columns else '')
    fig.show()
    
    # Save interactive plot as HTML
    fig.write_html(OUTPUT_INTERACTIVE_HTML)
    logger.info(f"Interactive plot saved to '{OUTPUT_INTERACTIVE_HTML}'.")
    
    # Optional: Saving a subset with representative movies highlighted
    # Assuming you have a separate DataFrame 'representative_movies.xlsx'
    REPRESENTATIVE_EXCEL = 'representative_movies.xlsx'
    
    try:
        logger.info(f"Loading representative movies from '{REPRESENTATIVE_EXCEL}'...")
        rep_df = pd.read_excel(REPRESENTATIVE_EXCEL, engine='openpyxl')
        logger.info(f"Representative movies loaded with {len(rep_df)} records.")
    except FileNotFoundError:
        logger.error(f"File not found: {REPRESENTATIVE_EXCEL}")
        rep_df = pd.DataFrame()  # Empty DataFrame
    except Exception as e:
        logger.error(f"An error occurred while loading representative movies: {e}")
        rep_df = pd.DataFrame()
    
    if not rep_df.empty:
        # Parse plot_embedding if necessary
        if rep_df['plot_embedding'].dtype == object:
            rep_df['plot_embedding'] = rep_df['plot_embedding'].apply(parse_plot_embedding)
        
        # Extract embeddings
        rep_embeddings = rep_df['plot_embedding'].tolist()
        rep_embeddings_array = np.array(rep_embeddings)
        
        # Apply the same dimensionality reduction
        if dim_reduction == 'PCA':
            rep_result = pca_final.transform(rep_embeddings_array)
            rep_x = rep_result[:, 0]
            rep_y = rep_result[:, 1]
        elif dim_reduction == 'tSNE':
            rep_result = tsne_result[:len(rep_df)]  # Assuming t-SNE was already fitted
            rep_x = rep_result[:, 0]
            rep_y = rep_result[:, 1]
        elif dim_reduction == 'UMAP':
            rep_result = umap_reducer.transform(rep_embeddings_array)
            rep_x = rep_result[:, 0]
            rep_y = rep_result[:, 1]
        
        # Add representative movies to the interactive plot
        fig.add_trace(
            px.scatter(
                rep_df,
                x=rep_x,
                y=rep_y,
                text=rep_df['title'],
                marker=dict(color='red', size=12, symbol='x'),
                hover_data=['title', 'genres', 'directors', 'cast'],
                name='Representative Movies'
            ).data[0]
        )
        
        # Update layout
        fig.update_layout(legend_title_text='Cluster')
        fig.show()
        
        # Save updated interactive plot
        fig.write_html('movies_interactive_with_representatives.html')
        logger.info("Interactive plot with representative movies saved to 'movies_interactive_with_representatives.html'.")
    else:
        logger.warning("No representative movies data found. Skipping highlight on the plot.")

if __name__ == "__main__":
    main()

