import os
import json
import pandas as pd
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import math

# ================== Configuration ==================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load environment variables from .env file
load_dotenv()

# Primo API Configuration
PRIMO_API_KEY = os.getenv('PRIMO_API_KEY')
if not PRIMO_API_KEY:
    logger.error("PRIMO_API_KEY environment variable not set.")
    exit(1)

# Hugging Face Repository Configuration
REPO_ID = "MongoDB/embedded_movies"  # Replace with your actual repo ID if different
FILENAME = "sample_mflix.embedded_movies.json"  # Replace with the actual filename if different

# Output Configuration
OUTPUT_JSON = 'movies_with_dvd_or_blu_ray.json'

# Threading Configuration
MAX_THREADS = 10  # Adjust based on your system and API rate limits

# =====================================================

def download_dataset(repo_id: str, filename: str) -> str:
    """
    Downloads a specific file from a Hugging Face repository.

    :param repo_id: The repository ID on Hugging Face.
    :param filename: The name of the file to download.
    :return: The local path to the downloaded file.
    """
    try:
        logger.info(f"Downloading '{filename}' from repository '{repo_id}'...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        logger.info(f"Downloaded '{filename}' to '{file_path}'.")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download '{filename}' from '{repo_id}': {e}")
        exit(1)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a JSON file into a pandas DataFrame.

    :param file_path: The local path to the JSON file.
    :return: A pandas DataFrame containing the dataset.
    """
    try:
        logger.info(f"Loading dataset from '{file_path}'...")
        df = pd.read_json(file_path)  # Removed lines=True based on JSON structure
        logger.info(f"Dataset loaded with {len(df)} records.")
        return df
    except ValueError as ve:
        logger.error(f"ValueError while loading JSON: {ve}")
        exit(1)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred while loading the dataset: {e}")
        exit(1)

def fetch_movie_availability(query: str, api_key: str, offset: int = 0, limit: int = 10) -> dict:
    """
    Fetches search results from the Primo API for a given movie title.

    :param query: The movie title to search for.
    :param api_key: Your Primo API key.
    :param offset: Pagination offset.
    :param limit: Number of results to fetch per request.
    :return: JSON response containing search results or None if an error occurs.
    """
    base_url = "https://utdallas.primo.exlibrisgroup.com/primaws/rest/pub/pnxs"

    params = {
        'acTriggered': 'false',
        'blendFacetsSeparately': 'false',
        'citationTrailFilterByAvailability': 'true',
        'disableCache': 'false',
        'getMore': '0',
        'inst': '01UT_DALLAS',
        'isCDSearch': 'false',
        'lang': 'en',
        'limit': str(limit),
        'newspapersActive': 'true',
        'newspapersSearch': 'false',
        'offset': str(offset),
        'otbRanking': 'false',
        'pcAvailability': 'false',
        'q': f'any,contains,{query}',
        'qExclude': '',
        'qInclude': '',
        'rapido': 'false',
        'refEntryActive': 'false',
        'rtaLinks': 'true',
        'scope': 'MyInst_and_CI',
        'searchInFulltextUserSelection': 'false',
        'skipDelivery': 'Y',
        'sort': 'rank',
        'tab': 'Everything',
        'vid': '01UT_DALLAS:UTDALMA'
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error for '{query}': {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error for '{query}': {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error for '{query}': {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception for '{query}': {req_err}")
    except ValueError:
        logger.error(f"Invalid JSON response for '{query}'.")
    return None

def check_dvd_availability(doc: dict) -> bool:
    """
    Checks if the 'lds10' attribute indicates DVD/Blu-ray availability.

    :param doc: A single document entry from the JSON response.
    :return: Boolean indicating availability.
    """
    lds10 = doc.get('pnx', {}).get('display', {}).get('lds10', [])
    if not lds10:
        return False

    for entry in lds10:
        if "DVD/Blu-ray player can be checked-out!" in entry:
            return True
    return False

def process_movie(record: dict, api_key: str) -> dict:
    """
    Processes a single movie record to determine DVD/Blu-ray availability.

    :param record: The dataset record (a dictionary) containing movie information.
    :param api_key: Your Primo API key.
    :return: Augmented record with library link if available, or None.
    """
    query = record.get('title')
    if not query:
        return None

    data = fetch_movie_availability(query, api_key)
    if not data or 'docs' not in data or not data['docs']:
        # No data found or error occurred, skip this movie
        return None

    # Assuming the first document is the most relevant
    first_doc = data['docs'][0]

    # Check availability
    if not check_dvd_availability(first_doc):
        # Movie not available for DVD/Blu-ray
        return None

    # Construct the link
    record_id = first_doc.get('pnx', {}).get('control', {}).get('recordid', ['No Record ID'])[0]
    link = f"https://utdallas.primo.exlibrisgroup.com/discovery/fulldisplay?docid={record_id}&context=L&vid=01UT_DALLAS:UTDALMA&lang=en"

    # Prepare the augmented record with the library link included
    augmented_record = {'library_link': link}
    for key, value in record.items():
        augmented_record[key] = value

    return augmented_record

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

def main():
    # Step 1: Download the dataset
    dataset_path = download_dataset(REPO_ID, FILENAME)

    # Step 2: Load the dataset into pandas DataFrame
    df = load_dataset(dataset_path)

    # Step 3: Convert DataFrame to list of records
    records = df.to_dict(orient='records')
    logger.info(f"Total movie records to process: {len(records)}")

    results = []

    # Step 4: Use ThreadPoolExecutor for concurrent API requests with progress bar
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_movie, record, PRIMO_API_KEY): record for record in records}

        # Initialize tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing movies"):
            record = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error(f"Exception processing record '{record.get('title', 'Unknown')}': {exc}")

    # Step 5: Convert results to list of dictionaries
    if results:
        # Step 6: Replace NaN values with empty strings (recursively)
        records_list = [replace_nan(record) for record in results]

        logger.info(f"Total movies available: {len(records_list)}")

        # Step 7: Display the first few results
        logger.info("\n--- Movie Availability Report ---\n")
        print(json.dumps(records_list[:5], ensure_ascii=False, indent=4))

        # Step 8: Save the filtered results to a formatted JSON file without escaped slashes
        try:
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(records_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Results saved to '{OUTPUT_JSON}'.")
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")
    else:
        logger.info("No movies available for DVD/Blu-ray.")

if __name__ == "__main__":
    main()

