import requests
from pathlib import Path
from utils import load_search_queries, write_results_to_csv

QUERIES_FILEPATH = Path("search_strings.json")

queries = load_search_queries(QUERIES_FILEPATH)

master_query = ''

for query_category in queries:
    category_name = query_category.get("category")
    search_strings = query_category.get("search_strings", [])
    
    for search_string in search_strings:
        master_query += f'({search_string}) OR '

master_query = master_query.rstrip(' OR ')
print("Master Query:")
print(master_query)

# 1. Query
#   This master query is used on https://link.springer.com/search' advanced search
#   to retrieve articles from the International Journal of Data Science and Analytics.
#   With:
#       - Keywords: master_query
#       - In journal(s): "International Journal of Data Science and Analytics"
#
# 2. Data
#   Then, I pressed "Download results (.csv)" to get the data.
#   Logged in Springer account is required so I used mine
#
# 3. Downloaded File
#   After downloading, I moved the file to '.\data' and renamed it to springer_idsr_search_results.csv'
# 
# Ain't no need for automation here. Just these steps for added clarity and reproducibility.
# Note: If you want to automate this, consider using Selenium to log in and download the file.