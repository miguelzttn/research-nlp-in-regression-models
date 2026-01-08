import requests
from pathlib import Path
from utils import load_search_queries, write_results_to_csv

QUERIES_FILEPATH = Path("1 - scrapping") / "search_strings.json"

queries = load_search_queries(QUERIES_FILEPATH)

for query_category in queries:
    category_name = query_category.get("category")
    search_strings = query_category.get("search_strings", [])

    searach_string_combined = ''
    for search_string in search_strings:
        searach_string_combined += f'({search_string}) OR '

    searach_string_combined = searach_string_combined.rstrip(' OR ')

    print(100 * "=")
    print(f"Category: {category_name}")
    print(f"Search Strings Combined: {searach_string_combined}")
    print(100 * "=")
    print("\n")


# 1. Query
#   For each category, I created a master query by combining all search strings with OR operator.
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
#   After downloading, I moved the file to '.\data' and renamed it to 'springer_idsr_search_results_{category_name}.csv'
# 
# Ain't no need for automation here. Just these steps for added clarity and reproducibility.
# Note: If you want to automate this, consider using Selenium to log in and download the file.