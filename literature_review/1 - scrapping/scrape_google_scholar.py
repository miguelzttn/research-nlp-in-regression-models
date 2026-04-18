import sys
import re
import time
import random
import urllib.parse
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from utils import load_search_queries, write_results_to_csv

QUERIES_FILEPATH = Path("1 - scrapping") / "search_strings.json"
QUERIES_RESULT_PATH = Path("1 - scrapping") / "result" / "google_scholar_search_results.csv"


def get_driver():
    """Initializes a Chrome driver with anti-detection options."""
    chrome_options = Options()
    # Run in headed mode so you can manually solve CAPTCHAs if they appear
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # Standard User Agent
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def scrape_google_scholar_selenium(queries, pages_to_scrape=2):
    driver = get_driver()
    results_data = []

    try:
        for query in tqdm(queries, desc="Executando consultas"):

            for page in range(pages_to_scrape):
                start_index = page * 10
                url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}&start={start_index}&hl=pt-BR"

                driver.get(url)

                # Human-like pause (3-6 seconds)
                time.sleep(random.uniform(3, 6))

                # Check for CAPTCHA title
                if (
                    "comprovar que você não é um robô" in driver.title
                    or "show you're not a robot" in driver.title
                ):
                    print(
                        "⚠️ CAPTCHA detected! Please solve it in the browser window within 30 seconds."
                    )
                    time.sleep(30)  # Waiting for you to solve it manually

                soup = BeautifulSoup(driver.page_source, "html.parser")
                articles = soup.select("div.gs_r.gs_or.gs_scl")

                if not articles:
                    print(f"No results found on page {page+1} for '{query}'.")
                    break

                for i, article in enumerate(articles):
                    rank = start_index + i + 1

                    # 1. Title & Main Link
                    title_tag = article.select_one("h3.gs_rt")
                    title_text = "N/A"
                    main_link = ""

                    if title_tag:
                        title_text = title_tag.get_text(strip=True)
                        # Clean prefixes like [PDF], [HTML], [CITAÇÃO]
                        title_text = re.sub(r"^\[.*?\]\s*", "", title_text)

                        # Extract the anchor tag inside the title for the main link
                        link_tag = title_tag.select_one("a")
                        if link_tag and "href" in link_tag.attrs:
                            main_link = link_tag["href"]

                    # 2. PDF/Side Link (The one on the right side)
                    pdf_div = article.select_one("div.gs_or_ggsm a")
                    pdf_link = pdf_div["href"] if pdf_div else ""

                    # 3. Citations
                    # Find the footer div and loop through all <a> tags
                    footer_div = article.select_one("div.gs_fl.gs_flb")
                    citations = 0
                    if footer_div:
                        for link in footer_div.select("a"):
                            href = link.get("href", "")
                            if href.startswith("/scholar?cites="):
                                cit_text = link.get_text()
                                cit_match = re.search(r"(\d+)", cit_text)
                                if cit_match:
                                    citations = int(cit_match.group(1))
                                break

                    # 4. Year
                    meta_div = article.select_one("div.gs_a")
                    year = ""
                    if meta_div:
                        meta_text = meta_div.get_text()
                        year_match = re.findall(r"\b(19\d{2}|20\d{2})\b", meta_text)
                        if year_match:
                            year = year_match[-1]

                    results_data.append(
                        {
                            "str_search_string": query,
                            "nr_page": page + 1,
                            "nr_order": rank,
                            "str_article": title_text,
                            "nr_year": year,
                            "nr_citations": citations,
                            "str_original_Link": main_link,
                            "str_pdf_link": pdf_link,
                        }
                    )

    finally:
        driver.quit()

    return pd.DataFrame(results_data)

if __name__ == "__main__":

    search_queries = load_search_queries(QUERIES_FILEPATH)
    df_full = []

    for query_category in search_queries:

        category_name = query_category.get("category")
        search_strings = query_category.get("search_strings", [])

        print(f"Retrieving by category: {category_name}")

        df = scrape_google_scholar_selenium(queries=search_strings, pages_to_scrape=2)
        df["category"] = category_name

        df_full.append(df)
        df = None

    df = pd.concat(df_full, ignore_index=True)
    write_results_to_csv(df=df, filepath=QUERIES_RESULT_PATH)
