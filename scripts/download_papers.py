#!/usr/bin/env python3
"""
Academic Paper Downloader
Downloads papers from reference lists using multiple sources and strategies.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

# Academic sources
import arxiv
import requests

# Web scraping
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    title: str
    authors: str
    venue: str
    year: str
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    source: str | None = None

class PaperDownloader:
    def __init__(self, output_dir: str = "papers/references/downloaded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # IEEE credentials
        self.ieee_username = os.getenv('IEEE_USERNAME')
        self.ieee_password = os.getenv('IEEE_PASS')

        # Setup requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

        # Stats
        self.downloaded_count = 0
        self.failed_count = 0
        self.download_log = []

    def parse_reference_file(self, file_path: str) -> list[Paper]:
        """Parse a reference file and extract paper information."""
        papers = []

        with open(file_path) as f:
            content = f.read()

        # Split by reference numbers [1], [2], etc.
        refs = re.split(r'\n\[(\d+)\]', content)[1:]  # Skip first empty element

        for i in range(0, len(refs), 2):
            if i + 1 < len(refs):
                ref_num = refs[i]
                ref_text = refs[i + 1].strip()

                paper = self._parse_single_reference(ref_text)
                if paper:
                    papers.append(paper)
                    logger.info(f"Parsed reference {ref_num}: {paper.title[:50]}...")

        return papers

    def _parse_single_reference(self, ref_text: str) -> Paper | None:
        """Parse a single reference string into a Paper object."""
        try:
            # Extract arXiv ID if present
            arxiv_match = re.search(r'arXiv:(\d+\.\d+)', ref_text)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None

            # Extract DOI if present
            doi_match = re.search(r'doi:([^\s,]+)', ref_text)
            doi = doi_match.group(1) if doi_match else None

            # Extract URL if present
            url_match = re.search(r'Available: (https?://[^\s\]]+)', ref_text)
            url = url_match.group(1) if url_match else None

            # Extract year
            year_match = re.search(r'(\d{4})', ref_text)
            year = year_match.group(1) if year_match else "Unknown"

            # Extract title (usually in quotes or before comma)
            title_match = re.search(r'"([^"]+)"', ref_text)
            if not title_match:
                # Try alternative pattern: first part before "in" or venue
                title_match = re.search(r'^([^,]+),', ref_text)

            title = title_match.group(1).strip() if title_match else "Unknown Title"

            # Extract authors (usually at the beginning)
            authors_match = re.search(r'^([^"]+),?\s*"', ref_text)
            if not authors_match:
                authors_match = re.search(r'^([^,]+)', ref_text)

            authors = authors_match.group(1).strip() if authors_match else "Unknown Authors"

            # Extract venue
            venue = "Unknown Venue"
            if "IEEE" in ref_text:
                ieee_match = re.search(r'IEEE[^,]*[^,]*', ref_text)
                venue = ieee_match.group(0) if ieee_match else "IEEE Publication"
            elif "ACM" in ref_text:
                venue = "ACM Publication"
            elif "arXiv" in ref_text:
                venue = "arXiv"
            elif "Proceedings" in ref_text:
                proc_match = re.search(r'in ([^,]+Proceedings[^,]*)', ref_text)
                venue = proc_match.group(1) if proc_match else "Conference Proceedings"

            return Paper(
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                url=url
            )

        except Exception as e:
            logger.error(f"Error parsing reference: {e}")
            return None

    def download_arxiv_paper(self, paper: Paper) -> bool:
        """Download paper from arXiv."""
        if not paper.arxiv_id:
            return False

        try:
            logger.info(f"Downloading from arXiv: {paper.arxiv_id}")

            # Search for paper
            search = arxiv.Search(id_list=[paper.arxiv_id])
            paper_obj = next(search.results())

            # Generate filename
            first_author = paper.authors.split(',')[0].split()[-1]  # Last name
            filename = f"{first_author}_{paper.year}_{paper.arxiv_id.replace('.', '_')}.pdf"
            # filepath = self.output_dir / filename  # Will be used by download method

            # Download
            paper_obj.download_pdf(dirpath=str(self.output_dir), filename=filename)

            logger.info(f"✓ Downloaded: {filename}")
            self.downloaded_count += 1
            self.download_log.append(f"✓ {paper.title} - arXiv:{paper.arxiv_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to download from arXiv: {e}")
            return False

    def download_ieee_paper(self, paper: Paper) -> bool:
        """Download paper from IEEE Xplore using credentials."""
        if "IEEE" not in paper.venue:
            return False

        if not self.ieee_username or not self.ieee_password:
            logger.warning("IEEE credentials not found in .env file")
            return False

        try:
            # Setup Selenium for IEEE login
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            driver = webdriver.Chrome(options=options)

            try:
                # Search for paper on IEEE
                search_query = paper.title.replace('"', '')
                search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={search_query}"

                driver.get(search_url)
                time.sleep(3)

                # Look for the first paper result
                results = driver.find_elements(By.CSS_SELECTOR, ".List-results-items")
                if not results:
                    return False

                # Click on first result
                first_result = results[0].find_element(By.TAG_NAME, "a")
                paper_url = first_result.get_attribute('href')

                driver.get(paper_url)
                time.sleep(2)

                # Login if needed
                if "login" in driver.current_url.lower():
                    self._ieee_login(driver)

                # Look for PDF download link
                pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(text(), 'PDF')]")

                if pdf_links:
                    pdf_url = pdf_links[0].get_attribute('href')

                    # Download PDF
                    first_author = paper.authors.split(',')[0].split()[-1]
                    filename = f"{first_author}_{paper.year}_IEEE.pdf"
                    filepath = self.output_dir / filename

                    response = self.session.get(pdf_url)
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)

                        logger.info(f"✓ Downloaded from IEEE: {filename}")
                        self.downloaded_count += 1
                        self.download_log.append(f"✓ {paper.title} - IEEE Xplore")
                        return True

                return False

            finally:
                driver.quit()

        except Exception as e:
            logger.error(f"Failed to download from IEEE: {e}")
            return False

    def _ieee_login(self, driver):
        """Login to IEEE Xplore."""
        try:
            # Navigate to login page
            driver.get("https://ieeexplore.ieee.org/Xplorelogin/login.jsp")
            time.sleep(2)

            # Fill login form
            username_field = driver.find_element(By.ID, "usernameProv")
            password_field = driver.find_element(By.ID, "passwordProv")

            username_field.send_keys(self.ieee_username)
            password_field.send_keys(self.ieee_password)

            # Submit form
            login_button = driver.find_element(By.ID, "signin-btn")
            login_button.click()

            time.sleep(3)

        except Exception as e:
            logger.error(f"IEEE login failed: {e}")

    def download_from_doi(self, paper: Paper) -> bool:
        """Download paper using DOI."""
        if not paper.doi:
            return False

        try:
            # Try DOI resolver
            doi_url = f"https://doi.org/{paper.doi}"
            response = self.session.get(doi_url, allow_redirects=True)

            # Look for PDF links in the page
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_links = soup.find_all('a', href=re.compile(r'\.pdf'))

            if pdf_links:
                pdf_url = urljoin(response.url, pdf_links[0]['href'])

                pdf_response = self.session.get(pdf_url)
                if pdf_response.status_code == 200 and 'application/pdf' in pdf_response.headers.get('content-type', ''):
                    first_author = paper.authors.split(',')[0].split()[-1]
                    filename = f"{first_author}_{paper.year}_DOI.pdf"
                    filepath = self.output_dir / filename

                    with open(filepath, 'wb') as f:
                        f.write(pdf_response.content)

                    logger.info(f"✓ Downloaded via DOI: {filename}")
                    self.downloaded_count += 1
                    self.download_log.append(f"✓ {paper.title} - DOI")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to download via DOI: {e}")
            return False

    def download_from_url(self, paper: Paper) -> bool:
        """Download paper from provided URL."""
        if not paper.url:
            return False

        try:
            response = self.session.get(paper.url)

            if 'application/pdf' in response.headers.get('content-type', ''):
                first_author = paper.authors.split(',')[0].split()[-1]
                filename = f"{first_author}_{paper.year}_URL.pdf"
                filepath = self.output_dir / filename

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✓ Downloaded from URL: {filename}")
                self.downloaded_count += 1
                self.download_log.append(f"✓ {paper.title} - Direct URL")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to download from URL: {e}")
            return False

    def search_and_download(self, paper: Paper) -> bool:
        """Try multiple download strategies for a paper."""
        logger.info(f"Attempting to download: {paper.title[:50]}...")

        # Strategy 1: arXiv (highest success rate)
        if paper.arxiv_id and self.download_arxiv_paper(paper):
            return True

        # Strategy 2: Direct URL
        if paper.url and self.download_from_url(paper):
            return True

        # Strategy 3: DOI
        if paper.doi and self.download_from_doi(paper):
            return True

        # Strategy 4: IEEE Xplore
        if self.download_ieee_paper(paper):
            return True

        # If all strategies failed
        logger.warning(f"✗ Failed to download: {paper.title}")
        self.failed_count += 1
        self.download_log.append(f"✗ {paper.title} - All strategies failed")
        return False

    def generate_report(self) -> str:
        """Generate a download report."""
        report = f"""
=== PAPER DOWNLOAD REPORT ===

Total Attempts: {self.downloaded_count + self.failed_count}
Successfully Downloaded: {self.downloaded_count}
Failed Downloads: {self.failed_count}
Success Rate: {(self.downloaded_count / (self.downloaded_count + self.failed_count) * 100):.1f}%

=== DOWNLOAD LOG ===
"""
        for entry in self.download_log:
            report += f"{entry}\n"

        return report

def main():
    """Main function to download all papers from reference files."""
    downloader = PaperDownloader()

    # Reference files
    ref_files = [
        "papers/references/iscas2018_references.txt",
        "papers/references/time_domain_mac_thesis_references.txt"
    ]

    all_papers = []

    # Parse all reference files
    for ref_file in ref_files:
        if Path(ref_file).exists():
            logger.info(f"Parsing {ref_file}...")
            papers = downloader.parse_reference_file(ref_file)
            all_papers.extend(papers)
            logger.info(f"Found {len(papers)} papers in {ref_file}")
        else:
            logger.warning(f"Reference file not found: {ref_file}")

    logger.info(f"Total papers to download: {len(all_papers)}")

    # Download all papers
    for i, paper in enumerate(all_papers, 1):
        logger.info(f"[{i}/{len(all_papers)}] Processing paper...")
        downloader.search_and_download(paper)
        time.sleep(1)  # Be respectful to servers

    # Generate and save report
    report = downloader.generate_report()

    # Save report to file
    report_path = downloader.output_dir / "download_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    # Print summary
    print(report)
    logger.info(f"Download complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()
