#!/usr/bin/env python3
"""
Targeted IEEE Paper Downloader
Attempts to download a specific IEEE paper with enhanced strategies.
"""

import os
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests

# Web scraping
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Load environment variables
load_dotenv()

def setup_driver():
    """Setup Chrome driver with enhanced options for IEEE access."""
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def search_ieee_paper(title, authors, year):
    """Search for paper on IEEE Xplore with detailed logging."""
    print("\\n=== Searching IEEE Xplore ===")
    print(f"Title: {title}")
    print(f"Authors: {authors}")
    print(f"Year: {year}")

    driver = setup_driver()

    try:
        # Go to IEEE Xplore search
        search_query = f'"{title}"'
        encoded_query = quote_plus(search_query)
        search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={encoded_query}"

        print(f"Search URL: {search_url}")
        driver.get(search_url)
        time.sleep(3)

        # Check if we found results
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".List-results-items"))
            )
            print("✓ Found search results")
        except TimeoutException:
            print("✗ No search results found")
            return None

        # Get all result items
        results = driver.find_elements(By.CSS_SELECTOR, ".List-results-items")
        print(f"Found {len(results)} potential matches")

        for i, result in enumerate(results[:3]):  # Check first 3 results
            try:
                # Try multiple selectors to get paper title
                title_selectors = [
                    "h2 a",
                    ".result-item-title a",
                    ".document-title a",
                    ".title a",
                    "a[href*='/document/']"
                ]

                paper_title = ""
                paper_url = ""

                for selector in title_selectors:
                    try:
                        title_elements = result.find_elements(By.CSS_SELECTOR, selector)
                        if title_elements:
                            title_element = title_elements[0]
                            paper_title = title_element.text.strip()
                            paper_url = title_element.get_attribute('href')
                            if paper_title:
                                break
                    except:
                        continue

                if not paper_title:
                    # Try to get any text from the result
                    paper_title = result.text[:200] if result.text else "Unknown title"
                    # Try to find any IEEE document link
                    links = result.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute('href')
                        if href and '/document/' in href:
                            paper_url = href
                            break

                print(f"\\nResult {i+1}: {paper_title[:80]}...")
                print(f"URL: {paper_url}")

                # Check if this looks like a match (simple similarity check)
                title_words = [word.lower() for word in title.split() if len(word) > 3]
                matches = sum(1 for word in title_words if word in paper_title.lower())
                match_ratio = matches / len(title_words) if title_words else 0

                print(f"Match ratio: {match_ratio:.2f} ({matches}/{len(title_words)} key words)")

                if match_ratio > 0.3 or "eyeriss" in paper_title.lower():  # More flexible matching
                    print("✓ Title match found!")
                    return paper_url

                # If we have a valid IEEE document URL, try it anyway
                if paper_url and '/document/' in paper_url:
                    print("✓ Found valid IEEE document URL, will try this one")
                    return paper_url

            except Exception as e:
                print(f"Error processing result {i+1}: {e}")
                continue

        print("✗ No matching papers found in results")
        return None

    finally:
        driver.quit()

def download_ieee_paper(paper_url, ieee_username, ieee_password, output_dir):
    """Attempt to download paper from IEEE with login."""
    print("\\n=== Attempting Download ===")
    print(f"Paper URL: {paper_url}")

    driver = setup_driver()

    try:
        # Go to paper page
        driver.get(paper_url)
        time.sleep(3)

        # Try to login using the Personal Sign In button on the paper page
        print("Looking for Personal Sign In button...")

        try:
            # Look for Personal Sign In button on top right of page
            signin_button_selectors = [
                "//button[contains(text(), 'Personal Sign In')]",
                "//a[contains(text(), 'Personal Sign In')]",
                ".personal-signin-btn",
                "#personal-signin",
                "//button[contains(@class, 'personal')]",
                "//a[contains(@class, 'signin')]"
            ]

            signin_button = None
            for selector in signin_button_selectors:
                try:
                    if selector.startswith('//'):
                        elements = driver.find_elements(By.XPATH, selector)
                    else:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)

                    if elements:
                        signin_button = elements[0]
                        print(f"✓ Found Personal Sign In button with selector: {selector}")
                        break
                except:
                    continue

            if not signin_button:
                # Look more broadly for any sign in elements
                all_signin_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Sign In') or contains(text(), 'Sign in') or contains(text(), 'signin')]")
                print(f"Found {len(all_signin_elements)} elements with sign in text:")
                for elem in all_signin_elements[:5]:
                    try:
                        text = elem.text.strip()
                        tag = elem.tag_name
                        print(f"  - {tag}: '{text}'")
                        if 'Personal' in text or 'personal' in text:
                            signin_button = elem
                            print("✓ Found Personal Sign In element")
                            break
                    except:
                        pass

            if signin_button:
                print("Clicking Personal Sign In button...")
                driver.execute_script("arguments[0].click();", signin_button)
                time.sleep(2)

                # Now look for the hidden div/form that should have appeared
                print("Looking for revealed login form...")

                # Try to find the xpl-personal-signin element or login form
                try:
                    # Wait for the hidden form to appear
                    login_form = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "xpl-personal-signin"))
                    )
                    print("✓ Found xpl-personal-signin form after clicking button")
                except:
                    # Try alternative selectors for the revealed form
                    form_selectors = [
                        ".signin-form",
                        ".login-form",
                        ".personal-signin-form",
                        "#signin-form",
                        "form[name='signin']",
                        "div[style*='block']"  # Look for divs that became visible
                    ]

                    login_form = None
                    for selector in form_selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            if elements:
                                login_form = elements[0]
                                print(f"✓ Found login form with selector: {selector}")
                                break
                        except:
                            continue

                if login_form:
                    # Find username and password fields within the revealed form
                    try:
                        username_field = login_form.find_element(By.NAME, "usernameProv")
                        password_field = login_form.find_element(By.NAME, "passwordProv")
                    except:
                        # Try alternative field names
                        try:
                            username_field = login_form.find_element(By.CSS_SELECTOR, "input[type='email'], input[name*='user'], input[name*='email']")
                            password_field = login_form.find_element(By.CSS_SELECTOR, "input[type='password'], input[name*='pass']")
                        except:
                            # Look more broadly in the form
                            username_field = login_form.find_element(By.CSS_SELECTOR, "input[type='text'], input[type='email']")
                            password_field = login_form.find_element(By.CSS_SELECTOR, "input[type='password']")

                    print("✓ Found login form fields in revealed div")

                    # Clear and fill the fields
                    username_field.clear()
                    username_field.send_keys(ieee_username)

                    password_field.clear()
                    password_field.send_keys(ieee_password)

                    print("✓ Filled login credentials")

                    # Find and click the sign in button within the form
                    try:
                        submit_button = login_form.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button:contains('Sign'), input[value*='Sign']")
                    except:
                        # Look for any clickable element with sign in text
                        submit_button = login_form.find_element(By.XPATH, ".//*[contains(text(), 'Sign In') or contains(text(), 'Sign in') or contains(text(), 'Login')]")

                    print("Clicking Sign In button...")
                    driver.execute_script("arguments[0].click();", submit_button)
                    time.sleep(5)

                    print("✓ Login submitted via Personal Sign In form")

                    # Check if we're still on the same page or redirected
                    current_url = driver.current_url
                    print(f"Current URL after login: {current_url}")

                    # Check for login success by looking for "Sign Out" button/text
                    try:
                        # Look specifically for Sign Out text which indicates successful login
                        signout_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Sign Out') or contains(text(), 'Sign out') or contains(text(), 'Logout')]")
                        if signout_elements:
                            print("✓ Login successful - 'Sign Out' button found")
                            login_successful = True
                        else:
                            print("⚠ Login may have failed - no 'Sign Out' button found")
                            login_successful = False

                            # Also check if Personal Sign In text is still there (would indicate failed login)
                            signin_still_there = driver.find_elements(By.XPATH, "//*[contains(text(), 'Personal Sign In')]")
                            if signin_still_there:
                                print("✗ Login failed - 'Personal Sign In' still visible")

                    except Exception as e:
                        print(f"Could not verify login status: {e}")
                        login_successful = False

                else:
                    print("✗ Could not find login form after clicking Personal Sign In")
                    return False

            else:
                print("✗ Could not find Personal Sign In button")
                return False

        except Exception as e:
            print(f"✗ Personal Sign In process failed: {e}")
            return False

        # Look for PDF download options
        print("\\nLooking for PDF download options...")

        # If we have login_successful variable, use it for better logic
        if 'login_successful' not in locals():
            login_successful = False

        if login_successful:
            print("User is logged in - trying enhanced PDF download methods...")
        else:
            print("User may not be logged in - trying basic PDF download methods...")

        # First try to click PDF download button and wait for download
        pdf_buttons = [
            'a[title*="PDF"]',
            '.pdf-btn',
            '.document-btn-pdf',
            'a[aria-label*="PDF"]',
            '.stats-document-abstract-downloadPdf',
            'a[href*=".pdf"]',
            '.document-ft-pdf-link'  # Full text PDF link
        ]

        clicked_pdf = False
        for selector in pdf_buttons:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    element = elements[0]
                    print(f"✓ Found PDF element with selector: {selector}")

                    # Check if it's a direct PDF link
                    href = element.get_attribute('href')
                    if href and href.startswith('http') and '.pdf' in href:
                        print(f"Direct PDF link: {href}")
                        pdf_link = href
                        break

                    # Try clicking the button
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        print("✓ Clicked PDF download button")
                        time.sleep(3)
                        clicked_pdf = True
                        break
                    except Exception as click_error:
                        print(f"Could not click element: {click_error}")
                        continue
            except Exception as e:
                print(f"Error with selector {selector}: {e}")
                continue

        # If we clicked a PDF button, check for new tabs or redirects
        if clicked_pdf:
            # Check if a new tab opened
            if len(driver.window_handles) > 1:
                print("✓ New tab opened, switching to it")
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(2)
                current_url = driver.current_url
                if '.pdf' in current_url:
                    pdf_link = current_url
                    print(f"✓ Found PDF URL in new tab: {pdf_link}")
                else:
                    print(f"New tab URL: {current_url}")
            else:
                # Check if current page changed to PDF
                current_url = driver.current_url
                if '.pdf' in current_url or 'pdf' in current_url:
                    pdf_link = current_url
                    print(f"✓ Current page is PDF: {pdf_link}")

        # If still no direct PDF link found
        if 'pdf_link' not in locals() or not pdf_link or pdf_link == "javascript:void()":
            print("✗ No direct PDF download link found")
            # Try to find any download options for debugging
            download_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Download') or contains(text(), 'PDF') or contains(text(), 'pdf')]")
            print(f"Found {len(download_elements)} elements with 'Download' or 'PDF' text:")
            for elem in download_elements[:10]:
                try:
                    text = elem.text.strip()[:100]
                    tag = elem.tag_name
                    href = elem.get_attribute('href') or 'no href'
                    if text:
                        print(f"  - {tag}: '{text}' ({href[:50]})")
                except:
                    pass

            # Maybe the paper is open access - try to find the full text
            print("\\nChecking if paper content is available...")
            try:
                # Look for indicators that full text is available
                full_text_indicators = driver.find_elements(By.XPATH, "//*[contains(text(), 'Open Access') or contains(text(), 'Full Text') or contains(text(), 'View full text')]")
                if full_text_indicators:
                    print("✓ Found full text indicators - this might be an open access paper")
                    # But for now, we can't extract the PDF
                    return False
                else:
                    print("✗ No open access indicators found - paper likely requires subscription")
                    return False
            except:
                return False

        # Try to download the PDF
        print(f"Attempting to download PDF from: {pdf_link}")

        # Get cookies from selenium session
        cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}

        # Use requests to download
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        session.cookies.update(cookies)

        response = session.get(pdf_link, allow_redirects=True)

        if response.status_code == 200 and 'application/pdf' in response.headers.get('content-type', ''):
            # Save the PDF
            filename = f"IEEE_Paper_{int(time.time())}.pdf"
            filepath = Path(output_dir) / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"✓ Successfully downloaded: {filename}")
            print(f"File size: {len(response.content)} bytes")
            return True
        else:
            print(f"✗ Download failed. Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
            return False

    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False

    finally:
        driver.quit()

def main():
    """Try to download a specific IEEE paper."""

    # Choose a specific IEEE paper to test with - try a simpler/older one
    target_paper = {
        "title": "A 10 b, 20 msample/s, 35 mw pipeline a/d converter",
        "authors": "T. B. Cho and P. R. Gray",
        "year": "1995",
        "venue": "IEEE Journal of Solid-State Circuits"
    }

    print("=== IEEE Paper Download Attempt ===")
    print(f"Target: {target_paper['title']}")

    # Get credentials
    ieee_username = os.getenv('IEEE_USERNAME')
    ieee_password = os.getenv('IEEE_PASS')

    if not ieee_username or not ieee_password:
        print("✗ IEEE credentials not found in .env file")
        return

    print(f"Using IEEE account: {ieee_username}")

    # Search for the paper
    paper_url = search_ieee_paper(
        target_paper["title"],
        target_paper["authors"],
        target_paper["year"]
    )

    if not paper_url:
        print("\\n✗ Could not find paper on IEEE Xplore")
        return

    # Attempt download
    output_dir = "papers/references/downloaded/ieee"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    success = download_ieee_paper(paper_url, ieee_username, ieee_password, output_dir)

    if success:
        print("\\n🎉 IEEE paper download successful!")
    else:
        print("\\n❌ IEEE paper download failed")

if __name__ == "__main__":
    main()
