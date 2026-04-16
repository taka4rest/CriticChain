"""
Browser-based search tool using undetected-chromedriver.
Provides stable search results by bypassing bot detection.
Also extracts actual page content for deeper analysis.
"""
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException
from bs4 import BeautifulSoup
import logging
import time
import random
import re
import requests

logger = logging.getLogger(__name__)

def extract_page_content(driver, url: str, max_chars: int = 800) -> str:
    """
    Extract main text content from a webpage.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL to extract content from
        max_chars: Maximum characters to extract (default: 800)
    
    Returns:
        Extracted text content, or error message
    """
    try:
        logger.info(f"Extracting content from: {url[:60]}...")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(2)
        
        # Try to find main content areas
        content_text = ""
        
        # Priority order: article > main > body
        for selector in ['article', 'main', '[role="main"]', 'body']:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    # Get text from all matching elements
                    for elem in elements[:3]:  # Limit to first 3 elements
                        text = elem.text.strip()
                        if text and len(text) > 100:  # Must have substantial content
                            content_text += text + "\n"
                    
                    if content_text:
                        break
            except Exception as e:
                logger.debug(f"Failed to extract with selector '{selector}': {e}")
                continue
        
        if not content_text:
            logger.warning(f"No content extracted from {url}")
            return "No content could be extracted from this page."
        
        # Clean up text
        content_text = re.sub(r'\s+', ' ', content_text)  # Normalize whitespace
        content_text = content_text.strip()
        
        # Truncate to max_chars
        if len(content_text) > max_chars:
            content_text = content_text[:max_chars] + "..."
        
        logger.info(f"Extracted {len(content_text)} characters")
        return content_text
        
    except TimeoutException:
        logger.warning(f"Timeout loading {url}")
        return "Page load timeout."
    except Exception as e:
        logger.error(f"Error extracting from {url}: {e}")
        return f"Extraction error: {str(e)[:100]}"

def _search_google(driver, query: str, max_results: int) -> list:
    """Internal function for Google search."""
    logger.info("Navigating to Google homepage...")
    driver.get("https://www.google.com")
    
    logger.info("Waiting for page to stabilize...")
    time.sleep(3)
    
    logger.info(f"Using JavaScript to search for: {query}")
    time.sleep(1)
    
    try:
        search_box = None
        for selector in ['textarea[name="q"]', 'input[name="q"]']:
            try:
                search_box = driver.find_element(By.CSS_SELECTOR, selector)
                if search_box:
                    break
            except:
                continue
        
        if not search_box:
            raise Exception("Could not find search box")
        
        search_box.click()
        time.sleep(random.uniform(0.3, 0.6))
        
        logger.info(f"Typing query with human-like timing: {query}")
        for char in query:
            search_box.send_keys(char)
            delay = random.uniform(0.05, 0.2)
            time.sleep(delay)
        
        time.sleep(random.uniform(0.5, 1.0))
        search_box.send_keys(Keys.RETURN)
        logger.info("Query submitted successfully")
        
    except Exception as e:
        logger.error(f"Error submitting query: {e}")
        # Fallback to JavaScript
        js_script = f"""
        var searchBox = document.querySelector('textarea[name="q"]') || document.querySelector('input[name="q"]');
        if (searchBox) {{
            searchBox.value = `{query}`;
            searchBox.form.submit();
            return true;
        }}
        return false;
        """
        driver.execute_script(js_script)
    
    logger.info("Waiting for search results to load...")
    time.sleep(5)
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    results = []
    result_divs = soup.select('div.g')
    if not result_divs:
        result_divs = soup.select('[data-sokoban-container]')
    if not result_divs:
        result_divs = soup.select('div[jscontroller]')
    
    logger.info(f"Found {len(result_divs)} result containers")
    
    if not result_divs:
        logger.warning("No Google result containers found! Saving HTML for debugging...")
        with open("debug_google.html", "w", encoding="utf-8") as f:
            f.write(html)
            
    for i, div in enumerate(result_divs[:max_results * 3]):
        # Skip hidden elements
        if 'style="display:none"' in str(div) or 'style="display: none"' in str(div):
            continue
            
        try:
            title = None
            # Try standard heading selectors
            for title_selector in ['h3', 'h2', '[role="heading"]', '.LC20lb']:
                title_elem = div.select_one(title_selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    if title and len(title) > 2: # Lowered threshold
                        break
            
            if not title:
                # Fallback: get title from first link
                link_elem = div.select_one('a')
                if link_elem:
                    title = link_elem.get_text().strip()
            
            if not title:
                continue
                
            # Filter out accessibility links and internal Google pages
            if "メイン コンテンツにスキップ" in title or "Skip to main content" in title:
                continue
            if "Google" in title and "ヘルプ" in title:
                continue
            
            url = None
            link_elems = div.select('a[href]')
            for link_elem in link_elems:
                potential_url = link_elem.get('href', '')
                if potential_url and potential_url.startswith('http'):
                    if 'google.com/search?' in potential_url:
                        continue
                    if 'translate.google' in potential_url or 'webcache.googleusercontent' in potential_url:
                        continue
                    if 'support.google.com' in potential_url: # Skip support pages
                        continue
                    if 'google.com/webhp' in potential_url: # Skip homepage links
                        continue
                    # Check if it's a valid result URL (not a google tracking link if possible)
                    url = potential_url
                    break
            
            if not url:
                logger.debug(f"Skipping Google result {i}: No valid URL found")
                continue
            
            snippet = "No description"
            for snippet_selector in ['div[data-sncf="1"]', 'div.VwiC3b', 'span.aCOpRe', 'div[style*="line-height"]', '.IsZvec']:
                snippet_elem = div.select_one(snippet_selector)
                if snippet_elem:
                    snippet = snippet_elem.get_text().strip()
                    if snippet:
                        break
            
            results.append({
                'title': title,
                'snippet': snippet,
                'url': url
            })
            
            if len(results) >= max_results:
                break
                
        except Exception as e:
            continue
            
    return results

def _search_ddg(driver, query: str, max_results: int) -> list:
    """Internal function for DuckDuckGo Lite search."""
    logger.info(f"Navigating directly to search results for: {query}")
    encoded_query = requests.utils.quote(query)
    search_url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}&df=y"
    driver.get(search_url)
    
    logger.info("Waiting for search results...")
    time.sleep(3)
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    results = []
    links = soup.select('a.result-link')
    if not links:
        main_table = soup.select_one('table')
        if main_table:
            links = main_table.select('a')
    
    if not links:
        logger.warning("No links found! Saving HTML for debugging...")
        with open("debug_ddg_lite.html", "w", encoding="utf-8") as f:
            f.write(html)
    
    for i, link in enumerate(links):
        if len(results) >= max_results:
            break
            
        url = link.get('href')
        title = link.get_text().strip()
        
        if not url or not title:
            continue
            
        if 'duckduckgo.com' in url and 'uddg=' not in url:
            continue
            
        if 'uddg=' in url:
            try:
                from urllib.parse import unquote
                match = re.search(r'uddg=([^&]+)', url)
                if match:
                    url = unquote(match.group(1))
            except Exception:
                pass
        
        snippet = "No description"
        parent = link.parent
        if parent:
            full_text = parent.get_text().strip()
            if title in full_text:
                snippet = full_text.replace(title, "", 1).strip()
            
            if not snippet or len(snippet) < 10:
                try:
                    parent_tr = parent.parent
                    if parent_tr and parent_tr.name == 'tr':
                        next_tr = parent_tr.find_next_sibling('tr')
                        if next_tr:
                            snippet = next_tr.get_text().strip()
                except:
                    pass
        
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
            
        results.append({
            'title': title,
            'snippet': snippet,
            'url': url
        })
        
    return results

def browser_search(query: str, max_results: int = 5, engine: str = "ddg") -> str:
    """
    Performs a web search using undetected-chromedriver.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        engine: Search engine to use ("ddg" or "google") (default: "ddg")
    
    Returns:
        Formatted string with search results
    """
    driver = None
    try:
        # Launch undetected Chrome
        options = uc.ChromeOptions()
        options.headless = False
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        logger.info("Launching Chrome...")
        driver = uc.Chrome(options=options)
        logger.info("Chrome launched successfully")
        
        if engine == "google":
            results = _search_google(driver, query, max_results)
        else:
            results = _search_ddg(driver, query, max_results)
            
        if not results:
            return f"Search completed but no results extracted. Query: '{query}'."
        
        # Extract page content from top 2-3 results
        logger.info(f"Extracting content from top {min(3, len(results))} pages...")
        detailed_results = []
        
        for idx, result in enumerate(results[:3]):
            result_text = f"{idx+1}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}"
            
            try:
                page_content = extract_page_content(driver, result['url'], max_chars=800)
                result_text += f"\n   Content Preview: {page_content}"
            except Exception as e:
                logger.warning(f"Could not extract content from {result['url']}: {e}")
                result_text += f"\n   Content Preview: (unable to extract)"
            
            detailed_results.append(result_text)
        
        for idx, result in enumerate(results[3:], start=3):
            detailed_results.append(f"{idx+1}. {result['title']}\n   {result['snippet']}\n   URL: {result['url']}")
        
        return "\n\n".join(detailed_results)
        
    except Exception as e:
        logger.error(f"Browser search failed: {e}")
        return f"Search failed: {str(e)}"
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# For LangChain Tool compatibility
class BrowserSearchRun:
    """Wrapper class to make browser_search compatible with LangChain."""
    
    def run(self, query: str) -> str:
        return browser_search(query)

