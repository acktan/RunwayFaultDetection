import urllib.error

import re
import os
import shutil
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from urllib.request import urlretrieve
from pyunpack import Archive
from tqdm import tqdm


class ScrapeImages:
    def __init__(geckoPath: str):
        self.driver = webdriver.Firefox(service=Service(executable_path=geckoPath))
        self.driver.get('https://geoservices.ign.fr/bdortho#telechargement')
        self.urls_to_download = None

    def get_download_urls(self) -> list[str]:
        """This method creates a list with the download links to the coordinate files."""
        # Navigating to the html part that contains the urls
        download_section = self.driver.find_element(By.XPATH,
                                                    '/html/body/div[2]/div[1]/div/section/div[3]/section/div/article/'
                                                    'div[2]/div/div[1]/div/div[2]/div[3]/div/div/div[2]')
        # Retrieving the download links
        self.urls_to_download = download_section.find_elements(By.XPATH,
                                                               ".//*")
        self.urls_to_download = [url.text for url in self.urls_to_download]
        self.urls_to_download = [s for s in downloadURLs if "://" in s]
        self.urls_to_download = [s.split('\n') for s in self.urls_to_download][0]
        self.driver.quit()
        return self.urls_to_download

    