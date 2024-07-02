import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

class TestVersion2():
    def setup_method(self, method):
        self.driver = webdriver.Firefox()
        self.driver.set_window_size(1472, 913)
        self.vars = {}
        self.data = []  # To store extracted data

    def teardown_method(self, method):
        self.driver.quit()

    def wait_for_window(self, timeout=2):
        time.sleep(timeout)
        wh_now = self.driver.window_handles
        wh_then = self.vars["window_handles"]
        if len(wh_now) > len(wh_then):
            return set(wh_now).difference(set(wh_then)).pop()

    def open_and_select_university(self, university):
        self.driver.get("https://tez.yok.gov.tr/UlusalTezMerkezi/tarama.jsp")
        self.vars["window_handles"] = self.driver.window_handles
        self.driver.find_element(By.NAME, "uniad").click()
        self.vars["new_window"] = self.wait_for_window(2)
        self.driver.switch_to.window(self.vars["new_window"])
        self.driver.find_element(By.CSS_SELECTOR, ".renka:nth-child(4) > td").click()
        self.driver.find_element(By.ID, "filt").click()
        self.driver.find_element(By.ID, "filt").send_keys(university)
        self.driver.find_element(By.LINK_TEXT, university).click()
        self.driver.switch_to.window(self.vars["window_handles"][0])

    def fill_and_search(self, subject):
        self.driver.find_element(By.ID, "konu").click()
        self.driver.find_element(By.ID, "konu").send_keys(subject)
        self.driver.find_element(By.ID, "tabs-1").click()
        self.driver.find_element(By.NAME, "-find").click()

    def wait_for_overlay_to_disappear(self):
        WebDriverWait(self.driver, 10).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "ui-widget-overlay"))
        )

    def extract_info(self, page_number, row_number, university, subject):
        try:
            self.wait_for_overlay_to_disappear()  # Ensure overlay is not present
            row_selector = f"tr:nth-child({row_number}) > td:nth-child(2) > span"
            element = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, row_selector))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()

            def get_element_text(element_id):
                element = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, element_id))
                )
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
                actions = ActionChains(self.driver)
                actions.double_click(element).perform()
                return self.driver.find_element(By.ID, element_id).text

            tr = get_element_text("td0")
            en = get_element_text("td1")

            close_button = self.driver.find_element(By.CSS_SELECTOR, ".ui-icon")
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", close_button)
            close_button.click()
            # Append the data to the list
            self.data.append({
                "university": university,
                "konu": subject,
                "tr": tr,
                "en": en
            })

        except Exception as e:
            print(f"Error: {e}")

    def go_to_page(self, page_number):
        current_page = 1
        while current_page < page_number:
            next_page_link = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, str(current_page + 1)))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_page_link)
            time.sleep(1)
            next_page_link.click()
            current_page += 1

    def do_for_university_and_subject(self, university, subject, total_num_rows, start_page=1):
        self.open_and_select_university(university)
        self.fill_and_search(subject)

        rows_per_page = 30
        total_pages = (total_num_rows + rows_per_page - 1) // rows_per_page  # Total pages required
        last_page_rows = total_num_rows % rows_per_page

        if start_page > 1:
            self.go_to_page(start_page)

        for page_number in range(start_page, total_pages + 1):
            if page_number == total_pages and last_page_rows != 0:
                rows = last_page_rows
            else:
                rows = rows_per_page

            for row_number in range(1, rows + 1):
                self.extract_info(page_number, row_number, university, subject)

            # Save the data for each page
            df = pd.DataFrame(self.data)
            df.to_csv(f"./output/{university}_{subject}_page_{page_number}.csv", index=False)
            self.data = []  # Clear data after saving each page

            if page_number < total_pages:
                next_page_link = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, str(page_number + 1)))
                )
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_page_link)
                time.sleep(1)
                next_page_link.click()

    def test_version2(self):
        universities = [
            "Boğaziçi Üniversitesi",
            "Koç Üniversitesi",
            "İhsan Doğramacı Bilkent Üniversitesi",
            "Sabancı Üniversitesi",
            "Orta Doğu Teknik Üniversitesi",
            "İstanbul Teknik Üniversitesi"
        ]

        subjects = [
            "Bilgisayar Mühendisliği Bilimleri-Bilgisayar ve Kontrol = Computer Engineering and Computer Science and Control",
            "Matematik = Mathematics",
            "Fizik ve Fizik Mühendisliği = Physics and Physics Engineering",
            "Elektrik ve Elektronik Mühendisliği = Electrical and Electronics Engineering",
            "Endüstri ve Endüstri Mühendisliği = Industrial and Industrial Engineering",
            "Makine Mühendisliği = Mechanical Engineering"
        ]

        total_num_rows_list = [1001, 204, 303, 935, 710, 604,
                               215, 92, 132, 234, 210, 231,
                               685, 189, 336, 874, 417, 110,
                               326, 110, 60, 281, 215, 121,
                               2000, 772, 963, 2000, 737, 1881,
                               1422, 314, 467, 2000, 1167, 2000]
        start_page = 1

        for university in universities:
            for subject, total_num_rows in zip(subjects, total_num_rows_list):
                self.do_for_university_and_subject(university, subject, total_num_rows, start_page)
