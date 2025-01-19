from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd

options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

website ='https://www.youtube.com/feed/trending?bp=6gQJRkVleHBsb3Jl'
chrome_service = Service(executable_path="chromedriver.exe")
driver = webdriver.Chrome(service=chrome_service,options=options)
driver.get(website)
driver.implicitly_wait(30)

videos = driver.find_elements(By.CLASS_NAME, 'style-scope ytd-expanded-shelf-contents-renderer')

vid_list = []

for video in videos :
    try:
        title = driver.find_element(by='xpath',value='.//*[@id="video-title"]').text
        channel = driver.find_element(by='xpath',value='.//*[@id="text"]/a').text
        views = driver.find_element(by='xpath',value='.//*[@id="metadata-line"]/span[1]').text
        date_posted = driver.find_element(by='xpath',value='.//*[@id="metadata-line"]/span[2]').text
        vid_dict = {
            'title' : title,
            'channel' : channel,
            'views' : views,
            'date_posted' : date_posted
        }
        vid_list.append(vid_dict)
    except Exception as e:
        print(f"Error: {e}")
        
print(vid_list)

df = pd.DataFrame(vid_list)
print(df)
        
    
    
    

    









