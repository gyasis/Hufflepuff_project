import pandas as pd
import selenium
from selenium import webdriver
import time


# This is the path I use
# DRIVER_PATH = '.../Desktop/Scraping/chromedriver 2'
# Put the path for your ChromeDriver hereabs
DRIVER_PATH = '/Users/tobiasschulz/Documents/Scraping/chromedriver'
wd = webdriver.Chrome(executable_path=DRIVER_PATH)


#wd.get('https://google.com')

#search_box = wd.find_element_by_css_selector('input.gLFyf')
#search_box.send_keys('Dogs')



df = pd.read_csv('data_final.csv')
dp = df.iloc[700:799]

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.de/search?as_st=y&tbm=isch&as_q={q}&as_epq=map&as_oq=&as_eq=set+box+boxe+coverd&cr=&as_sitesearch=&safe=images&tbs=isz:xxl"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = []
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        if number_results == 0:
            image_urls.append('NaN')
        else:
            print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
            
            for img in thumbnail_results[results_start:number_results]:
                # try to click every thumbnail such that we can get the real image behind it
                try:
                    img.click()
                    time.sleep(sleep_between_interactions)
                except Exception:
                    continue

                # extract image urls    
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.append(actual_image.get_attribute('src'))
                    #else:
                        #image_urls.append('NaN')
                        
                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    break
            else:
                image_urls.append('NaN')
                #print("Found:", len(image_urls), "image links, looking for more ...")
                #time.sleep(10)
                #return
                #load_more_button = wd.find_element_by_css_selector(".mye4qd")
                #if load_more_button:
                #    wd.execute_script("document.querySelector('.mye4qd').click();")

            # move the result startpoint further down
            results_start = len(thumbnail_results)

        return image_urls
    
    wd.quit()

scraping_count = 0

for i in df.index:
    x = 'map from world in the book ' + str(df.at[i, 'Title']) 
    df.at[i, 'map_url'] = fetch_image_urls(x, 1, wd)[0]
    scraping_count +=1
    title = str(df.at[i, 'Title'])
    url = df.at[i, 'map_url']
    print(f'Inserted {url} in {title} at {i}.')
    print(f'Currently scraped {scraping_count} maps.')

df.to_csv('data_final.csv')

print(df.at[1, 'Title'])
print(df.at[1, 'map_url'])
print(df.at[2, 'Title'])
print(df.at[2, 'map_url'])
print(df.at[3, 'Title'])
print(df.at[3, 'map_url'])
print(df.at[4, 'Title'])
print(df.at[4, 'map_url'])
print(df['map_url'])