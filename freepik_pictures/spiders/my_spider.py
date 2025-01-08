import scrapy
import os
import csv
import time
import logging
import requests
import psutil
import speedtest
from sentence_transformers import SentenceTransformer, util

class MySpider(scrapy.Spider):
    name = 'my_spider'

    def __init__(self, csv_file=None, download_ir='false', max_retries=3, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)

        if not csv_file or not os.path.exists(csv_file):
            raise ValueError("You must provide a valid CSV file path")

        self.csv_file = csv_file
        self.download_ir = download_ir.lower() == 'true'
        self.max_retries = int(max_retries)
        self.retry_topics = {}  # Track retry attempts per topic
        self.logger.info(f"IR Download setting: {self.download_ir}")
        self.logger.info(f"Max retries per topic: {self.max_retries}")

        # Load the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Dynamically adjust Scrapy settings based on system resources
        self.custom_settings = self.adjust_scrapy_settings()

        # Configure logging
        self.configure_logging()

        # Add headers and cookies as class attributes
        self.headers = {
            'authority': 'www.freepik.com',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'accept-language': 'en-US,en;q=0.9,el;q=0.8,ur;q=0.7,sl;q=0.6',
            'cache-control': 'no-cache',
            'dnt': '1',
            'pragma': 'no-cache',
            'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        }
        
        self.cookies = {
            '_cs_ex': '1709818470',
            '_cs_c': '0',
            'usprivacy': '1---',
            'OptanonAlertBoxClosed': '2024-09-21T10:39:20.541Z',
            'OneTrustWPCCPAGoogleOptOut': 'false',
            '_au_1d': 'AU1D-0100-001709006278-QV0BJASJ-L0AE',
            '_hjSessionUser_1331604': 'eyJpZCI6IjJhZjliM2IwLWQ5ZDMtNWQwOC1iYzVjLTU2ZWM2ZWQyNzY3ZCIsImNyZWF0ZWQiOjE3MjY5MTUxNjExMTUsImV4aXN0aW5nIjp0cnVlfQ==',
            'cto_optout': '1',
            'GRID': '71985952',
            'premiumGen': 'B',
            'EXPCH': 'true',
            'new_regular_detail_test': 'A',
            'TUNES_IN_VIDEO': '1',
            'premiumQueue': 'A',
            'skip_expch': 'true'
        }

    def adjust_scrapy_settings(self):
        """Dynamically adjust Scrapy settings based on system performance."""
        # Get the number of CPU cores
        cpu_cores = psutil.cpu_count(logical=False)  # Get physical cores

        # Get network speed (Download speed in Mbps)
        try:
            st = speedtest.Speedtest()
            st.get_best_server()
            download_speed = st.download() / 1_000_000  # Convert to Mbps
        except Exception as e:
            self.logger.warning(f"Network speed check failed: {str(e)}")
            download_speed = 50  # Fallback to a reasonable default

        # Set CONCURRENT_REQUESTS based on CPU cores and network speed
        concurrent_requests = min(int(cpu_cores * 4), int(download_speed // 10))

        # Adjust other settings based on network speed (more network = lower timeout)
        download_timeout = 30 if download_speed > 100 else 60

        return {
            "CONCURRENT_REQUESTS": concurrent_requests,  # Adjust based on system
            "DOWNLOAD_TIMEOUT": download_timeout,  # Adjust based on network speed
            "RETRY_ENABLED": False,  # Disable retry to avoid delays
            "DOWNLOAD_FAIL_ON_DATALOSS": False,
            "RETRY_TIMES": 0,
            "RETRY_HTTP_CODES": []
        }

    def configure_logging(self):
        # Create a custom logger for timing messages
        self.timing_logger = logging.getLogger('timing_logger')
        self.timing_logger.setLevel(logging.INFO)

        # Create handlers
        timing_handler = logging.FileHandler('timing.log')
        timing_handler.setLevel(logging.INFO)

        other_handler = logging.FileHandler('other_logs.log')
        other_handler.setLevel(logging.DEBUG)

        # Create formatters
        timing_formatter = logging.Formatter('%(message)s')
        other_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # Add formatters to handlers
        timing_handler.setFormatter(timing_formatter)
        other_handler.setFormatter(other_formatter)

        # Add handlers to loggers
        self.timing_logger.addHandler(timing_handler)

        # Get the Scrapy logger and add the other handler
        scrapy_logger = logging.getLogger()
        scrapy_logger.addHandler(other_handler)

    def start_requests(self):
        unscraped_topics = self.get_unscraped_topics()
        
        if not unscraped_topics:
            self.logger.info("All topics have been scraped successfully!")
            return

        self.logger.info(f"Found {len(unscraped_topics)} unscraped topics to process")
        
        for row in unscraped_topics:
            topic = row['Topic']
            query = "+".join(topic.lower().split())
            api_url = f"https://www.freepik.com/api/regular/search?filters[ai-generated][only]=1&filters[content_type]=photo&locale=en&term={query}"

            # Initialize retry count for this topic
            if topic not in self.retry_topics:
                self.retry_topics[topic] = 0

            self.logger.info(f"Processing topic: {topic} (Attempt {self.retry_topics[topic] + 1})")
            yield scrapy.Request(
                url=api_url,
                callback=self.parse,
                meta={
                    'row': row, 
                    'topic': topic,
                    'dont_retry': False,
                    'max_retry_times': 3
                },
                headers=self.headers,
                cookies=self.cookies,
                dont_filter=True,
                errback=self.handle_error,
                priority=10  # Higher priority for initial requests
            )

    def handle_error(self, failure):
        """Handle any errors during request"""
        row = failure.request.meta.get('row')
        topic = failure.request.meta.get('topic')
        
        if topic in self.retry_topics:
            self.retry_topics[topic] += 1
            
            if self.retry_topics[topic] < self.max_retries:
                self.logger.warning(f"Retrying topic '{topic}' (Attempt {self.retry_topics[topic] + 1})")
                
                # Create a new request with the same parameters
                query = "+".join(topic.lower().split())
                api_url = f"https://www.freepik.com/api/regular/search?filters[ai-generated][only]=1&filters[content_type]=photo&locale=en&term={query}"
                
                return scrapy.Request(
                    url=api_url,
                    callback=self.parse,
                    meta={
                        'row': row, 
                        'topic': topic,
                        'dont_retry': False,
                        'max_retry_times': 3
                    },
                    headers=self.headers,
                    cookies=self.cookies,
                    dont_filter=True,
                    errback=self.handle_error,
                    priority=20  # Higher priority for retry requests
                )
        
        self.logger.error(f"Failed to process topic '{topic}' after {self.max_retries} attempts: {str(failure.value)}")
        if row:
            self.update_csv_with_image_path(row, '', 'No')

    def get_unscraped_topics(self):
        unscraped = []
        try:
            with open(self.csv_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    # Check multiple conditions for considering a topic unscraped:
                    # 1. No 'Scraped' field or not 'Yes'
                    # 2. No 'Image Path' field
                    # 3. Has 'Image Path' but file doesn't exist
                    # 4. Has 'Image Path' but it's empty
                    if (not row.get('Scraped') or row['Scraped'].lower() != 'yes' or
                        not row.get('Image Path') or
                        not row['Image Path'].strip() or
                        (row.get('Image Path') and not os.path.exists(row['Image Path']))):
                        unscraped.append(row)
                        self.logger.debug(f"Topic '{row['Topic']}' needs processing because: "
                                       f"Scraped={row.get('Scraped')}, "
                                       f"Image Path={row.get('Image Path')}, "
                                       f"File exists={os.path.exists(row.get('Image Path', ''))}")
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {str(e)}")
            raise
        return unscraped

    def parse(self, response):
        topic = response.meta['topic']
        self.logger.info(f"Processing API response for topic '{topic}'")

        try:
            data = response.json()
            if 'items' not in data or len(data['items']) == 0:
                self.logger.warning(f"No items found for topic '{topic}'")
                # Treat this as an error and retry
                raise ValueError(f"No items found for topic '{topic}'")

            # Extract image names and URLs from API response
            image_urls = []
            image_descriptions = []
            for item in data['items']:
                img_url = item['preview']['url']
                image_urls.append(img_url)
                image_descriptions.append(item['name'])  # Image description/name for relevance checking

            # Check relevance of images
            yield from self.check_relevance_and_download_best_image(image_urls, image_descriptions, topic, response.meta['row'])

        except Exception as e:
            self.logger.error(f"Error parsing response for topic '{topic}': {str(e)}")
            return self.handle_error(failure=scrapy.exceptions.IgnoreRequest(e))

    def check_relevance_and_download_best_image(self, image_urls, image_descriptions, topic, row):
        try:
            self.logger.info(f"Evaluating image relevance for topic '{topic}'")

            start_time = time.time()

            # Create embeddings for topic and images
            topic_embedding = self.model.encode(topic, convert_to_tensor=True)
            image_embeddings = self.model.encode(image_descriptions, convert_to_tensor=True)
            similarities = util.cos_sim(topic_embedding, image_embeddings).squeeze()

            # Find the most relevant image
            image_similarities = [
                (img_url, similarity.item())
                for img_url, similarity in zip(image_urls, similarities)
            ]
            sorted_images = sorted(image_similarities, key=lambda x: x[1], reverse=True)

            # Most relevant image is the first one
            best_image_url, best_similarity = sorted_images[0]

            self.logger.info(f"Most relevant image for topic '{topic}': {best_image_url} (similarity: {best_similarity})")

            # Log time taken for processing
            end_time = time.time()
            time_taken = end_time - start_time
            self.timing_logger.info(f"Topic '{topic}' took {time_taken:.2f} sec to process image relevance")

            # First check if best image exists
            best_file_path = self.get_image_file_path(best_image_url, is_best=True)
            if os.path.exists(best_file_path):
                self.logger.info(f"Best image already exists: {best_file_path}")
                # Update main CSV with existing best image and mark as scraped
                self.update_csv_with_image_path(row, best_file_path, 'Yes')
                
                # If download_ir is True, process other images
                if self.download_ir:
                    for img_url, _ in sorted_images[1:]:  # Skip the best image
                        file_path = self.get_image_file_path(img_url, is_best=False)
                        if os.path.exists(file_path):
                            self.update_ir_csv_with_image(file_path)
                        else:
                            yield scrapy.Request(
                                url=img_url,
                                callback=self.save_image,
                                meta={
                                    'topic': topic,
                                    'row': row,
                                    'file_path': file_path,
                                    'is_best': False
                                }
                            )
                return

            # If best image doesn't exist, process all images as before
            for img_url, similarity in sorted_images:
                is_best = img_url == best_image_url
                file_path = self.get_image_file_path(img_url, is_best)

                if os.path.exists(file_path):
                    self.logger.info(f"File already exists: {file_path}")
                    if is_best:
                        self.update_csv_with_image_path(row, file_path, 'Yes')
                    elif self.download_ir:
                        self.update_ir_csv_with_image(file_path)
                else:
                    if is_best or (not is_best and self.download_ir):
                        yield scrapy.Request(
                            url=img_url,
                            callback=self.save_image,
                            meta={
                                'topic': topic,
                                'row': row,
                                'file_path': file_path,
                                'is_best': is_best
                            }
                        )

        except Exception as e:
            self.logger.error(f"Error in relevance check for topic '{topic}': {str(e)}")
            self.update_csv_with_image_path(row, '', 'No')

    def get_image_file_path(self, img_url, is_best):
        # Relevant images go to "downloaded_images"
        # Non-relevant images go to "downloaded_images_ir"
        directory = 'downloaded_images' if is_best else 'downloaded_images_ir'
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_name = img_url.split('/')[-1]
        return os.path.join(directory, image_name)

    def save_image(self, response):
        try:
            file_path = response.meta['file_path']
            topic = response.meta['topic']
            row = response.meta['row']
            is_best = response.meta['is_best']

            self.logger.info(f"Saving image to {file_path}")

            start_time = time.time()

            try:
                with open(file_path, 'wb') as f:
                    f.write(response.body)
            except Exception as e:
                self.logger.error(f"Failed to save image: {str(e)}")
                return

            end_time = time.time()
            time_taken = end_time - start_time

            self.logger.info(f"Downloaded image for topic '{topic}': {file_path}")

            # Log time taken to download the image
            self.timing_logger.info(f"Topic '{topic}' took {time_taken:.2f} sec to download image")

            if is_best:
                # Update main CSV file for the most relevant image
                self.update_csv_with_image_path(row, file_path, 'Yes')
            elif self.download_ir:
                # Only update IR CSV if download_ir is True
                self.update_ir_csv_with_image(file_path)

        except Exception as e:
            self.logger.error(f"Failed to save image for topic '{topic}': {str(e)}")
            self.update_csv_with_image_path(row, '', 'No')

    def update_csv_with_image_path(self, row, image_path, scraped):
        """Update the CSV file with image path and scraped status"""
        try:
            # Read all rows
            with open(self.csv_file, mode='r', newline='') as file:
                csv_reader = list(csv.DictReader(file))
                fieldnames = csv_reader[0].keys()

            # Update the specific row
            updated = False
            for r in csv_reader:
                if r['Topic'] == row['Topic']:
                    r['Image Path'] = image_path
                    r['Scraped'] = scraped
                    updated = True
                    break

            if not updated:
                self.logger.error(f"Could not find topic '{row['Topic']}' in CSV file")
                return

            # Write all rows back to the CSV
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_reader)

            self.logger.info(f"Updated CSV for topic '{row['Topic']}': Image Path='{image_path}', Scraped='{scraped}'")

        except Exception as e:
            self.logger.error(f"Error updating CSV file for topic '{row['Topic']}': {str(e)}")
            raise

    def update_ir_csv_with_image(self, file_path):
        """Update or create the IR CSV file with all non-relevant images"""
        try:
            csv_file_ir = self.csv_file.replace('.csv', '_ir.csv')

            # Ensure the IR CSV file exists or create it with headers
            if not os.path.exists(csv_file_ir):
                with open(csv_file_ir, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Name', 'Path'])  # Add headers

            # Get the image name (without extension)
            image_name = os.path.basename(file_path).rsplit('.', 1)[0]

            # Append the image name and path to the IR CSV
            with open(csv_file_ir, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_name, file_path])

            self.logger.info(f"Added image '{image_name}' to '{csv_file_ir}'")

        except Exception as e:
            self.logger.error(f"Error updating IR CSV file: {str(e)}")

    def closed(self, reason):
        """Called when the spider is closed"""
        # Double check for any remaining unscraped topics
        unscraped = self.get_unscraped_topics()
        
        if unscraped:
            self.logger.warning(f"There are still {len(unscraped)} unscraped topics!")
            self.logger.warning("Unscraped topics:")
            for row in unscraped:
                topic = row['Topic']
                attempts = self.retry_topics.get(topic, 0)
                self.logger.warning(f"- {topic} (Attempted {attempts} times)")
            
            # If there are still unscraped topics, automatically retry them
            self.logger.info("Automatically retrying unscraped topics...")
            for row in unscraped:
                topic = row['Topic']
                query = "+".join(topic.lower().split())
                api_url = f"https://www.freepik.com/api/regular/search?filters[ai-generated][only]=1&filters[content_type]=photo&locale=en&term={query}"
                
                # Reset retry count for this topic
                self.retry_topics[topic] = 0
                
                return scrapy.Request(
                    url=api_url,
                    callback=self.parse,
                    meta={
                        'row': row, 
                        'topic': topic,
                        'dont_retry': False,
                        'max_retry_times': 3
                    },
                    headers=self.headers,
                    cookies=self.cookies,
                    dont_filter=True,
                    errback=self.handle_error,
                    priority=30  # Highest priority for final retry requests
                )
        else:
            self.logger.info("All topics have been successfully scraped!")