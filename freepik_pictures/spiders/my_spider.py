import scrapy
import os
import csv
import time
import json
import logging
import requests
import psutil
import speedtest
import sys
from sentence_transformers import SentenceTransformer, util
import urllib.parse

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/2)

class MySpider(scrapy.Spider):
    name = 'my_spider'
    
    custom_settings = {
        'CONCURRENT_REQUESTS': 1,  # Reduce concurrent requests
        'DOWNLOAD_DELAY': 2,      # Add delay between requests
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [403, 429, 500, 502, 503, 504],
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
            'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
        }
    }

    def __init__(self, csv_file=None, download_ir=False, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)

        if not csv_file or not os.path.exists(csv_file):
            raise ValueError("You must provide a valid CSV file path")

        self.csv_file = csv_file
        self.download_ir = str(download_ir).lower() == 'true'
        
        # Load the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize performance monitoring
        self.last_speed_check = 0
        self.speed_check_interval = 300  # Check every 5 minutes
        self.success_count = 0
        self.fail_count = 0
        self.current_delay = 0.1

        # Initialize required attributes
        self.batch_size = 100  # Process 10 items at a time
        self.processed_topics = set()  # Keep track of processed topics
        self.total_topics = 0
        self.completed_topics = 0
        self.cookies = {}  # Initialize empty cookies dict
        
        # Configure logging
        self.configure_logging()
        
        # Initial settings adjustment
        self.custom_settings = self.adjust_scrapy_settings()

    def adjust_scrapy_settings(self):
        """Dynamically adjust settings based on system and network performance"""
        try:
            # Get CPU and memory info
            cpu_cores = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            memory_available_gb = memory.available / (1024 * 1024 * 1024)
            cpu_percent = psutil.cpu_percent(interval=1)

            # Check network speed if enough time has passed
            current_time = time.time()
            if current_time - self.last_speed_check > self.speed_check_interval:
                try:
                    st = speedtest.Speedtest()
                    st.get_best_server()
                    download_speed = st.download() / 1_000_000  # Mbps
                    self.last_speed_check = current_time
                except Exception as e:
                    self.logger.warning(f"Network speed check failed: {str(e)}")
                    download_speed = 50  # Default fallback
            else:
                download_speed = 50  # Use default between checks

            # Calculate success rate
            total_requests = self.success_count + self.fail_count
            success_rate = self.success_count / total_requests if total_requests > 0 else 1

            # Adjust concurrent requests based on resources and success rate
            base_concurrent = min(int(cpu_cores * 8), int(download_speed // 5))
            adjusted_concurrent = int(base_concurrent * success_rate)
            
            # Adjust delay based on CPU usage and success rate
            if cpu_percent > 80:
                self.current_delay = min(self.current_delay * 1.5, 1.0)
            elif cpu_percent < 50 and success_rate > 0.9:
                self.current_delay = max(self.current_delay * 0.8, 0.1)

            # Log current performance metrics
            self.logger.info(f"""
            Performance Metrics:
            - CPU Usage: {cpu_percent}%
            - Memory Available: {memory_available_gb:.2f}GB
            - Download Speed: {download_speed:.2f}Mbps
            - Success Rate: {success_rate:.2%}
            - Concurrent Requests: {adjusted_concurrent}
            - Request Delay: {self.current_delay:.3f}s
            """)

            return {
                "CONCURRENT_REQUESTS": adjusted_concurrent,
                "CONCURRENT_REQUESTS_PER_DOMAIN": adjusted_concurrent,
                "DOWNLOAD_TIMEOUT": min(30, max(15, int(30 / success_rate))),
                "RETRY_ENABLED": True,
                "RETRY_TIMES": min(5, max(2, int(5 * (1 - success_rate)))),
                "DOWNLOAD_DELAY": self.current_delay,
                "COOKIES_ENABLED": True,
                "REACTOR_THREADPOOL_MAXSIZE": cpu_cores * 2,
                "LOG_LEVEL": 'INFO',
                "CONCURRENT_ITEMS": adjusted_concurrent * 2,
                "DEPTH_PRIORITY": 1,
                "SCHEDULER_DISK_QUEUE": 'scrapy.squeues.PickleLifoDiskQueue',
                "SCHEDULER_MEMORY_QUEUE": 'scrapy.squeues.LifoMemoryQueue',
                "JOBDIR": 'jobs',  # Enable job persistence
            }

        except Exception as e:
            self.logger.error(f"Error adjusting scrapy settings: {str(e)}")
            return {
                "CONCURRENT_REQUESTS": 10,
                "CONCURRENT_REQUESTS_PER_DOMAIN": 10,
                "DOWNLOAD_TIMEOUT": 15,
                "RETRY_ENABLED": True,
                "RETRY_TIMES": 2,
                "DOWNLOAD_DELAY": 0.1,
                "COOKIES_ENABLED": True,
                "REACTOR_THREADPOOL_MAXSIZE": 20,
                "LOG_LEVEL": 'INFO',
                "CONCURRENT_ITEMS": 20,
                "DEPTH_PRIORITY": 1,
                "SCHEDULER_DISK_QUEUE": 'scrapy.squeues.PickleLifoDiskQueue',
                "SCHEDULER_MEMORY_QUEUE": 'scrapy.squeues.LifoMemoryQueue',
                "JOBDIR": 'jobs',
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
        # Add proper headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.freepik.com/',
            'Origin': 'https://www.freepik.com',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'DNT': '1',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }

        # Add cookies if you have them
        self.cookies = {
            # Add any required cookies here
        }

        with open(self.csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if row.get('Topic') and not row.get('Scraped') == 'Yes':
                    yield scrapy.Request(
                        url=self.get_search_url(row['Topic']),
                        callback=self.parse_search_results,
                        headers=self.headers,
                        cookies=self.cookies,
                        meta={
                            'topic': row['Topic'],
                            'row': row,
                            'dont_retry': False,
                            'download_timeout': 30
                        },
                        dont_filter=True,
                        errback=self.handle_error
                    )
                    # Add delay between requests
                    time.sleep(2)

    def handle_error(self, failure):
        """Handle request failures"""
        request = failure.request
        topic = request.meta.get('topic', '')
        self.logger.error(f'Request failed for topic "{topic}": {failure.value}')
        
        # Update CSV to mark as failed
        row = request.meta.get('row', {})
        self.update_csv_with_image_path(row, '', 'Failed')

    def get_search_url(self, topic):
        """Create search URL with proper encoding"""
        base_url = 'https://www.freepik.com/api/regular/search'
        params = {
            'filters[ai-generated][only]': 1,
            'filters[content_type]': 'photo',
            'locale': 'en',
            'term': topic.strip()
        }
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def parse(self, response):
        topic = response.meta['topic']
        row = response.meta['row']

        # Handle HTTP errors
        if response.status != 200:
            self.logger.error(f"HTTP {response.status} for topic '{topic}'")
            self.update_csv_with_image_path(row, '', f'Error: HTTP {response.status}')
            self.completed_topics += 1
            return

        try:
            data = response.json()
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response for topic '{topic}'")
            self.update_csv_with_image_path(row, '', 'Error: Invalid JSON')
            self.completed_topics += 1
            return

        if 'items' not in data or len(data['items']) == 0:
            self.logger.warning(f"No items found for topic '{topic}'")
            self.update_csv_with_image_path(row, '', 'No results')
            self.completed_topics += 1
            return

        # Extract only necessary information
        images = [(item['preview']['url'], item['name']) for item in data['items']]
        
        if not images:
            return

        # Process relevance check
        yield from self.process_images(images, response.meta['topic'], response.meta['row'])

    def process_images(self, images, topic, row):
        try:
            image_urls, image_descriptions = zip(*images)
            
            # Calculate relevance scores
            topic_embedding = self.model.encode(topic, convert_to_tensor=True)
            image_embeddings = self.model.encode(image_descriptions, convert_to_tensor=True)
            similarities = util.cos_sim(topic_embedding, image_embeddings).squeeze()

            # Get the most relevant image
            best_idx = similarities.argmax().item()
            best_image_url = image_urls[best_idx]
            best_similarity = similarities[best_idx].item()

            # Download the best image
            file_path = self.get_image_file_path(best_image_url, True)
            if not os.path.exists(file_path):
                yield scrapy.Request(
                    url=best_image_url,
                    callback=self.save_image,
                    meta={
                        'topic': topic,
                        'row': row,
                        'file_path': file_path,
                        'is_best': True
                    },
                    priority=2  # Highest priority for best images
                )

            # Download other images if download_ir is True
            if self.download_ir:
                for idx, (img_url, similarity) in enumerate(zip(image_urls, similarities)):
                    if idx != best_idx:  # Skip the best image as it's already being downloaded
                        file_path = self.get_image_file_path(img_url, False)
                        if not os.path.exists(file_path):
                            yield scrapy.Request(
                                url=img_url,
                                callback=self.save_image,
                                meta={
                                    'topic': topic,
                                    'row': row,
                                    'file_path': file_path,
                                    'is_best': False
                                },
                                priority=0  # Lower priority for IR images
                            )

        except Exception as e:
            self.logger.error(f"Error in processing images for topic '{topic}': {str(e)}")
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
                # Increment completed topics counter
                self.completed_topics += 1
                print(f"\nCompleted {self.completed_topics}/{self.total_topics} topics")
                
                # Check if all topics are completed
                if self.completed_topics >= self.total_topics:
                    print("\n=================================")
                    print("All topics have been processed!")
                    print("=================================\n")
                    self.crawler.engine.close_spider(self, 'All topics completed')
            else:
                # Add the image to the additional images CSV
                self.update_ir_csv_with_image(file_path)

        except Exception as e:
            self.logger.error(f"Failed to save image for topic '{topic}': {str(e)}")
            self.update_csv_with_image_path(row, '', 'No')
            # Still increment counter even if there was an error
            self.completed_topics += 1

    def update_csv_with_image_path(self, row, image_path, scraped):
        # Update the main CSV file with the relevant image
        try:
            with open(self.csv_file, mode='r', newline='') as file:
                csv_reader = list(csv.DictReader(file))
                fieldnames = csv_reader[0].keys()

            # Update the specific row
            for r in csv_reader:
                if r['Topic'] == row['Topic']:
                    r['Image Path'] = image_path
                    r['Scraped'] = scraped
                    break

            # Write the updated rows back to the main CSV
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_reader)

        except Exception as e:
            self.logger.error(f"Error updating main CSV file: {str(e)}")

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

    def update_performance_metrics(self, success=True):
        """Update success/failure counts and adjust settings if needed"""
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

        # Adjust settings every 100 requests or if success rate drops below 70%
        total_requests = self.success_count + self.fail_count
        if total_requests % 100 == 0 or (total_requests > 10 and self.success_count / total_requests < 0.7):
            new_settings = self.adjust_scrapy_settings()
            self.crawler.settings.update(new_settings)

    def parse_search_results(self, response):
        topic = response.meta['topic']
        row = response.meta['row']

        try:
            # Check if response is valid
            if response.status == 403:
                self.logger.error(f"Access denied for topic '{topic}'. Might need authentication.")
                self.update_csv_with_image_path(row, '', 'Auth Required')
                return
            
            # Parse JSON response
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON response for topic '{topic}'")
                self.update_csv_with_image_path(row, '', 'Invalid Response')
                return

            # Process results
            if not data.get('data'):
                self.logger.warning(f"No results found for topic '{topic}'")
                self.update_csv_with_image_path(row, '', 'No Results')
                return

            # Continue with your existing processing logic
            # ...

        except Exception as e:
            self.logger.error(f"Error processing topic '{topic}': {str(e)}")
            self.update_csv_with_image_path(row, '', 'Error')
