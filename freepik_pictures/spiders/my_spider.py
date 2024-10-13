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

    def __init__(self, csv_file=None, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)

        if not csv_file or not os.path.exists(csv_file):
            raise ValueError("You must provide a valid CSV file path")

        self.csv_file = csv_file

        # Load the sentence transformer model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Dynamically adjust Scrapy settings based on system resources
        self.custom_settings = self.adjust_scrapy_settings()

        # Configure logging
        self.configure_logging()

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
            "RETRY_ENABLED": False  # Disable retry to avoid delays
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
        with open(self.csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if not row.get('Scraped') or row['Scraped'] != 'Yes':
                    topic = row['Topic']
                    query = "+".join(topic.lower().split())
                    api_url = f"https://www.freepik.com/api/regular/search?filters[ai-generated][only]=1&filters[content_type]=photo&filters[license]=premium&locale=en&term={query}"

                    yield scrapy.Request(
                        url=api_url,
                        callback=self.parse,
                        meta={'row': row, 'topic': topic},
                        dont_filter=True
                    )

    def parse(self, response):
        self.logger.info(f"Processing API response for topic '{response.meta['topic']}'")

        data = response.json()
        if 'items' not in data or len(data['items']) == 0:
            self.logger.warning(f"No items found for topic '{response.meta['topic']}'")
            self.update_csv_with_image_path(response.meta['row'], '', 'No')
            return

        # Extract image names and URLs from API response
        image_urls = []
        image_descriptions = []
        for item in data['items']:
            img_url = item['preview']['url']
            image_urls.append(img_url)
            image_descriptions.append(item['name'])  # Image description/name for relevance checking

        # Check relevance of images
        yield from self.check_relevance_and_download_best_image(image_urls, image_descriptions, response.meta['topic'], response.meta['row'])

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

            # Download all images, skipping those that already exist
            for img_url, similarity in sorted_images:
                is_best = img_url == best_image_url
                file_path = self.get_image_file_path(img_url, is_best)
                if not os.path.exists(file_path):
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
                else:
                    self.logger.info(f"Skipping download, file already exists: {file_path}")
                    if is_best:
                        # Update main CSV for the most relevant image
                        self.update_csv_with_image_path(row, file_path, 'Yes')
                    else:
                        # Add the image to the IR CSV for non-relevant images
                        self.update_ir_csv_with_image(file_path)

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
            else:
                # Add the image to the additional images CSV
                self.update_ir_csv_with_image(file_path)

        except Exception as e:
            self.logger.error(f"Failed to save image for topic '{topic}': {str(e)}")
            self.update_csv_with_image_path(row, '', 'No')

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
