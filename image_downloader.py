from time import sleep
import random
import requests
import cv2
import os


def download_url(url, filepath):
	"""
	Downloads and saves a file to a given filepath
	:param url: URL to file
	:param filepath: Destination path to file
	"""
	try:
		img_data = requests.get(url, timeout=60).content
		with open(filepath, 'wb') as handler:
			print("[INFO] downloaded: {}".format(url))
			handler.write(img_data)
			sleep(0.5)

	except requests.exceptions.ConnectionError:
		print("[ERROR] error downloading image {}...skipping".format(url))


def download_urls(url_path, destination_folder, relative_name="banana"):
	"""
	Downloads and saves files from a list of urls to a given destination folder
	:param url_path: URLs to files for scraping
	:param destination_folder: Destination folder for file-saving
	:param relative_name: Naming of retrieved files - will be enumerated
	"""

	urls = open(url_path, 'r').read().strip().split('\n')
	for index, url in enumerate(urls):
		try:
			req = requests.get(url, timeout=60)

			# Save image to specified destination folder with relative name
			path = os.path.sep.join([destination_folder, "{}_{}.jpg".format(relative_name, str(index).zfill(6))])
			f = open(path, "wb")
			f.write(req.content)
			f.close()

			print("[INFO] Downloaded: {}".format(url))

		# Handle if any exceptions are thrown during the download process
		except:
			print("[ERROR] Error downloading {}".format(url))


# download_urls('urls/google_banana_urls.txt', 'google_images/training/banana', 'g_banana')


def remove_invalid_pictures(image_paths):
	"""
	Remove invalid pictures in a given folder
	:param image_paths: Relative or absolute path to folder with images
	:return Number of deleted images
	"""
	images = os.listdir(image_paths)
	for image in images:
		relative_path = '{}/{}'.format(image_paths, image)
		delete = False
		try:
			image = cv2.imread(relative_path)
			delete = True if image is None else delete

		# If OpenCV cannot load image - delete
		except:
			delete = True

		if delete:
			print("[INFO] deleting {}".format(relative_path))
			os.remove(relative_path)

	deleted_images = len(images) - len(os.listdir(image_paths))
	print("Removed {} images".format(deleted_images))

	return deleted_images


# remove_invalid_pictures('google_images/training/banana')


def download_google():
	"""
	Downloads images from the url list from Google Images, scraped with 'google_images_scraper.js'
	:return:
	"""

	urls = open("google_banana_urls.txt", "r").readlines()
	for index, url in enumerate(urls):
		image_type = 'jpg' if 'jpg' in url else 'jpeg'
		filepath = 'google_images/training/banana/g_banana_{}.{}'.format(index, image_type)
		download_url(url, filepath)


def download_synset():
	"""
	Downloads images from the url list from ImageNet - SynSet
	"""

	url_path = "urls/synset_banana_urls.txt"
	download_urls(url_path, 'synset_fruit365/training/banana')


def download_random(n, destination_path):
	"""
	Download n random images from <lorempixel>
	:return:
	"""
	for i in range(0, n):
		category = random.choice(['city', 'people', 'sports', 'technics'])
		width, height = random.randint(400, 500), random.randint(400, 500)
		filepath = destination_path + '/random_{}.jpg'.format(i)
		download_url('http://lorempixel.com/{}/{}/{}/'.format(width, height, category), filepath)

# download_random(1000, 'synset_fruit365/training/other')
