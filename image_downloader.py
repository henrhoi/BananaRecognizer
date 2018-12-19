from time import sleep
import random
import requests
import cv2
import os


def download_url(url, filepath):
	try:
		img_data = requests.get(url, timeout=60).content
		with open(filepath, 'wb') as handler:
			print("[INFO] downloaded: {}".format(url))
			handler.write(img_data)
			sleep(0.5)


	except requests.exceptions.ConnectionError:
		print("[ERROR] error downloading image {}...skipping".format(url))


def download_urls(url_path, destination_folder, relative_name="banana"):
	urls = open(url_path, 'r').read().strip().split('\n')
	for index, url in enumerate(urls):
		try:
			req = requests.get(url, timeout=60)

			# Save image to specified destination folder with relative name
			path = os.path.sep.join([destination_folder, "{}_{}.jpg".format(relative_name, str(index).zfill(6))])
			f = open(path, "wb")
			f.write(req.content)
			f.close()

			# update the counter
			print("[INFO] Downloaded: {}".format(url))

		# handle if any exceptions are thrown during the download process
		except:
			print("[ERROR] Error downloading {}".format(url))


#download_urls('urls/google_banana_urls.txt', 'google_images/training/banana', 'g_banana')


def remove_invalid_pictures(image_paths):
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

	print("Removed {} images".format(len(images) - len(os.listdir(image_paths))))


#remove_invalid_pictures('google_images/training/banana')


def download_google():
	urls = open("google_banana_urls.txt", "r").readlines()
	for index, url in enumerate(urls):
		image_type = 'jpg' if 'jpg' in url else 'jpeg'
		filepath = 'google_images/training/banana/g_banana_{}.{}'.format(index, image_type)
		download_url(url, filepath)


# download_google()


def download_synset():
	url_path = "urls/synset_banana_urls.txt"
	download_urls(url_path, 'synset_fruit365/training/banana')


def download_random():
	for i in range(0, 1000):
		category = random.choice(['city', 'people', 'sports', 'technics'])
		width, height = random.randint(400, 500), random.randint(400, 500)
		filepath = 'synset_fruit365/training/other/random_{}.jpg'.format(i)

		download_url('http://lorempixel.com/{}/{}/{}/'.format(width, height, category), filepath)
