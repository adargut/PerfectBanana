# Import google images download
from google_images_download import google_images_download

# Construct a response object
response = google_images_download.googleimagesdownload()

# Search query for google images
args = {"keywords": "banana",
        "limit": 21,  # TODO maybe a higher limit is needed for an efficient network?
        "print_urls": True,
        "size": "medium"}

# Download all images and print their absolute values
paths = response.download(args)
print(paths)
