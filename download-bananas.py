# Import google images download
from google_images_download import google_images_download

# Construct a response object
response = google_images_download.googleimagesdownload()

# Search query for google images
args = {"keywords": "underripe banana, ripe banana, overripe banana",
        "limit": 100,  # TODO maybe a higher limit is needed for an efficient network?
        "print_urls": True}

# Download all images and print their absolute values
paths = response.download(args)
print(paths)
