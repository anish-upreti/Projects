import requests

urls = {
    "alice.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "pride.txt": "https://www.gutenberg.org/files/1342/1342-0.txt"
}

for filename, url in urls.items():
    r = requests.get(url)
    with open(f"raw_texts/{filename}", "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"{filename} downloaded.")
