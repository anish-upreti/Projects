import glob

all_text = ""
for file_path in glob.glob("raw_texts/*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        all_text += text + "\n"

# Save combined raw corpus
with open("data/corpus.txt", "w", encoding='utf-8') as f:
    f.write(all_text)

print("Raw corpus.txt created in data/ folder.")


with open("data/corpus.txt", "r", encoding='utf-8') as f:
    text = f.read()
    word_count = len(text.split())
    print(f"Total words in corpus: {word_count}")
