# Handwritten Museum Collections Records — OCR → Structured Data → Data Analysis and Insights (Google Colab)

This tutorial shows you how to convert *handwritten museum collection records* (scans or PDFs) into structured data and then carry out **data analysis and insights**. It is designed for **scholars, museum professionals, graduate students, and advanced undergraduates**. You do **not** need prior coding experience. We proceed carefully and conversationally.

By the end you will:
- Run a notebook in Google Colab (in your browser—no local installs).
- Upload scans or PDFs.
- Convert PDFs to page images, enhance them, and run OCR.
- Use **two OCR engines** (EasyOCR and TrOCR) to handle different handwriting challenges.
- Extract structured fields (Accession Number, Object Name, Provenance, etc.).
- Export your dataset to CSV and JSON.
- Perform **basic data analysis and insights** with tables and charts.

---

## Table of Contents
1. [ What you’ll need ](#what-youll-need)  
2. [ Why Python ](#why-python)  
3. [ What is a Python Notebook? ](#what-is-a-python-notebook)  
4. [ What you’ll learn ](#what-youll-learn)  
5. [ Ethics, rights, and responsible handling ](#ethics-rights-and-responsible-handling)  
6. [ Meet Google Colab ](#meet-google-colab)  
7. [Install the tools we’ll use](#1-install-the-tools-well-use)  
8. [Import (load) the libraries](#2-import-load-the-libraries)  
9. [Upload your scans or PDFs](#3-upload-your-scans-or-pdfs)  
10. [Convert PDFs to images](#4-convert-pdfs-to-images)  
11. [Preprocess (lightly clean up) the images](#5-preprocess-lightly-clean-up-the-images)  
12. [Run OCR two ways: EasyOCR and TrOCR](#6-run-ocr-two-ways-easyocr-and-trocr)  
13. [Turn raw text into fields](#7-turn-raw-text-into-fields)  
14. [Flag items for human review](#8-flag-items-for-human-review)  
15. [Clean, standardize, and export](#9-clean-standardize-and-export)  
16. [Data analysis and insights](#10-data-analysis-and-insights)  
17. [ Troubleshooting ](#troubleshooting)  
18. [ Contributing to this tutorial ](#contributing-to-this-tutorial)  
19. [ License & Citation ](#license--citation)  
20. [ Glossary (plain-English definitions) ](#glossary-plainenglish-definitions)



## What you’ll need

Before you start, make sure you have a Google account (for Colab), a stable internet connection, and a few scanned or PDF copies of handwritten collection records that you’re allowed to experiment with. You don’t need anything fancy—just a few pages are enough for practice. The whole process should take about half an hour the first time you try it.

The best part: you don’t need to install Python or any special software on your computer. Everything runs in your browser.



## Why Python

Think of Python as a versatile digital toolkit 🛠️ for your research. It's not just a single tool but a collection of specialized gadgets that let us handle many different tasks, from converting documents and analyzing text to creating charts. The best part? It's designed to be readable, almost like plain English, which makes it easier for us to learn and use.



## What is a Python Notebook?

Imagine a digital lab notebook 🧪 where you can write down your notes, thoughts, and explanations right next to the code you're running. That's a Python notebook. It lets us mix text (like the instructions you're reading now), live code that we can run, and the results from that code (like a clean table or a cool chart). It's a perfect way to tell a story with our data, showing our work step-by-step.



## What you’ll learn

By the end of this tutorial, you’ll be able to take scans or PDFs of handwritten records, process them with OCR, organize the results into structured fields, and then perform data analysis on the dataset you created. Along the way, you’ll also learn how to:

- Work responsibly with sensitive cultural data.  
- Use Google Colab to run Python code in the cloud.  
- Preprocess images to improve OCR accuracy.  
- Compare two different OCR approaches and understand their trade-offs.  
- Export your data into formats like CSV and JSON that can be used in spreadsheets, web applications, or archives.  
- Carry out basic **data analysis and insights** such as counting object types, looking at collection dates, and spotting gaps in the records.



## Ethics, rights, and responsible handling

Before we dive in, it’s important to pause and consider the ethics of digitizing and analyzing museum records. These documents sometimes contain personal names, addresses, or sensitive site information. They may also involve cultural knowledge that communities prefer to control.

Ask yourself: do I have permission to digitize and analyze this material? Are there privacy concerns? How will I store and share the data? And perhaps most importantly—who benefits from this work? Being clear about these issues from the start is just as important as learning the technical steps.



## Meet Google Colab

Google Colab is where we’ll run all of our work. It’s a free, cloud-based environment for Python notebooks, hosted by Google. You don’t need to install anything—if you can open a browser, you can use Colab.

Colab is also collaborative, like a Google Doc: you can share a notebook with a colleague, and they can rerun or modify it. For heavier lifting, Colab can even give you free access to GPUs (graphics processors), which makes OCR models like TrOCR run much faster.

To try it out, go to [Google Colab](https://colab.research.google.com), click **New Notebook**, and in the first cell type:

```python
print("Colab runtime ready. Run cells with Shift+Enter.")
```

Press **Shift+Enter** and you should see the message appear. If it does, your environment is ready to go.



## 1. Install the tools we’ll use

**What:** Install specialized tools (think “apps for Python”).  
**Why:** OCR, image conversion, and analysis aren’t built into base Python.  
**Validate:** Install finishes with no red error messages.

**Tools and why we need them:**
- **`poppler-utils`** (system tool): Converts PDF pages into images. OCR engines read images, not PDFs.  
- **`pdf2image`** (Python): The bridge that lets Python use Poppler easily.  
- **`pillow`**: The main image library in Python (open, save, grayscale, sharpen, etc.).  
- **`easyocr`**: OCR engine #1—fast, multi-language, line-by-line results with **confidence scores**.  
- **`transformers`**: Modern ML models; we’ll use it for **TrOCR**, which is trained on handwriting.  
- **`timm`**: A helper library required by TrOCR.

**Why not just one library?**  
No single library does all steps reliably. This is a toolkit: each tool is specialized, like having both a screwdriver and a hammer.

**Run this in Colab:**
```python
!apt-get -q update
!apt-get -q install -y poppler-utils
!pip -q install easyocr pillow pdf2image transformers==4.43.3 timm==0.9.16
print("Install complete.")
```



## 2. Import (load) the libraries

**What:** Tell Python which libraries you’ll use.  
**Why:** Installation puts them on the system; `import` loads them into your notebook.  
**Validate:** You see “Libraries imported.” and no red errors.

```python
import os, re, io, time, string
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

import easyocr
from pdf2image import convert_from_path
from transformers import pipeline

print("Libraries imported.")
```



## 3. Upload your scans or PDFs

**What:** Upload files from your computer to Colab’s working space.  
**Why:** Colab runs on Google’s servers and can’t see your laptop until you upload.  
**Validate:** The output lists the filenames you selected.

```python
from google.colab import files
uploaded = files.upload()  # select JPG/PNG and/or PDF files
list(uploaded.keys())
```

> Tip: Click the **folder icon** on the left side of Colab to browse files you’ve uploaded.



## 4. Convert PDFs to images

**What:** Convert multi-page PDFs to per-page PNG images (≈300 DPI).  
**Why:** OCR engines operate on images, not PDFs.  
**Validate:** You see a list of new `_pageN.png` image files.

**Why both Poppler and pdf2image?**  
- Poppler is the **engine** that splits PDFs.  
- pdf2image is the **Python bridge** that makes Poppler easy to use in Colab.  
- If you already uploaded images, this step will just pass those through.

```python
def pdfs_to_images(paths, dpi=300):
    out = []
    for p in paths:
        if p.lower().endswith(".pdf"):
            pages = convert_from_path(p, dpi=dpi)
            for i, img in enumerate(pages, 1):
                op = f"{os.path.splitext(p)[0]}_page{i}.png"
                img.save(op, "PNG")
                out.append(op)
        else:
            out.append(p)  # not a PDF, assume it's already an image
    return out

image_paths = pdfs_to_images(list(uploaded.keys()), dpi=300)
print("Images ready:", image_paths[:10], "... total:", len(image_paths))
```



## 5. Preprocess (lightly clean up) the images

**What:** Improve legibility before OCR: grayscale → contrast → sharpen.  
**Why:** Handwriting is often faint or uneven; preprocessing reduces noise and clarifies strokes.  
**Validate:** When you preview a processed image, handwriting should look a little darker and crisper—not over-sharpened.

```python
def preprocess(img: Image.Image) -> Image.Image:
    # 1) Grayscale: focus on text shapes
    g = ImageOps.grayscale(img)
    # 2) Contrast: darken faint handwriting slightly
    g = ImageOps.autocontrast(g, cutoff=2)
    # 3) Sharpen: small unsharp mask to help OCR detect edges
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return g

proc_paths = []
for p in image_paths:
    try:
        out = os.path.splitext(p)[0] + "_proc.png"
        preprocess(Image.open(p)).save(out, "PNG")
        proc_paths.append(out)
    except Exception as e:
        print("Skipped:", p, e)

print("Processed images:", len(proc_paths))
from IPython.display import display
if proc_paths:
    display(Image.open(proc_paths[0]).resize((600, None)))
```

> If the result looks over-processed (haloing or artifacts), reduce sharpening or skip it.



## 6. Run OCR two ways: EasyOCR and TrOCR

We're going to use two different OCR "engines" because handwriting can be tricky! ✍️ EasyOCR is a great all-around tool, a bit like a sturdy shovel for a lot of different jobs. But sometimes, for really messy or unique handwriting, we'll use a more specialized tool called TrOCR. Think of it as a finely tuned spade for those more difficult spots. This two-tool approach gives us the best chance of getting good results from all sorts of records.

**What:** Extract machine-readable text from images.  
**Why two engines?** Because they excel at different things.
- **EasyOCR**: fast, line-by-line text with **confidence scores** (0–1). Great for triage and quick baselines.  
- **TrOCR**: a modern **transformer** model trained for handwriting; slower but often more accurate on cursive or messy text.

**Why not just one?**  
- EasyOCR is quick and transparent (line confidence) but can struggle with cursive.  
- TrOCR handles handwriting better but doesn’t give per-line confidence and is slower.  
Using both gives you **speed + confidence** and **accuracy**.

### 6A) EasyOCR (baseline & confidence)
**What:** Recognize text per line; get a confidence score.  
**Why:** Confidence helps you build a **review queue** later.  
**Validate:** You see a small sample of (bbox, text, confidence) tuples.

```python
# 'en' = English; add more languages if needed, e.g., ['en','fr']
reader = easyocr.Reader(['en'], gpu=True)  # set gpu=False if GPU not available

def easyocr_page(path: str):
    # returns list of dicts: text, confidence (0..1), and bbox
    results = reader.readtext(path, detail=1)
    lines = []
    for bbox, text, conf in results:
        lines.append({"engine":"easyocr", "text":text, "confidence":float(conf), "bbox":bbox})
    return lines

easyocr_results = {p: easyocr_page(p) for p in proc_paths}
sample_img = proc_paths[0] if proc_paths else None
pd.DataFrame(easyocr_results[sample_img]).head(10) if sample_img else "No images"
```

### 6B) TrOCR (transformer handwriting model)
**What:** Page-level transcription using a model tuned for handwriting.  
**Why:** Often better than traditional OCR on cursive or degraded scans.  
**Validate:** You see a block of recognized text for a page.

```python
trocr = pipeline(task="image-to-text", model="microsoft/trocr-base-handwritten", device_map="auto")

def trocr_page(path: str) -> str:
    img = Image.open(path).convert("RGB")
    out = trocr(img)
    return out[0].get("generated_text", "") if out else ""

trocr_results = {p: trocr_page(p) for p in proc_paths}
print((trocr_results[sample_img] or "")[:400] if sample_img else "No images")
```



## 7. Turn raw text into fields

**What:** Convert free text into **structured fields** (e.g., Accession Number, Object Name, Maker, Culture, Site/Location, Provenance, Materials, Dimensions, Date, Notes).  
**Why:** Structured data can be searched, validated, analyzed, and integrated with collection systems.  
**Validate:** A DataFrame with columns corresponding to your fields.

We’ll parse **headings** commonly found on record cards using *regular expressions* (regex). **Edit these patterns** to match your institution’s forms.

```python
# Choose which OCR output to parse: "trocr" (page text) or "easyocr" (line by line)
USE_ENGINE = "trocr"  # or "easyocr"

FIELD_PATTERNS = {
    "accession_number": r"^(accession|acc\.?\s*no\.?|catalog\s*no\.?|cat\.?\s*no\.?)[\s:]*([A-Za-z0-9\-./]+)",
    "object_name":      r"^(object\s*name|artifact|item)[:\s]+(.+)",
    "creator_maker":    r"^(maker|artist|creator)[:\s]+(.+)",
    "culture":          r"^(culture|cultural\s*group)[:\s]+(.+)",
    "site_location":    r"^(site|find\s*spot|location)[:\s]+(.+)",
    "provenance":       r"^(provenance|provenience)[:\s]+(.+)",
    "materials":        r"^(material|materials|medium)[:\s]+(.+)",
    "dimensions":       r"^(dimension|dimensions|size)[:\s]+(.+)",
    "date":             r"^(date|dated)[:\s]+(.+)",
    "notes":            r"^(notes?|remarks?)[:\s]+(.+)",
}

def normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\t\u00A0]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def extract_fields_from_lines(lines):
    """Given a list of lines, return a dict of fields captured by patterns."""
    fields = {}
    for raw in lines:
        line = normalize_line(raw)
        for key, pat in FIELD_PATTERNS.items():
            m = re.match(pat, line, flags=re.IGNORECASE)
            if m:
                fields[key] = m.group(m.lastindex).strip(" .;:")
    return fields

records = []
for img_path in proc_paths:
    if USE_ENGINE == "easyocr":
        lines = [d["text"] for d in easyocr_results.get(img_path, [])]
        full_text = "\n".join(lines)
    else:
        full_text = trocr_results.get(img_path, "")
        lines = [l for l in full_text.splitlines() if l.strip()]

    fields = extract_fields_from_lines(lines)
    records.append({"source_image": os.path.basename(img_path), "full_text": full_text, **fields})

df = pd.DataFrame(records)
df.head()
```

> **Tip:** If your headings are different (e.g., “Specimen No.”), adjust `FIELD_PATTERNS` to match your exact forms.



## 8. Flag items for human review

**What:** Automatically flag likely problem cases (e.g., missing Accession Number or very short pages).  
**Why:** Handwriting OCR is imperfect; human-in-the-loop quality control is essential.  
**Validate:** A `needs_review` column and a `records_review_queue.csv` file are created.

```python
def needs_review(row: pd.Series) -> bool:
    # Simple heuristics—edit to suit your workflow
    if not row.get("accession_number"):
        return True
    if isinstance(row.get("full_text"), str) and len(row["full_text"]) < 40:
        return True
    return False

df["needs_review"] = df.apply(needs_review, axis=1)
print("Rows needing review:", df["needs_review"].sum(), "of", len(df))

review_df = df[df["needs_review"]].copy()
review_df.to_csv("records_review_queue.csv", index=False)
"Created records_review_queue.csv"
```



## 9. Clean, standardize, and export

**What:** Tidy whitespace and case; export the dataset.  
**Why both CSV and JSON?**  
- **CSV:** Spreadsheet-friendly (Excel, Access, most CMS imports).  
- **JSON:** Structured for web apps, APIs, and archival packaging.

**Validate:** Files appear in the Colab file browser and download correctly.

```python
def tidy(s):
    if pd.isna(s): return s
    s = str(s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

for col in ["accession_number","object_name","creator_maker","culture",
            "site_location","provenance","materials","dimensions","date","notes"]:
    if col in df.columns:
        df[col] = df[col].apply(tidy)

# Light normalization example
if "object_name" in df.columns:
    df["object_name"] = df["object_name"].str.strip().str.title()

df.to_csv("museum_handwritten_records.csv", index=False)
df.to_json("museum_handwritten_records.json", orient="records", indent=2, force_ascii=False)
"Saved museum_handwritten_records.csv and museum_handwritten_records.json"
```

> **Optional (save to Google Drive):**
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> df.to_csv('/content/drive/MyDrive/museum_handwritten_records.csv', index=False)
> ```



## 10. Data analysis and insights

**What:** Transform your structured data into understanding.  
**Why:** OCR is the beginning; the goal is **data analysis and insights**—what your records *say* about the collection.  
**Validate:** You see tables and charts (bar, line), and they match your expectations.

We’ll do four complementary views:

### 10.1 Frequency analysis (materials, object names, cultures)
**What:** Count common values to see collection composition.  
**Why:** Reveals focus areas, cataloging patterns, potential biases.  
**Validate:** Tables and horizontal bar charts for top values.

```python
import matplotlib.pyplot as plt

def top_counts(series, n=10, title="Top values"):
    vc = series.dropna().astype(str).str.strip().value_counts().head(n)
    display(vc)
    plt.figure()
    vc.sort_values().plot(kind="barh", edgecolor="black")
    plt.title(title); plt.xlabel("Count"); plt.ylabel(series.name or "Value")
    plt.tight_layout(); plt.show()

for col, label in [("materials","Top 10 Materials"),
                   ("object_name","Top 10 Object Names"),
                   ("culture","Top 10 Cultures")]:
    if col in df.columns:
        top_counts(df[col], n=10, title=label)
```

### 10.2 Temporal analysis (objects by year)
**What:** Extract a 4-digit year from `date` and count records per year.  
**Why:** Shows collecting, creation, or cataloging trends (clarify what “date” represents).  
**Validate:** A line chart of counts by year.

```python
if "date" in df.columns:
    df["year"] = df["date"].astype(str).str.extract(r"(\d{4})")
    year_counts = df["year"].value_counts().sort_index()
    display(year_counts.tail(15))  # peek at recent years
    plt.figure()
    year_counts.plot(kind="line", marker="o")
    plt.title("Objects by Year")
    plt.xlabel("Year"); plt.ylabel("Count")
    plt.tight_layout(); plt.show()
```

> **Note:** If “date” mixes “created/acquired/cataloged,” you’ll refine this later.

### 10.3 Provenance signals (quick word frequencies)
**What:** Count frequent words in `provenance`, `site_location`, and `notes`.  
**Why:** Surfaces recurring places/agents to investigate further (authority control, mapping).  
**Validate:** A small table and bar chart of top words.

```python
from collections import Counter
import string

def tokenize(s):
    s = str(s).lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in s.split() if len(w) > 2]

stop = set("the and for with from into over under out are was were this that have has had not but of in on to by".split())

texts = []
for col in ["provenance","site_location","notes"]:
    if col in df.columns:
        texts.extend(df[col].dropna().astype(str).tolist())

tokens = []
for t in texts:
    tokens.extend([w for w in tokenize(t) if w not in stop])

common = pd.DataFrame(Counter(tokens).most_common(20), columns=["word","count"])
display(common)

plt.figure()
common.set_index("word")["count"].plot(kind="bar", edgecolor="black")
plt.title("Common Words in Provenance / Location / Notes")
plt.xlabel("Word"); plt.ylabel("Count")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()
```

*(Optional) Word cloud for a quick visual feel:*
```python
try:
    import wordcloud  # noqa
except ImportError:
    !pip -q install wordcloud
from wordcloud import WordCloud

text_blob = " ".join(tokens)
wc = WordCloud(width=900, height=400).generate(text_blob)
plt.figure(figsize=(9,4)); plt.imshow(wc); plt.axis("off"); plt.title("Provenance Word Cloud"); plt.show()
```

### 10.4 Missing data (where are the gaps?)
**What:** Visualize fields with lots of missing values.  
**Why:** Guides cleanup priorities and prevents biased conclusions.  
**Validate:** A horizontal bar chart of missing counts by field.

```python
missing = df.isna().sum().sort_values(ascending=True)
plt.figure()
missing.plot(kind="barh", edgecolor="black")
plt.title("Missing Values by Field")
plt.xlabel("Missing count"); plt.tight_layout(); plt.show()
```

> **Narrative tip:** Combine these views into a clear story (e.g., “Most cataloged items are ceramic vessels (materials), concentrated in 1950–1970 (years), with sparse provenance detail (missing data).”).



## Troubleshooting

Things don’t always go smoothly, especially when working with OCR and scanned records. Here are some common issues and quick fixes:

- **“No module named …” or install errors**  
  Re-run the install cell in **Step 1**, then the import cell in **Step 2**.  

- **PDF conversion fails**  
  Make sure `poppler-utils` installed correctly. Try lowering the DPI (e.g., 200). Confirm your PDFs aren’t encrypted. (See **Step 4**.)  

- **OCR gibberish**  
  Preview the processed images to check legibility. Adjust or skip sharpening. If EasyOCR fails, try TrOCR, and vice versa. For EasyOCR, add extra language codes if your records aren’t in English (e.g., `['en','fr']`). (See **Step 6**.)  

- **Empty plots or tables**  
  Use `df.head()` and `df.columns` to confirm that the field actually contains data. Some charts won’t show anything if the column is missing or empty.  

- **Slow runtime**  
  In Colab, switch to GPU: **Runtime → Change runtime type → Hardware accelerator → GPU**. Then re-run the import step. TrOCR especially benefits from GPU acceleration.  



## Contributing to this tutorial

We welcome improvements. If you want to tweak the tutorial, fix bugs, or add examples:  

1. **Fork** this repository.  
2. **Clone** your fork (or open in GitHub Codespaces).  
3. **Make your changes**: README, patterns, notebooks, examples.  
4. **Commit and push** to your fork.  
5. Open a **Pull Request** back to this repo—explain what changed and why.  



## License & Citation

- **Code:** Apache-2.0  
- **Docs:** CC BY 4.0  

If you use or adapt this tutorial, please **cite the repository**. Include the provided `CITATION.cff` so GitHub can generate APA/BibTeX automatically (look for **“Cite this repository”** on the repo page).  



## Beginner Friendly Glossary

- **Notebook:** An interactive document with text, code cells, and outputs.  
- **Cell:** A single block you run; it shows results underneath.  
- **Library / package:** Reusable code that adds features (like OCR).  
- **Install vs. import:** Install puts a library on the system; import loads it into the notebook.  
- **OCR (Optical Character Recognition):** Recognizes text in images.  
- **EasyOCR:** A fast OCR engine with confidence scores per line.  
- **TrOCR:** A transformer-based OCR model tuned for handwriting.  
- **Regex (regular expression):** A pattern for matching text like “Accession No: #####”.  
- **DataFrame:** A table of data in Python (rows/columns), like a spreadsheet.  
- **CSV / JSON:** Common file formats for tabular and structured data.  
- **DPI:** Dots per inch—image resolution.  
- **GPU:** Graphics processor; speeds up some models (optional in Colab).  
