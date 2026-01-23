Here’s a concrete, end-to-end pipeline that works well for consistent two-column scanned dictionary pages (images), using PaddleOCR + Qwen3-VL.

0) Decide your unit of work
	•	Input: one page image at a time (PNG/JPG).
	•	Output: TSV rows with columns:
Headword	Grammatical_Info	Russian_Definition	Examples/Notes

You’ll extract many entries per page, so your pipeline should produce a list of per-entry “packs”:
	•	entry_crop_image
	•	ordered_lines: [{bbox, text, conf}] for that entry

Then feed each pack to Qwen3-VL to normalise to your 4 TSV fields.

⸻

1) Image preprocessing (cheap, high ROI)

Do these per page:
	1.	Convert to RGB (or grayscale) consistently.
	2.	Deskew (important for column split + line grouping):
	•	estimate skew angle via Hough lines or min-area-rect on binarised text mask
	•	rotate to correct
	3.	Light denoise (median or bilateral).
	4.	Contrast normalisation:
	•	CLAHE often helps small Cyrillic
	5.	Optional: mild sharpening (avoid halos).

Keep preprocessing conservative. Over-binarising can destroy thin serifs and small diacritics.

Artifacts to save for debugging:
	•	preprocessed page image
	•	deskew angle used

⸻

2) Run PaddleOCR (detection + recognition)

Run PaddleOCR in full OCR mode so you get:
	•	bbox/polygon
	•	text
	•	confidence

Recommendations:
	•	Use a model that supports the scripts you have (Latin + Cyrillic).
	•	Keep the raw OCR output; don’t “clean” it yet.

Store OCR results in a structure like:

[
  {"bbox":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text":"...", "conf":0.98},
  ...
]


⸻

3) Normalise boxes and build “line objects”

Paddle can return rotated polygons. Convert each to:
	•	axis-aligned bounding rect: x_min, y_min, x_max, y_max
	•	derived features:
	•	cx = (x_min + x_max)/2
	•	cy = (y_min + y_max)/2
	•	h = y_max - y_min
	•	w = x_max - x_min

Create line objects (even if they’re sometimes word-level, we’ll merge later).

⸻

4) Column segmentation (2-column pages)

Because your pages are consistent, do the simplest reliable thing:

A) Midline split (fastest)
	•	Let page_mid_x = image_width / 2
	•	Column = left if cx < page_mid_x, else right

B) K-means on cx (more robust)
	•	Cluster all cx into 2 clusters
	•	Left cluster = smaller mean cx
	•	Right cluster = larger mean cx

Then, for each column:
	•	sort by y_min ascending, then x_min

Save a debug overlay image with:
	•	left column boxes in one colour
	•	right column boxes in another
(You’ll catch 90% of issues here.)

⸻

5) Merge into true text lines (if needed)

If Paddle output is already line-level, you can skip this. If it’s word-level or fragmented:

Within each column:
	1.	Compute a typical line height H (median of h)
	2.	Group boxes into lines by y-overlap:
	•	two boxes belong to same line if their vertical centres differ by less than ~0.5*H (tune)
	3.	Within each merged line, sort by x_min and concatenate texts with spaces.

Now you have a clean sequence of lines[] per column:

{"bbox": [x_min,y_min,x_max,y_max], "text":"...", "conf":...}


⸻

6) Entry segmentation within a column

This is the key step for dictionary pages.

6.1 Compute per-line features

For each line i:
	•	indent = x_min - col_left_margin
	•	gap = y_min(i) - y_max(i-1)
	•	is_left_aligned = indent < indent_threshold (threshold based on distribution)
	•	optional token cues from text:
	•	contains grammatical marker patterns like (...)
	•	begins with a likely headword pattern (often short, maybe followed by comma/semicolon/parentheses)

6.2 Choose a simple, stable “new entry” rule

Start with this hybrid rule (works well on consistent layouts):

A line starts a new entry if any of:
	1.	gap is large (e.g., gap > median_gap + k * mad_gap)
	2.	is_left_aligned AND previous line was not left aligned
	3.	line text matches a “headword start” heuristic (tune per dictionary), for example:
	•	starts with letters and soon has a grammatical parenthetical: ^[A-Za-z][^\\n]{0,30}\\(.+\\)
	•	or starts with a short token then comma: ^[A-Za-z-]{2,20},

Then:
	•	begin a new entry at that line
	•	continue adding subsequent lines until the next “new entry” line.

6.3 Keep “carry-over” and “broken entry” handling

Some entries wrap awkwardly. Add guardrails:
	•	If an “entry” is only 1 line and looks like a continuation (high indent, starts with punctuation), merge it with previous.
	•	If an entry has no plausible headword line but has definition-style text, it might be a continuation: merge back.

⸻

7) Build entry crops (image snippets)

For each entry (a set of lines):
	1.	Union their bboxes into an entry bbox:
	•	x_min = min(line.x_min), etc.
	2.	Expand with padding (important):
	•	left/right: 10–30 px
	•	top/bottom: 10–30 px
	3.	Crop the preprocessed page image to that bbox.
	4.	Save crop (or keep in memory) with an entry_id.

Also keep metadata:
	•	page number / filename
	•	column id (L/R)
	•	entry bbox on page
	•	list of line bboxes + OCR texts

⸻

8) Create the VLM input payload per entry

For each entry, build:

8.1 “Lines JSON”

Ordered list:

{
  "page_id":"...",
  "column":"L",
  "entry_bbox":[x1,y1,x2,y2],
  "lines":[
    {"bbox":[...],"text":"...","conf":0.98},
    ...
  ]
}

8.2 Entry crop image

Pass the crop image as the visual input.

Why both:
	•	OCR text gives Qwen a strong prior and helps it be fast/consistent
	•	Image lets it correct OCR mistakes and disambiguate punctuation/letters

⸻

9) Prompt Qwen3-VL to output strict TSV fields

You want deterministic output. Use a tight instruction like this:

System / instruction style:
	•	“You are extracting dictionary entries into a TSV schema.”

User content:
	•	provide the JSON (lines) and the image crop
	•	ask for a strict JSON response (easier to parse than TSV), then you convert to TSV yourself.

Example prompt (conceptual):
	•	Output only JSON with keys:
	•	Headword
	•	Grammatical_Info
	•	Russian_Definition
	•	Examples_Notes
	•	Use empty string "" if missing.
	•	Do not include extra keys.
	•	Do not add commentary.

Also:
	•	If the entry contains multiple senses, keep them inside Russian_Definition as a single string with separators (e.g., 1) ...; 2) ...)
	•	Put cross-references (см. etc.) into Examples_Notes unless they are the definition.

⸻

10) Post-process: JSON → TSV

For each entry JSON:
	•	normalise whitespace
	•	replace tabs/newlines inside fields with spaces (or escape them)
	•	write one TSV row per entry

Keep an entry_id in your internal data, even if you don’t put it in TSV, so you can trace errors back to:
	•	crop image
	•	OCR lines
	•	page location

⸻

11) QA loop (you will need this)

Automate checks:
	•	Headword empty but other fields present → flag
	•	Very low OCR confidence overall → flag (maybe re-OCR with different settings or re-crop)
	•	Duplicate headword+graminfo repeated many times on one page → possible segmentation error
	•	Entry crop too tall (spanning multiple entries) → segmentation error

Create a simple review artefact:
	•	a page image with entry boxes drawn + entry ids
	•	a CSV/TSV with entry_id and extracted fields
This makes fixing heuristics fast.

⸻

12) Tuning strategy (minimum pain)

Tune in this order:
	1.	Deskew reliability
	2.	Column split (almost always easy)
	3.	Line merging (if needed)
	4.	Entry boundary heuristics (this is where most time goes)
	5.	Qwen prompt strictness and post-processing

⸻

If you want, I can give you:
	•	a concrete set of heuristics for “headword start” based on what your dictionary headwords look like (Latin only? hyphens? commas? italics markers?), and
	•	a compact JSON schema + prompt that tends to produce stable outputs from Qwen3-VL without drift.			