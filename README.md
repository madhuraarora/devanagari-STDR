# Devanagari Scene Text Recognition Using Synthetic Data and Progressive CRNN Training

This repository contains the implementation of a three-phase training pipeline for Devanagari Scene Text Recognition (STDR).  
The approach integrates script-aware synthetic data generation using the HarfBuzz text-shaping engine and a progressive CRNN training curriculum.  
The final model achieves a Character Error Rate (CER) of **0.3391** on the IIIT-ILST cropped images dataset for Devanagari script.

---

## Training Phases

### Phase 0 Grapheme-Level Pretraining
Training on isolated synthetic graphemes (~100 classes) to learn basic shapes.

### Phase 1 Synthetic Word-Level Training
Training on ~275,000 synthetic word images rendered with HarfBuzz using ~40 Devanagari fonts and multiple scene-like augmentations.

### Phase 2 Real Scene Fine-Tuning
Fine-tuning on the IIIT-ISTR dataset to adapt to real-scene noise, background variation, and distortions.

---

## Results

| Training Phase | Dataset               | CER    | WER    |
|----------------|-----------------------|--------|--------|
| Phase 0        | Synthetic Graphemes   | 1.1576 | 1.0000 |
| Phase 1        | Synthetic Words       | 0.8557 | 0.9783 |
| Phase 2        | IIIT-ISTR             | 0.3391 | 0.6374 |

---

# Credits and Attributions

The following resources were used in accordance with their respective licenses.

---

## Datasets

### IIIT-ILST (Indic Scene Text Dataset)
Used for evaluation.  
Developed by CVIT, IIIT Hyderabad.  
Website: https://cvit.iiit.ac.in/usodi/istr.php

**Citation:**  
Mathew et al., *“Benchmarking Scene Text Recognition in Devanagari, Telugu and Malayalam,”* arXiv:2104.04437, 2021.

---

### IIIT-ISTR (Indic Scene Text Recognition Dataset)
Used for real-scene fine-tuning.  
Developed by CVIT, IIIT Hyderabad.

**Citation:**  
IIIT-Hyderabad, *“IIIT-ISTR: Indic Scene Text Recognition Dataset,”* 2025.

---

### Vocabulary Corpus (~11,300 words)
Used for synthetic word generation.  
Sourced from publicly available Devanagari lexical datasets (Hindi/Marathi/Sanskrit).  
All rights remain with original dataset creators.

---

## Fonts

Multiple Devanagari fonts were used for synthetic data generation.  
Fonts are licensed under the SIL Open Font License (OFL) unless specified otherwise.

---

## HarfBuzz License Notice

This project uses the HarfBuzz OpenType text shaping engine.

HarfBuzz — OpenType Text Shaping Engine
Copyright © 2010–2025 The HarfBuzz Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.


Website: https://harfbuzz.github.io/

---

## License

This repository is released for academic and research purposes.  
The code in this repository is not affiliated with or endorsed by the authors of any dataset used.
