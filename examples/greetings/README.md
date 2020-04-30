## Greetings dataset

### Pre-requisites
You should have the environment already setup as detailed in the main README

### Download data
```bash
python download.py
```

### Preprocess
```bash
python preprocess.py
```

### Training & Testing
Refer `train.ipynb`

#### Output format
 - Top 3 sequences are printed for each example as beam width is set to 3
 - Its log probability is seen printed next to it