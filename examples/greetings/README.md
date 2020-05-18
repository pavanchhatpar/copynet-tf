# Greetings dataset

## Google Colab link (Recommended)
 - [Running with the package](https://colab.research.google.com/drive/1fztwQ7baqrLhaKGnL4v_p_SjCP0yVESI?usp=sharing) (Recommended method)
 - [Running from source](https://colab.research.google.com/drive/1by8Ob-6JVN9_RSuDFEKAwJYOHIgPnN4f?usp=sharing)


## Refer only when running locally
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
 - Top n sequences are printed for each example as per the beam width set
 - Its probability is seen printed next to it