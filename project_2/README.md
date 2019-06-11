# NLU19 project_2: Story Cloze Test


## Preparing datasets
Run this file to generate random negative ending in the train set and prepare val and test sets:
```
python create_neg_ending.py
```

## Skip-thought embeddings
Pre-trained skip-thoughts embeddings from Ryankiros
https://github.com/ryankiros/skip-thoughts

- NLTK need to be installed including NLTK data (nltk.download("punkt"))
- Download model files and word embeddings (7 files) and store them in '/skip_thoughts/data`:
- Run "skip_vec.py" to generate Skip_thoughts vectors for the train, val, and test sets:
	```
	python skip_vec.py
	```
	Note: Datasets are stored in .h5 files for further use

## Model

### FFNN using TF DNNClassifer 

For training on train set run: main_train.py
For training on val set run: main_val.py

###  FFNN custom
Based on Tensorflow example notebook:
https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb

For training on train set run: main_train2.py
For training on val set run: main_val2.py


