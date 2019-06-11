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

- Download the following to '/skip_thoughts/data`:
```
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

- Run this file to to generate Skip_thoughts vectors for the train, val, and test sets:
```
python skip_vec.py
```
Note: Datasets are stored in .h5 files for further use


## Model

### FFNN using TF DNNClassifer 

For training on train set run:
```
python main_train.py
```
For training on val set run:
```
python main_val.py
```

###  FFNN custom
Based on Tensorflow example notebook:
https://github.com/soerendip/Tensorflow-binary-classification/blob/master/Tensorflow-binary-classification-model.ipynb

For training on train set run:
```
python main_train2.py
```
For training on val set run:
```
python main_val2.py


