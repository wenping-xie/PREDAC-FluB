## # PREDAC-FluB

## Prerequisites

    tensorflow==2.12.0
    python==3.10
    pandas==2.3.0
    Levenshtein==0.27.1
    numpy==1.23.5
    umap-learn==.5.1
    scikit-learn==1.5.2

## Getting started

1 Please prepare the input file. It is required that the input file contains 'new_name_1','new_name_2','seq_1','seq_2','year_1','year_2' and 'label' for training data and testing data or 'new_name_1','new_name_2','seq_1','seq_2','year_1','year_2' for prediction.<br>

Example of the training data and testing data:

![image-20251028164211849](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20251028164211849.png)



Example of the input data for prediction:

![image-20251028164119626](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20251028164119626.png)

2 Run matrix_generate.py to generate the input matrix from the input file (train_data,test_data or predict_data):

###First use esm2 to generate the coding file for each protein sequence, and then further generate the feature file!

```shell
#bash
esm-extract esm2_t6_8M_UR50D BV_HA1_forhi_sequences.fasta ./esm_seq --repr_layers 0 5 6 --include per_tok
```

```python
python3 matrix-generatre-ems2-7-features.py 
--subtype BV（or BY) 
--seq_dir /path/input_data-v1 
--type training 
--dir /path/save_dir/esm2-and-7-features 
--thread 20

```

3 Run train.py to train a CNN model:

```python
for i in {1..5}; do
  python3 /path/train.py 
  --aaindexfile /path/generate-feature/BV.csv.npy 
  --fold ${i} 
  --outdir /path/save_dir
  --type BV（or BY） 
  --number_columns 654
done


```

4 Run predict.py to predict the antigenic variants of the two sequences:

```python
       for i in 1 2 3 4 5
       do
            python3 predict.py 
            --test_data /path/test_data_BV.npy 
            --folddir /path/fold/ 
            --outdir /home/xie_wenping/influenza_B/predict/BV/train_validate_test/esm2-and-7-features/4-5fold_mode_pred_test/lr-e3-time-train-long \
            --type BV(or BY)
            --number_columns  654
            --fold ${i} 
        done

```

5 #UMAP-assisted K-means Clustering for for Antigenic Characterization,this section details the methodology employed for antigenic clustering, combining UMAP for dimensionality reduction and K-means for cluster assignment. 

```python
###The input here uses the probability value of antigenic similarity between two strains when the model predicts it.
python3 UMAP-and-kmeans.py 
```
Xie W, Liu J, Wang C, Wang J, Han W, Peng Y, Du X, Meng J, Ning K, Jiang T. PREDAC-FluB: predicting antigenic clusters of seasonal influenza B viruses with protein language model embedding based convolutional neural network. Brief Bioinform. 2025 Jul 2;26(4):bbaf308. doi: 10.1093/bib/bbaf308. PMID: 40665740; PMCID: PMC12264208.
