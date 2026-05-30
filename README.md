# BotMoE Reposiotry

This is the official repo for [Social-Media-Bot-Detection-by-Fusing-Multimodal-Information-with-Large-Language-Models]

## Step 1 Create Environment

```
    pip install -r requirements.txt
```

## Step 2 Download dataset

For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).

For TwiBot-22, visit the [TwiBot-22 github repository](https://github.com/LuoUndergradXJTU/TwiBot-22).

For other datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

We also provide the processed files for each dataset, they are available at [Google Drive](https://drive.google.com/drive/folders/1MupKMFU26FnwWCMJUP5BjiAeGjwIF1YU?usp=sharing). The Twibot-22 dataset in this link is the sampled dataset. 

## Step 4 Train the LM
Remember to change the parameters in train_LM/main.py and put the dataset in the corresponding location. For each dataset in [Google Drive](https://drive.google.com/drive/folders/1MupKMFU26FnwWCMJUP5BjiAeGjwIF1YU?usp=sharing), we also provide results for the trained LM outputs for different LLMs, which names "embeddings_iter_xxx.pt".

```
    # load data for LM(train_LM/main.py)
    data = load_raw_data(dataset='',  # setting dataset path
                         text_path='Shb_prompt.json',  # setting text path
                         label_path='label.pt'  # setting label path
                         )

    # set roberta path(train_LM/LM.py)
    self.LM = AutoModel.from_pretrained('')   # setting roberta path
```

Then you can start training the LM!
```
    python main.py
```

## Step 5 Train and Test the Model

Remember to modify the dataset path in the train.py file and Dataset.py file for each dataset and put the dataset in the corresponding location.

Then you can start training!
```
    python train.py
```

