# Visually grounded few-shot word learning in low-resource settings: Few-shot MattNet without fine-tuning on background data

This repo provides the source code for the MattNet model trained on few-shot mined pairs without fine-tuned background data, i.e. the model is pretrained and the fine-tuned only on mined pairs. 
The paper which is accepted to TASLP in 2024, is only available [here](https://arxiv.org/abs/2306.11371).

## Important note:

The following instructions should be followed within each of the model folders. For example follow these steps in the directory ```100-shot_5-way```.

## Data

Copy the ```support_set``` folder from [here](https://github.com/LeanneNortje/Mulitmodal_few-shot_word_acquisition.git), in each folder (```5-shot_5-way```, ```5-shot_40-way```, ```10-shot_5-way```, ```50-shot_5-way```, ```100-shot_5-way```). An example directory is:
```bash
├── 50-shot_5-way/
│   ├── support_set
│   │   ├── wavs
│   │   ├── support_set_50.npz
```
If this support set is not used, take note that the ```label_key.npz``` file should be regenerated using the support set repo.
Download the MSCOCO data [here](https://cocodataset.org/#download). We used the 2014 splits, but all the image samples can be taken from the 2017 splits since the few-shot instances are in both. Just replace the ```train_2014``` and ```val_2014``` in the image names with ```train_2017``` and ```val_2017```.
To get forced alignments required for testing, use the [Montreal forced aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/). 
Or simply use the words.txt file in the releases and paste it in the SpokenCOCO folder that has to be downloaded [here](https://groups.csail.mit.edu/sls/downloads/placesaudio/) regardless.


## QbERT

Before the data can be processed, the few-shot mined pairs should be generated.
You can either use the QbERT code we provide in this repo, or yoy can use an updated version that can be found [here](https://github.com/bshall/QbERT.git).
After extracting the repo, your directory structure should look as follows:
```bash
├── few-shot_word_acquisition
│   ├── 5-shot_5-way
│   ├── 5-shot_40-way
│   ├── 10-shot_5-way
│   ├── 50-shot_5-way
│   ├── 100-shot_5-way
│   ├── QbERT
```
Note that if you use the improved QbERT version, the ```few-shot_sampling/sample_audio_query_scores.py``` in each model directory should be adapted.
If you want to use our exact pairs, simply skip this QbERT section and go to the preprocessing step where you can use the already generated ```train_lookup.npz``` and ```val_lookup.npz``` files. 
In the QbERT repo, the following steps should be followed.
```
python encode.py /path/to/SpokenCOCO/wavs encoding_out_dir/ --extension .wav
python segment.py encoding_out_dir/ segments_out_dir/
python encode.py /path/to/support_set/wavs support-set_encoding_out_dir/ --extension .wav
python segment.py support-set_encoding_out_dir/ support-set_segments_out_dir/
```
Hereafter, in each folder, mine few-shot pairs. An example is:
```
cd few-shot_word_acquisition/5-shot_5-way/few-shot_sampling/
python sample_audio_query_scores.py /path/to/QbERT/segments/ /path/to/QbERT/support_set_segments/ /path/to/MSCOCO/ /path/to/SpokenCOCO/
```
To sample image pairs two scripts have to be run in this specific order. The first script ```extract_image_similarities.py``` can take around 5 hours to run, therefore if our exact support set is used, this step can be skipped so that you use our ```image_embeddings.npz`` file generated with this script. Otherwise, execute the following command:

```
python extract_image_similarities.py /path/to/MSCOCO/
```

Regardless of whether the previous command was run, the next step should be run:
```
python sample_image_positives_and_negatives.py /path/to/MSCOCO/
```

To analise the mined pairs, run the following scripts:
```
python audio_pair_analysis.py /path/to/SpokenCOCO
python image_pair_analysis.py /path/to/SpokenCOCO
cd ../
```

## Preprocessing

```
cd preprocessing/
python preprocess_spokencoco_dataset.py
cd ../
```

## Using pretrained model weights

If you want to use the model checkpoints, download the checkpoints given in the release and move the model_metadata folder to the model directory.
Take care to follow the exact directory layout given here:

```bash
├── model_metadata
│   ├── <model_name>
│   │   ├── <model_instance>
│   │   │   ├── models
│   │   │   ├── args.pkl
│   │   │   ├── params.json
│   │   │   ├── training_metadata.json
```
To get this, you can simply extract the appropriate model checkpoint zip file from the releases into the corresponding model folder. For example, extracting the ```100-shot_5-way.zip``` file would result in a folder called ```model_metadata```. Copy this ```model_metadata``` folder into the ``100-shot_5-way``` model folder.
```bash
├── 100-shot_5-way
│   ├── data
│   ├── configs
│   ├── ...
│   ├── model_metadata
│   │   ├── spokencoco_train
│   │   │   ├── <model_name>
│   │   │   │   ├── models
│   │   │   │   ├── args.pkl
│   │   │   │   ├── params.json
│   │   │   │   ├── training_metadata.json
``` 
## Model training

First download the ```pretrained``` weights to initialise the model in the releases and extract it in the project directory as follows:

```bash
├── pretrained
│   ├── best_ckpt.pt
│   ├── last_ckpt.pt
```

Various model parameters can be changed in ```configs/params.json```.

To run a new model:

```
python run.py --image-path /path/to/MSCOCO
```

To resume training:

```
python run.py --resume --image-path /path/to/MSCOCO
```


To resume training from a specific epoch:

```
python run.py --resume --restore-epoch <epoch_you_want_to_restore_from_minus_one> --image-path /path/to/MSCOCO
```
For example, to restore from epoch 8, run:

```
python run.py --resume --restore-epoch 7 --image-path /path/to/MSCOCO
```

## Evaluation
To use the same test episodes we used, download the test episode files from the [support set repo](https://github.com/LeanneNortje/Multimodal_few-shot_word_acquisition/tree/main/data), and paste the files in the model's data folder so that the directory looks like this:

```bash
├── 100-shot_5-way/
│   ├── data
│   │   ├── test_episodes.npz
```

To do few-shot classification:
```
python few-shot_classification.py --image-base /path/to/MSCOCO --audio-base /path/to/SpokenCOCO
```

To do few-shot retrieval:
```
python few-shot_retrieval.py  --image-base /path/to/MSCOCO --audio-base /path/to/SpokenCOCO
```
