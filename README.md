# PPCap

It's still not runnable; we're still making modifications. Just learned how to use GitHub.

The code for applying PPCap to PureT has been uploaded to [GitHub](https://github.com/gWeiXP/PPCapPureT). We will be uploading the data and trained models soon. 

See `'appendix.pdf'` for the revised appendix.

# Generating from models used in paper
Run `test_guide.py` to generate stylized captions. Since the code is modified based on CLIPCap and GeDi, there are many unused parameters. If you find it troublesome, you can go and see [the code for applying PPCap to PureT](https://github.com/gWeiXP/PPCapPureT), which has been tidied up and is now very clean and concise.


## Important arguments
Important arguments include:
* `--gen_model_name_or_path`, the path of pre-trained factual model
* `--base_model_type`, the type of factual model
* `--gedi_model_name_or_path`, the path of trained stylized model
* `--code_1`, the desired style
* `--code_0`, the undesired style
* `--disc_weight`, the weight w
* `--testdata`, the path of test set
* `--teststyle`, the desired style

Apologies, we noticed that there are some absolute paths in the code that need to be modified to match the file paths on your device.
* The paths like "/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_ro" in `utils.py eval_ppl()` : we have uploaded the `PPL` folder to this project.
* The paths like "/home/liwc/wxp/dataset/MSCOCO/train2014/" in `utils.py ClipCocoDataset()` : you may need to modify the code so that `filename` obtains the paths of images on your device.
* The paths ""/home/liwc/wxp/refercode/DataTestProcess/bert-base-uncased/vocab.txt"" in `utils.py eval_acc()` : we have uploaded the `bert-base-uncased` folder to this project.

## Datasets and trained models
* Download MSCOCO [validation images](http://images.cocodataset.org/zips/val2014.zip)
* Download the processed dataset from [Baidu Netdisk](https://pan.baidu.com/s/1NLn9wOwK6GajkDbdCfZVIw). the extracted code is 'zoz4'. 
* Download the trained classfier from [Baidu Netdisk](https://pan.baidu.com/s/1anksxmotMBsipLjeg1EvRg). the extracted code is 'eh91'. It is used to evaluate the cls. The 'classifier' folder needs to be placed in the current directory.
* Download the factual model from [Baidu Netdisk](https://pan.baidu.com/s/19yydiOBrLWp54SV1_-2w6Q). the extracted code is 'i55y'. 
* Download the stylized model from [Baidu Netdisk](https://pan.baidu.com/s/1fSmY-ypdjaoeOs7Knh0plQ). the extracted code is 'boqk'. 
* Download MSCOCO [validation images](http://images.cocodataset.org/zips/val2014.zip)
* Please make sure to modify the code for image paths in the ClipCocoDataset class within the utils.py file to obtain the correct image paths.
* Regarding the environment, please note that `transformers==2.8.0` is required; higher versions may cause incompatibility issues

## Factual model
If testing on SentiCap dataset, set `--gen_model_name_or_path` to the path where `'model_CLIPCap_small_4.pt'` is located, `--base_model_type` to `'gpt2'`, and `--gen_model_type` to `'base_small_senticap'`.
If testing on FlickrStyle10k dataset, set `--gen_model_name_or_path` to the path where `'model_CLIPCap_medium_5.pt'` is located, `--base_model_type` to `'gpt2-medium'`, and `--gen_model_type` to `'base_medium_flickr_style'`.

## Test on SentiCap dataset
* `--data_test` needs to be set to the path where `'Senticap_ViT-L_14_test.pkl'` is located.
* If generating positive captions, set `--gedi_model_name_or_path` to the path where `'model_pos_9.pt'` is located,  `--code_1` to `'positive'`, `--code_0` to `'negative'`, `--disc_weight` to `300`, `--teststyle` to `'positive'`.
* If generating negative captions, set `--gedi_model_name_or_path` to the path where `'model_neg_9.pt'` is located,  `--code_1` to `'negative'`, `--code_0` to `'positive'`, `--disc_weight` to `150`, `--teststyle` to `'negative'`.

## Test on FlickrStyle10k dataset
* `--data_test` needs to be set to the path where `'FlickrStyle10k_ViT-L_14_test.pkl'` is located.
* If generating romantic captions, set `--gedi_model_name_or_path` to the path where `'model_ro_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' romantic'`, `--code_0` to `' factual'`, `--disc_weight` to `30`, `--teststyle` to `'romantic'`.
* If generating humorous captions, set `--gedi_model_name_or_path` to the path where `'model_fu_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' humorous'`, `--code_0` to `' factual'`, `--disc_weight` to `39`, `--teststyle` to `'humorous'`.

## Examples on our device
* `positive rerults:`{"clipscore": 0.60205078125, "refclipscores": 0.65966796875, "bleu": [0.5334512154733899, 0.32402892866586547, 0.20344322551279603, 0.12937338929102704], "meteor": 0.18577257110673284, "rouge": 0.39125483800183614, "cider": 0.6814984903978035, "ppl": 13.13101, "acc": 0.9702823179791976}
* `negative rerults:`{"clipscore": 0.58984375, "refclipscores": 0.63671875, "bleu": [0.5152579892783051, 0.3042418000765133, 0.1900673140557742, 0.11723649343068628], "meteor": 0.1688917238930257, "rouge": 0.3754182740318773, "cider": 0.6271482401886957, "ppl": 14.70007, "acc": 0.9721669980119284}
* `romantic rerults:`{"clipscore": 0.642578125, "refclipscores": 0.63916015625, "bleu": [0.22356741899468535, 0.1045850053517702, 0.05280074080050402, 0.025500525755897775], "meteor": 0.10218282700411685, "rouge": 0.24985227626618087, "cider": 0.32640126766970795, "ppl": 35.859, "acc": 0.959}
* `humorous rerults:`{"clipscore": 0.63037109375, "refclipscores": 0.61279296875, "bleu": [0.21228245095838047, 0.0891456612789367, 0.039504376256192346, 0.018751173809324215], "meteor": 0.09291636135147632, "rouge": 0.21964106236745518, "cider": 0.2753052856719873, "ppl": 43.93906, "acc": 0.903}

 