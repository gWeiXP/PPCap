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
* Download the processed dataset and trained models from [Baidu Netdisk](https://pan.baidu.com/s/1rBb8-4_lp2IfxJkEO0MmmA). the extracted code is 'zp8c'. The 'classifier' folder needs to be placed in the current directory.
* Download MSCOCO [validation images](http://images.cocodataset.org/zips/val2014.zip)
* Please make sure to modify the code for image paths in the ClipCocoDataset class within the utils.py file to obtain the correct image paths.
* Regarding the environment, please note that `transformers==2.8.0` is required; higher versions may cause incompatibility issues


## Factual model
If using PureT-XE as the factual model, set `--pretrained_path` to the path where `'model_pureT_XE_16.pth'` is located, and set `--gen_model_type` to `'pureT_XE'`. If using PureT-SCST as the factual model, set `--pretrained_path` to the path where `'model_pureT_SCST_30.pth'` is located, and set `--gen_model_type` to `'pureT_SCST'`.

## Test on SentiCap dataset
* `--data_test` needs to be set to the path where `'Senticap_ViT-L_14_test.pkl'` is located.
* If generating positive captions, set `--gedi_model_name_or_path` to the path where `'model_pos_9.pt'` is located,  `--code_1` to `'positive'`, `--code_0` to `'negative'`, `--disc_weight` to `200`, `--teststyle` to `'positive'`.
* If generating negative captions, set `--gedi_model_name_or_path` to the path where `'model_neg_9.pt'` is located,  `--code_1` to `'negative'`, `--code_0` to `'positive'`, `--disc_weight` to `175`, `--teststyle` to `'negative'`.

## Test on FlickrStyle10k dataset
* `--data_test` needs to be set to the path where `'FlickrStyle10k_ViT-L_14_test.pkl'` is located.
* If generating romantic captions, set `--gedi_model_name_or_path` to the path where `'model_ro_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' romantic'`, `--code_0` to `' factual'`, `--disc_weight` to `140`, `--teststyle` to `'romantic'`.
* If generating humorous captions, set `--gedi_model_name_or_path` to the path where `'model_fu_1.pt'` is located, (Ensure there is a space at the beginning of `--code_1` and `--code_0`) `--code_1` to `' humorous'`, `--code_0` to `' factual'`, `--disc_weight` to `175`, `--teststyle` to `'humorous'`.

## Examples on our device
* `positive:`generate parameters %s Namespace(batch_size=36, code_0='negative', code_1='positive', data_test='/home/liwc/wxp/Alignment/github/dataset/Senticap/Senticap_ViT-L_14_test.pkl', device='cuda:1', disc_weight=200, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_pos_9.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='positive', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `negative:`generate parameters %s Namespace(batch_size=36, code_0='positive', code_1='negative', data_test='/home/liwc/wxp/Alignment/github/dataset/Senticap/Senticap_ViT-L_14_test.pkl', device='cuda:1', disc_weight=175, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_neg_9.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='negative', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `romantic:`generate parameters %s Namespace(batch_size=36, code_0=' factual', code_1=' romantic', data_test='/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl', device='cuda:1', disc_weight=140, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_ro_1.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='romantic', vocab_path='./mscoco/txt/coco_vocabulary.txt')
* `humorous:`generate parameters %s Namespace(batch_size=36, code_0=' factual', code_1=' humorous', data_test='/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl', device='cuda:1', disc_weight=175, gedi_model_name_or_path='/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_fu_1.pt', gen_model_type='pureT_SCST', generated_path='./generated/guide', max_seq_len=17, pretrained_path='/home/liwc/wxp/Alignment/github/trained_model/factual_model/model_pureT_SCST_30.pth', teststyle='humorous', vocab_path='./mscoco/txt/coco_vocabulary.txt')

 