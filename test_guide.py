import argparse
import json
import logging
import os
import random
import numpy as np
import torch

from tqdm import tqdm, trange
import sys
import math
from torch.utils.data import DataLoader

import clip
from transformers import GPT2Config, GPT2Tokenizer

from modeling_gpt2 import GPT2LMHeadModel
from clipscore.eval_clip import computer_clipscore_and_other
from utils import ClipCaptionModel, ClipCocoDataset, eval_ppl, eval_acc

import os.path


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}
#GPT2 added as per - https://huggingface.co/transformers/model_doc/gpt2.html
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--gen_model_name_or_path", default="/home/liwc/wxp/Alignment/github/trained_model_PPCap/factual_model/model_CLIPCap_medium_5.pt", type=str)
    parser.add_argument("--base_model_type", default="gpt2-medium", type=str)
    parser.add_argument("--generated_path", default="./generated/guide")
    parser.add_argument("--gen_model_type", default="base_medium_flickr_style")
    parser.add_argument("--gedi_model_name_or_path", default="/home/liwc/wxp/Alignment/github/trained_model/stylized_model/model_fu_1.pt", type=str)
    parser.add_argument("--gedi_model_type", default="gpt2", type=str)
    parser.add_argument("--testdata", default="/home/liwc/wxp/Alignment/github/dataset/FlickrStyle10k/FlickrStyle10k_ViT-L_14_test.pkl", type=str)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int)
    parser.add_argument("--teststyle", default="humorous")
    parser.add_argument("--code_1", default=" humorous")
    parser.add_argument("--code_0", default=" factual")
    parser.add_argument("--do_norm", default=False)
    parser.add_argument("--disc_weight", type=float, default=39)


    parser.add_argument("--gen_length", type=int, default=21, help= "generation length")
    parser.add_argument("--clip_model_type", default="ViT-L/14")
    parser.add_argument("--prefix_length", default=4)
    parser.add_argument("--prefix_dim", default=768)
    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--logit_scale", default=True, type=bool)
    parser.add_argument("--penalize_cond", default=True)

    parser.add_argument("--fp16",action="store_true")
    parser.add_argument("--load_in_half_prec",action="store_true")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--filter_p", type=float, default=0.8)
    parser.add_argument("--class_bias", type=float, default=None)
    parser.add_argument("--target_p", type=float, default=0.8)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--repetition_penalty", default=1.2, type=float, help="The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.")
    parser.add_argument("--rep_penalty_scale", default=10.0, type=float)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--p", type=float, default=None)
    parser.add_argument("--gen_type", type=str, default="gedi", help="gedi, cclm, or gpt2")
    parser.add_argument("--mode", type=str, default="sentiment", help="topic, sentiment, detoxify")
    parser.add_argument("--secondary_code", type=str, default="business", help="secondary topic control code")
    parser.add_argument("--gpt3_api_key", type=str, default=None,  help= "GPT-3 api key" )
    parser.add_argument("--prompt", type=str, default="",  help= "prompt for generation" )
    

    args = parser.parse_args()
    assert(args.gen_type=="gedi" or args.gen_type=="cclm" or args.gen_type=="gpt2")
    assert(args.mode=="topic" or args.mode=="sentiment" or args.mode=="detoxify")


    # args.outbias = True
    # args.do_sample = True
    # args.p = 0.6
    # args.k = 0
    # args.temperature = 1.1


    if args.mode == "topic":
        args.code_1 = "true"
        args.code_0 = "false"
        if not args.gen_type == "gpt2":
            if args.gedi_model_name_or_path is None:
                args.gedi_model_name_or_path = "../pretrained_models/gedi_topic"
                if not os.path.isdir(args.gedi_model_name_or_path):
                    raise Exception("GeDi model path not found, must either run `get_models.sh' or set `args.gedi_model_name_or_path'")
        if args.class_bias is None:
            args.class_bias = 0.0

    if args.mode == "sentiment":
        # args.code_desired = "positive"
        # args.code_undesired = "negative"
        if not args.gen_type == "gpt2":
            if args.gedi_model_name_or_path is None:
                args.gedi_model_name_or_path = "../pretrained_models/gedi_sentiment"
                if not os.path.isdir(args.gedi_model_name_or_path):
                    raise Exception("GeDi model path not found, must either run `get_models.sh' or set `args.gedi_model_name_or_path'")

        if args.class_bias is None:
            args.class_bias = 0.0

    if args.mode == "detoxify":
        args.code_1 = "clean"
        args.code_0 = "dirty"

        if not args.gen_type == "gpt2":
            if args.gedi_model_name_or_path is None:
                args.gedi_model_name_or_path = "../pretrained_models/gedi_detoxifier"
                if not os.path.isdir(args.gedi_model_name_or_path):
                    raise Exception("GeDi model path not found, must either run `get_models.sh' or set `args.gedi_model_name_or_path'")
        if args.class_bias is None:
            args.class_bias = 2.0


            if args.target_p<1 and args.target_p>0:
                inv_p = math.log(args.target_p/(1-args.target_p))+args.class_bias
                args.target_p = 1/(1+math.exp(-1*inv_p))
                print("changing target p to "  + str(args.target_p) + " to account for class bias term.")






    device = args.device
    args.n_gpu = 1

    args.seed = 42
    set_seed(args)





    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]




    tokenizer = tokenizer_class.from_pretrained("gpt2", do_lower_case=False)


    if args.gen_type == "cclm": # 暂不管
        model = model_class.from_pretrained(args.gedi_model_name_or_path)
        model.to(args.device)
        args.length = adjust_length_to_model(args.gen_length,
                             max_sequence_length=model.config.max_position_embeddings)
    else:

        if not(args.gpt3_api_key is None):
            print("It's a bit wasteful but our code needs to load GPT-2 even if using GPT-3")


        if args.load_in_half_prec:
            model = model_class.from_pretrained(args.gen_model_name_or_path,load_in_half_prec=True)
            model.to(args.device)
            #even if using --fp16, apex doesn't accept half prec models, so requires converting to full prec before converting back
            model = model.float()

        else:
            config = config_class.from_pretrained(args.base_model_type)
            config.nbias = 0
            config.logit_scale = True
            gpt_base = model_class.from_pretrained(args.base_model_type, config=config)
            model = ClipCaptionModel(tokenizer, gpt_base, args.prefix_length, prefix_size=args.prefix_dim)
            model.load_state_dict(torch.load(args.gen_model_name_or_path, map_location="cpu"))
            model.to(args.device)




        #using fp16 option with GPT2-XL can prevent OOM errors
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")



            #opt_level O3 gives fully half precision weights. Okay for GPT-2 as long as we don't need to finetune
            model = amp.initialize(model, opt_level='O3')
            torch.cuda.empty_cache()


        args.length = adjust_length_to_model(args.gen_length, max_sequence_length=1024)



    if args.gen_type == "gedi" :
        config = config_class.from_pretrained(args.gedi_model_type)
        config.nbias = 0
        config.logit_scale = True
        gpt_gedi = model_class.from_pretrained(args.gedi_model_type, config=config)
        gedi_model = ClipCaptionModel(tokenizer, gpt_gedi, args.prefix_length, prefix_size=args.prefix_dim)
        gedi_model.load_state_dict(torch.load(args.gedi_model_name_or_path, map_location="cpu"))
        gedi_model.to(args.device)

    else:
        gedi_model=None

    
    # clip模型
    clipmodel, preprocess = clip.load(args.clip_model_type, device=args.device, jit=False)

    # 打开数据集
    args.max_length = args.length
    dataset = ClipCocoDataset(args, args.testdata, train_or_test='test', tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False, drop_last=False)

    
    # weight_set = [
    #               (0.6, 75), (0.6, 90), (0.6, 120),
    #               (0.6, 150), (0.75, 30), (0.75, 35),
    #               (0.9, 25), (0.9, 30), (0.9, 35),
    #               ]
    # for weight in weight_set:
    #     args.p, args.disc_weight = weight
    # weight_set = [150, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    # weight_set = [0, 25, 50, 75, 100, 125, 150, 175, 200]

    # # args.disc_weight = 39    # weight_set = [100, 0, 25, 50, 75, 100, 125, 150, 175, 200]
    # for weight in weight_set:
    #     args.disc_weight = weight
    if True:

        ref = []
        # cocoeval_out = []
        gt = []
        image_paths = []


        for idx, (prefix, style, captions, imgpath, idxs) in tqdm(enumerate(test_dataloader)):
            with torch.no_grad():
                # 取出样本信息
                prefix = prefix.to(args.device, dtype=torch.float32)
                if args.do_norm:
                    prefix = prefix / prefix.norm(2, -1, keepdim=True)
                captions = [caption.split('\n') for caption in captions]
                image_paths.extend(imgpath)

                # 生成
                generated_sequences = model.gpt.generate(
                    input_ids=None,
                    pad_lens=None,
                    max_length= args.length,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    rep_penalty_scale=args.rep_penalty_scale,
                    eos_token_ids = tokenizer.eos_token_id,
                    pad_token_id = tokenizer.eos_token_id,
                    do_sample=args.do_sample,
                    penalize_cond=args.penalize_cond,
                    gedi_model=gedi_model,
                    base_model=model,   # 2
                    gpt3_api_key = args.gpt3_api_key,
                    tokenizer=tokenizer,
                    disc_weight=args.disc_weight,
                    filter_p = args.filter_p,
                    target_p = args.target_p,
                    class_bias = args.class_bias,
                    attr_class = 1,
                    code_0 = args.code_0,
                    code_1 = args.code_1,
                    multi_code=1,
                    prefix_sequence=prefix,  # 1
                    style = style # 3
                    )
                
                for i in range(generated_sequences.shape[0]):
                    generated_sequence = generated_sequences[i]
                    generated_sequence = generated_sequence[generated_sequence != tokenizer.eos_token_id]
                    generated_text = tokenizer.decode(generated_sequence.tolist(), clean_up_tokenization_spaces=True)
                    ref.append(generated_text)


                    # refs_cocoeval_out = []
                    # ref_cocoeval_out = {}
                    # ref_cocoeval_out['image_id'] = idxs[i]
                    # ref_cocoeval_out['caption'] = generated_text
                    # ref_cocoeval_out['id'] = idx*args.per_gpu_eval_batch_size + i
                    # refs_cocoeval_out.append(ref)
                    # cocoeval_out.append(refs_cocoeval_out)
                
                for caption in captions:
                    gt.append(caption)


        result = {}


        output_file_path = "output.txt"
        with open(output_file_path, 'w') as file:
        # 遍历列表中的每个字符串
            for path in image_paths:
                # 将字符串写入文件，每个字符串占据一行
                file.write(path + '\n')


        # 计算Bleu、Cider等指标
        clipscores, other_metrics = computer_clipscore_and_other(image_paths, clipmodel, device, ref, gt)

        
        
        result.update(clipscores)
        result.update(other_metrics)

        # 保存描述
        out_txt_dir = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle +"/captions_generate_"+ str(args.disc_weight) + ".txt"
        with open(out_txt_dir, "w") as file:
                for generate_ref in ref:
                    file.write(generate_ref + "\n")


        # 计算ppl评估指标
        ppl_out_path = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle + "/ppl_out_" + str(args.disc_weight) + ".txt"
        result["ppl"] = eval_ppl(out_txt_dir, args.teststyle, ppl_out_path)

        # 计算acc评估指标
        file_error = args.generated_path + "/" + args.gen_model_type + "/" + args.teststyle + "/file_error_"+str(args.disc_weight)+".txt"
        result["acc"] = eval_acc(out_txt_dir, args.teststyle, device, tokenizer, file_error)

        for key, value in result.items():
            if isinstance(value, np.float16):
                result[key] = float(value)
        print(json.dumps({**result}))

        print("Its weight is :" + str(args.disc_weight))











if __name__ == "__main__":
    main()
