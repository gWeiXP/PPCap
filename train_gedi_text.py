# senticap
import argparse
import json
import logging
import os
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import sys

from transformers import AdamW, get_linear_schedule_with_warmup, GPT2Config, GPT2Tokenizer
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors



from modeling_gpt2 import GPT2LMHeadModel
from utils import ClipCocoDataset, ClipCaptionModel, set_seed, noise_injection, add_sep, eval_ppl, eval_acc

import clip
from clipscore.eval_clip import computer_clipscore_and_other

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}



def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    model = model.to(args.device)
    model.train()
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * 1,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
    )
    set_seed(args)  # Added here for reproductibility

    pt_id = tokenizer.encode(args.code_1)[0]
    nt_id = tokenizer.encode(args.code_0)[0]

    for epoch_ in train_iterator:
        model.zero_grad()
        epoch_iterator = train_dataloader

        overall_gen_loss = 0
        eval_loss = 0
        loss_epoch = 0
        num_loss = 0

        for step, (tokens, mask, prefix, style_token, label) in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            tokens, mask, label = tokens.to(args.device), mask.to(args.device), label.to(args.device)
            prefix = prefix.to(args.device, dtype=torch.float32)
            if args.do_norm:
                prefix = prefix / prefix.norm(2, -1, keepdim=True)
            prefix = noise_injection(prefix, args.variance, modality_offset=None, uniform_noise=False, dont_norm=False)
            style_token = style_token.view(-1, 1).type_as(tokens)

            # batch_0 = add_sep(batch, tokenizer.encode('<sep>')) if args.add_sep else tokens
            weight = (style_token == pt_id).type_as(style_token).view(-1,1).to(args.device)
            seq_a = pt_id * weight + nt_id * (1-weight)
            seq_b = nt_id * weight + pt_id * (1-weight)



            seq_a = torch.cat((seq_a, tokens), dim=1)[:,:-1]
            seq_b = torch.cat((seq_b, tokens), dim=1)[:,:-1]
            bsz = seq_a.shape[0]
            seq_batched = torch.cat((seq_a,seq_b),dim=0)
            # 更新前缀
            prefix = (torch.cat((prefix, prefix), dim=0))
            # 更新mask
            mask = mask[:, :-1].to(torch.float32).to(args.device)   # mask，加入T/F和style的mask
            left_ = torch.ones(mask.shape[0], 1).type_as(mask)
            mask = torch.cat((left_, mask), dim=1)
            mask = (torch.cat((mask, mask), dim=0))
            # 输入到模型中
            outputs = model(seq_batched, prefix, mask)
            losses = outputs[0].view(seq_batched.shape[0], -1)
            #loss mask includes first padded token
            if args.mask_eos_token:
                loss_mask = mask[:,train_dataset.prefix_length+1:-1].to(torch.float16)
                if args.add_sep:
                    raise NotImplementedError
            else:
                loss_mask = mask[:,train_dataset.prefix_length:-1].to(torch.float32)
                #appending with ones to account for the control code token being added
                if args.add_sep:
                    #adding the sep token would require extending the loss mask of ones by one position to the right (equivalent to prepending one on the left)
                    left_ = torch.ones(loss_mask.shape[0],2).type_as(loss_mask)
                    loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1)


            loss_lengths = torch.sum(loss_mask,1,keepdim=True)
            loss_lengths_a, loss_lengths_b = torch.split(loss_lengths, bsz, dim=0)
            loss_a,loss_b=torch.split(losses, bsz, dim=0)
            mask_a, mask_b = torch.split(loss_mask, bsz, dim=0)

            loss_a = loss_a * mask_a
            loss_b = loss_b * mask_b

            gen_loss_a = (label==1).to(torch.float32).unsqueeze(1)*loss_a/loss_lengths_a
            gen_loss_b = (label==0).to(torch.float32).unsqueeze(1)*loss_b/loss_lengths_b

            gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz

            if args.sum_loss:
                loss_a = loss_a.sum(dim=1)
                loss_b= loss_b.sum(dim=1)
            else:
                loss_a = (loss_a/loss_lengths_a).sum(dim=1)
                loss_b= (loss_b/loss_lengths_b).sum(dim=1)

            class_logits = torch.stack((-loss_b, -loss_a), dim=1) #(bsz, 2) dimensional
            label[label == 2] = 1  #turning 3-ary to binary
            class_labels = label

            if args.logit_scale:
                class_logits*=model.gpt.logit_scale

            if args.outbias:
                class_logits+=model.gpt.bias

            loss_fn = torch.nn.CrossEntropyLoss()
            match_loss = loss_fn(class_logits, class_labels)
            loss = match_loss*args.disc_weight + args.gen_weight*gen_loss

            overall_gen_loss = overall_gen_loss + gen_loss * bsz
            eval_loss = eval_loss + match_loss * bsz
            loss_epoch = loss_epoch + loss * bsz
            num_loss = num_loss + bsz


            if np.isnan(loss.detach().cpu().numpy()):
                import pdb; pdb.set_trace()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        if epoch_ % args.save_every == 0 or epoch_ == epochs_trained - 1:
            logs = {}
            # train 
            logs["loss_last_epoch"] = loss.item()   # 本epoch最后一个batch的损失
            # train 这个是一个epoch所有batch的平均
            logs["train_overall_gen_loss"] = (overall_gen_loss / num_loss).item()   # 本epoch所有batch的平均生成损失
            logs["train_eval_loss"] = (eval_loss / num_loss).item() # 本epoch所有batch的平均鉴别损失
            logs["train_loss"] = (loss_epoch / num_loss).item() # 本epoch所有batch的平均损失

            # evaluate
            if (
                args.evaluate_during_training
            ):
                results = evaluate(args, model, tokenizer)
                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value

            # else
            learning_rate_scalar = scheduler.get_lr()[0]
            logs["learning_rate"] = learning_rate_scalar

            # writer and print
            for key, value in logs.items():
                tb_writer.add_scalar(key, value, epoch_)
            print(json.dumps({**logs, **{"epoch": epoch_}}))

        if epoch_ % args.save_every == 0 or epoch_ == epochs_trained - 1:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"{args.output_prefix}-{epoch_:03d}.pt"),
            )
            # evaluate(args, model, testdataset, tokenizer=tokenizer, epoch=epoch_)

    tb_writer.close()

def evaluate(args, model, tokenizer, prefix_name=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = ClipCocoDataset(args, args.testdata_loss, train_or_test='train', tokenizer=tokenizer)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly

        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix_name))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        overall_gen_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        pt_id = tokenizer.encode(args.code_1)[0]
        nt_id = tokenizer.encode(args.code_0)[0]


        for idx, (tokens, mask, prefix, style_token, label) in enumerate(eval_dataloader):
            model.eval()
            tokens, mask, label = tokens.to(args.device), mask.to(args.device), label.to(args.device)
            prefix = prefix.to(args.device, dtype=torch.float32)
            style_token = style_token.view(-1, 1).type_as(tokens)

            with torch.no_grad():
                # 加入style
                weight = (style_token == pt_id).type_as(style_token).view(-1,1).to(args.device)
                seq_a = pt_id * weight + nt_id * (1-weight)
                seq_b = nt_id * weight + pt_id * (1-weight)
                seq_a = torch.cat((seq_a, tokens), dim=1)[:,:-1]
                seq_b = torch.cat((seq_b, tokens), dim=1)[:,:-1]
                bsz = seq_a.shape[0]
                seq_batched = torch.cat((seq_a,seq_b),dim=0)
                # 更新前缀
                prefix = (torch.cat((prefix, prefix), dim=0))
                # 更新mask
                mask = mask[:, :-1].to(torch.float32).to(args.device)   # mask，加入T/F和style的mask
                left_ = torch.ones(mask.shape[0], 1).type_as(mask)
                mask = torch.cat((left_, mask), dim=1)
                mask = (torch.cat((mask, mask), dim=0))
                # 输入到模型中
                outputs = model(seq_batched, prefix, mask)  
                losses = outputs[0].view(seq_batched.shape[0], -1)

                #loss mask includes first padded token
                if args.mask_eos_token:
                    loss_mask = mask[:,eval_dataset.prefix_length+1:-1].to(torch.float16)
                    if args.add_sep:
                        raise NotImplementedError
                else:
                    loss_mask = mask[:,eval_dataset.prefix_length:-1].to(torch.float32)
                    #appending with ones to account for the control code token being added
                    if args.add_sep:
                        #adding the sep token would require extending the loss mask of ones by one position to the right (equivalent to prepending one on the left)
                        left_ = torch.ones(loss_mask.shape[0],2).type_as(loss_mask)
                        loss_mask = torch.cat((left_, loss_mask[:,:-1]), dim=1)

                loss_lengths = torch.sum(loss_mask,1,keepdim=True)
                loss_lengths_a, loss_lengths_b = torch.split(loss_lengths, bsz, dim=0)
                loss_a,loss_b=torch.split(losses, bsz, dim=0)
                mask_a, mask_b = torch.split(loss_mask, bsz, dim=0)

                loss_a = loss_a * mask_a
                loss_b = loss_b * mask_b

                gen_loss_a = (label==1).to(torch.float32).unsqueeze(1)*loss_a/loss_lengths_a    # 句子长度进行了平均
                gen_loss_b = (label==0).to(torch.float32).unsqueeze(1)*loss_b/loss_lengths_b

                gen_loss = torch.sum(gen_loss_a+gen_loss_b)/bsz     # bachsize进行了平均

                if args.sum_loss:
                    loss_a = loss_a.sum(dim=1)
                    loss_b= loss_b.sum(dim=1)
                else:
                    loss_a = (loss_a/loss_lengths_a).sum(dim=1)
                    loss_b= (loss_b/loss_lengths_b).sum(dim=1)

                class_logits = torch.stack((-loss_b, -loss_a), dim=1) #(bsz, 2) dimensional
                label[label == 2] = 1  #turning 3-ary to binary
                class_labels = label

                loss_fn = torch.nn.CrossEntropyLoss()

                if args.logit_scale:
                    class_logits*=model.gpt.logit_scale

                if args.outbias:
                    class_logits+=model.gpt.bias

                loss = loss_fn(class_logits, class_labels)

                tmp_eval_loss = loss
                tmp_gen_loss = gen_loss
                logits = class_logits


                eval_loss += tmp_eval_loss.mean().item()
                overall_gen_loss += tmp_gen_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = class_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, class_labels.detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        overall_gen_loss = overall_gen_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        result.update({'overall_gen_loss':overall_gen_loss})
        result.update({'eval_loss':eval_loss})

    return result

def evaluate2(args, model, tokenizer, epoch):
    # dataset
    dataset = ClipCocoDataset(args, args.testdata, train_or_test='test', tokenizer=tokenizer)
    test_dataloader = DataLoader(dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False, drop_last=False)

    ref = []
    gt = []
    image_paths = []
    
    device = args.device
    clipmodel, _ = clip.load("ViT-L/14", device=device, jit=False)

    result = {}
    for idx, (prefix, style, captions, imgpath, idxs) in enumerate(test_dataloader):
        with torch.no_grad():
            prefix = prefix.to(device, dtype=torch.float32)
            if args.do_norm:
                prefix = prefix / prefix.norm(2, -1, keepdim=True)
            if args.do_sign:
                prefix = torch.nn.functional.normalize(prefix, dim=1)
            image_paths.extend(imgpath)
            captions = [caption.split('\n') for caption in captions]
            
            generated_sequences = model.generate(prefix=prefix, args=args, tokenizer=tokenizer, style=style)
            for i in range(generated_sequences.shape[0]):
                generated_sequence = generated_sequences[i]
                generated_sequence = generated_sequence[generated_sequence != tokenizer.eos_token_id]
                generated_text = tokenizer.decode(generated_sequence.tolist(), clean_up_tokenization_spaces=True)
                ref.append(generated_text)   
            for caption in captions:
                gt.append(caption)

    # 计算Bleu、Cider等指标
    clipscores, other_metrics = computer_clipscore_and_other(image_paths, clipmodel, device, ref, gt)
    result.update(clipscores)
    result.update(other_metrics)

    # 保存描述
    out_txt_dir = args.generated_path + "/" + args.teststyle + "/captions_generate_"+str(epoch)+".txt"
    with open(out_txt_dir, "w") as file:
            for generate_ref in ref:
                file.write(generate_ref + "\n")

    # 计算ppl评估指标
    ppl_out_path = args.generated_path + "/" + args.teststyle + "/ppl_out_" + str(epoch) + ".txt"
    result["ppl"] = eval_ppl(out_txt_dir, args.teststyle, ppl_out_path)

    # 计算acc评估指标
    file_error = args.generated_path + "/" + args.teststyle + "/file_error_"+str(epoch)+".txt"
    result["acc"] = eval_acc(out_txt_dir, args.teststyle, device, tokenizer, file_error)

    return result


def main():
    parser = argparse.ArgumentParser()

    # train add
    parser.add_argument('--data', default='./data/dataset_train/Senticap_text_ViT-L_14_train.pkl')
    parser.add_argument('--testdata', default='./data/Senticap/Senticap_ViT-L_14_test.pkl')
    parser.add_argument('--teststyle', default='negative')
    parser.add_argument('--testdata_loss', default='')
    parser.add_argument("--do_train", default=True)
    parser.add_argument("--do_eval", default=False)
    parser.add_argument("--output_dir", default="./model/gedi/test", type=str, required=False)
    parser.add_argument("--generated_path", default="", type=str, required=False)
    parser.add_argument('--output_prefix', default='senticap')
    parser.add_argument("--do_norm", default=True)
    parser.add_argument("--do_sign", default=False) # 没有用了，可忽视
    parser.add_argument('--variance', default=0.016)
    parser.add_argument('--finetune', default=False)
    parser.add_argument('--pretrain_path', default="./model/base/base_mscoco_pretrain/mscoco-007.pt")


    parser.add_argument('--prefix_length', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=21)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--prefix_dim', type=int, default=768)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument("--code_0", type=str, default="positive")
    parser.add_argument("--code_1", type=str, default="negative")
    
    parser.add_argument("--num_train_epochs", default=20.0, type=float)
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int)
    parser.add_argument("--dropout",default=0.1,type=float)
    parser.add_argument("--gen_weight",default=0.8,type=float)
    parser.add_argument("--learning_rate", default=10e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument("--logit_scale", default=True, type=bool)
    parser.add_argument("--overwrite_output_dir", default=True, type=bool)
    parser.add_argument("--overwrite_cache", default=True, type=bool)
    parser.add_argument("--evaluate_during_training", default=False, type=bool)
    parser.add_argument("--eval_all_checkpoints", default=True, type=bool)
    parser.add_argument("--model_name_or_path", default="gpt2", type=str, required=False)
    parser.add_argument("--model_type", default="gpt2", type=str, required=False)
    parser.add_argument("--seed", type=int, default=42)

    # else
    parser.add_argument("--do_sample", action="store_true",
                        help="If set to False greedy decoding is used. Otherwise sampling is used. Defaults to True.")
    parser.add_argument("--temperature", type=float, default=1.0, help="lower tend toward greedy sampling")
    parser.add_argument("--top_k", type=float, default=None,
                        help="The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.",)
    parser.add_argument("--top_p", type=float, default=None,
                            help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.")
    parser.add_argument("--repetition_penalty", default=1.2, type=float,
                        help="The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.")
    parser.add_argument("--penalize_cond", action="store_true",
                        help="apply repetition penalty to tokens in coditioning text")
    parser.add_argument("--filter_p", type=float, default=0.8,
                        help="filters at up to filter_p cumulative probability from next token distribution.")
    parser.add_argument("--target_p", type=float, default=0.8,
                        help="In comination with filter_p, saves tokens with above target p probability of being in the correct class ")
    parser.add_argument("--class_bias", type=float, default=None, help="biases GeDi's classification probabilties")
    parser.add_argument("--task_name", default="sst-2", type=str, required=False, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),)
    parser.add_argument("--threeway", action="store_true", help="does 3-way classification")
    parser.add_argument("--sum_loss",action="store_true", help="sums losses")
    parser.add_argument("--outbias",action="store_true", help="learns output bias for each class")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--mask_eos_token", action="store_true", help="whether to mask eos token loss or not; prefer masking if training for DA",)
    parser.add_argument("--add_sep", action="store_true", help="Include sep token if this arg is used between the two sentences in a pair | can/should be used for mrpc/mnli/qqp/qnli")
    


    args, unknown = parser.parse_known_args()
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.n_gpu = 1

    args.disc_weight = 1.0 - args.gen_weight
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)


    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.outbias:
        if args.threeway:
            config.nbias=3
        else:
            config.nbias=2
    else:
        config.nbias=0

    config.embd_pdrop = args.dropout
    config.attn_pdrop = args.dropout
    config.resid_pdrop = args.dropout
    if args.logit_scale:
        config.logit_scale=True
    else:
        config.logit_scale=False

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.add_sep:
        special_tokens_dict = {'sep_token': '<SEP>'}
        tokenizer.add_special_tokens(special_tokens_dict)
    config.output_past = True #https://github.com/huggingface/transformers/pull/3734
    gpt = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    gpt.resize_token_embeddings(len(tokenizer))
    model = ClipCaptionModel(tokenizer, gpt, args.prefix_length, prefix_size=args.prefix_dim)

    if args.finetune:
        state_dict = torch.load(args.pretrain_path, map_location="cpu")
        model.load_state_dict(state_dict)


    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = ClipCocoDataset(args, args.data, train_or_test='train', tokenizer=tokenizer)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval:


        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [os.path.join(root, file) for root, dirs, files in os.walk(args.output_dir) for file in files]
            checkpoints.sort()
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            epoch = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            epoch = int(epoch[:-3])
            state_dict = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state_dict)
            model = model.to(args.device)

            result = evaluate2(args, model, tokenizer, epoch)

            for key, value in result.items():
                if isinstance(value, np.float16):
                    result[key] = float(value)
            print(json.dumps({**result, **{"epoch": epoch}}))





if __name__ == "__main__":
    main()
