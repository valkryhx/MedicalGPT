# -*- coding: utf-8 -*-
# time: 2023/8/30 14:27
# file: dpo_peftmodel_my_style.py
# author: hx


import random
import os
from dataclasses import dataclass, field
import argparse
from typing import List, Dict, Optional, Any, Mapping
from accelerate import init_empty_weights  # load an empty model,just structure , no real weight.
import bitsandbytes as bnb
import torch
import copy
from glob import glob
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig ,
    
    AutoConfig,
    BloomForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    BloomTokenizerFast,
)
from peft import (
    TaskType,
    LoraConfig,
    #AdaLoraConfig ,  #  提出自2020年 感觉和lora区别不大 而且和qlora有冲突 这里代码没有用到 
                     #例子https://www.zhihu.com/question/596950521/answer/3109759716
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM ,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers.deepspeed import HfDeepSpeedConfig ,is_deepspeed_zero3_enabled
import deepspeed
import json
from itertools import chain

from trl import DPOTrainer

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

MODEL_CLASSES = {
    "bloom": (AutoConfig, BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoConfig, AutoModel, AutoTokenizer),
    "llama": (AutoConfig, LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
}

# 关闭dataset的cache 这样每次都重新生成 测试用 不用cache避免使用到旧数据集
# import datasets
# datasets.disable_caching()



@dataclass
class ScriptArguments:
    """
    这里换写法 不用原生的argparse.ArgumentParser 换dataclass试试
    """
    # Model arguments
    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "Train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "Eval batch size per device"})
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
    min_target_length: Optional[int] = field(default=4, metadata={"help": "Min length of output text"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # Training arguments
    use_ref_model: bool = field(default=True, metadata={"help": "Whether to create a ref_model by yourself,set False will invoke the function to automatically create a ref_model."})
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    target_modules: Optional[str] = field(default=None)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=16.0)
    peft_path: Optional[str] = field(default=None)
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the validation set."})
    beta: Optional[float] = field(default=0.1, metadata={"help": "The beta parameter for DPO loss"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "Learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "The lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "The number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "The weight decay"})
    optim: Optional[str] = field(default="adamw_hf", metadata={"help": "The optimizer type"})
    fp16: Optional[bool] = field(default=True, metadata={"help": "Whether to use fp16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to use bf16"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "The number of gradient accumulation steps"}
    )
    save_total_limit: Optional[int] = field(default=None, metadata={"help": "limit total numbers of checkpoints."})
    load_best_model_at_end: Optional[bool] =field(default=False, metadata={"help": "whether to load the best checkpoints found during training."})
    save_steps: Optional[int] = field(default=50, metadata={"help": "X steps to save the model"})
    eval_steps: Optional[int] = field(default=50, metadata={"help": "X steps to evaluate the model"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "X steps to log the model"})
    output_dir: Optional[str] = field(default="outputs-dpo", metadata={"help": "The output directory"})
    max_steps: Optional[int] = field(default=200, metadata={"help": "Number of steps to train"})
    eval_strategy: Optional[str] = field(default="steps", metadata={"help": "Evaluation strategy"})
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset if `datasets.Dataset` is used"},
    )
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "Report to wandb or tensorboard"})

    ## add 20230830
    train_args_json: Optional[str] = field(default='luzi.json',metadata={"help": "默认TrainingArguments的json文件"})
    compute_dtype: Optional[str]=field(default='fp16',metadata={"help": "计算数据类型,可选范围为fp32,fp16,bf16"})
    local_rank: Optional[int]=field(default=0,metadata = {"help": "multi gpu scenario , for deepspeed use"})  
    #deepspeed:  Optional[str]=field(default=None,metadata={"help": "指定deepspeed config file path"})
    max_length: Optional[int] = field(default=512, metadata={"help": "没有用到。Max length of prompt+response text,整个QA长度的最大值"})

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            if 'default' in name: ## add 20230829
                continue 
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

# 用于数据集加载后的预处理 原始数据集json中有三个字段 Question/response_chosen/response_rejected 如果不是这三个字段 下面代码return对应更换
def return_prompt_and_responses(examples) -> Dict[str, str]:
    """Load the paired dataset and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    return {
        "prompt": ["Question: " + question + "\n\nAnswer: " for question in examples["question"]],
        "chosen": examples["response_chosen"],
        "rejected": examples["response_rejected"],
    }

# 这次先不自定义Trainer 直接用trl官方的DPOTrainer
# class LoRATrainer(Trainer): pass


def train():  
    # 其实就是dpo官方例子的def main 改造
    ## STEP 0 从命令行获取参数，包括trainingArgs在内的，以及各类附属参数
    logger.info("从json file中读取默认参数 并试用命令行参数覆盖")
    
    # 先读取命令行参数 因为其中有默认参数json file的path参数train_args_json
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]  # 这次试用dataclass的方法

    # 从train_args_json中读取默认的超参数  以及 deepspeed配置文件的config内容 都存到hf_train_args 这个变量是真正的要传入trainer的args
    hf_parser = HfArgumentParser(TrainingArguments)
    hf_train_args, = hf_parser.parse_json_file(json_file=args.train_args_json)

    # if args.deepspeed is not None :
    #     with open(args.deepspeed,'r',encoding='utf-8') as fr:   # 这里就是向TrainingArgs中添加deepseed字段
    #         hf_train_args.deepspeed = json.load(fr)  # set trainingArgs中deepspeed=ds_config

    # 使用命令行参数覆盖默认参数
    hf_train_args.per_device_train_batch_size=args.per_device_train_batch_size,
    hf_train_args.per_device_eval_batch_size=args.per_device_eval_batch_size,
    hf_train_args.max_steps=args.max_steps,
    hf_train_args.logging_steps=args.logging_steps,
    hf_train_args.save_steps=args.save_steps,
    hf_train_args.save_total_limit = args.save_total_limit,
    hf_train_args.load_best_model_at_end = args.load_best_model_at_end,
    hf_train_args.gradient_accumulation_steps=args.gradient_accumulation_steps,
    hf_train_args.gradient_checkpointing=args.gradient_checkpointing,
    hf_train_args.learning_rate=args.learning_rate,
    hf_train_args.evaluation_strategy=args.eval_strategy,
    hf_train_args.eval_steps=args.eval_steps,
    hf_train_args.output_dir=args.output_dir,
    hf_train_args.report_to=args.report_to,
    hf_train_args.lr_scheduler_type=args.lr_scheduler_type,
    hf_train_args.warmup_steps=args.warmup_steps,
    hf_train_args.optim=args.optim,
    hf_train_args.bf16=args.bf16,
    hf_train_args.fp16=args.fp16,
    hf_train_args.remove_unused_columns=args.remove_unused_columns,
    hf_train_args.run_name=f"dpo_{args.model_type}",
    hf_train_args.logging_dir = args.output_dir

    logger.debug(f"hf_train_args={hf_train_args}")
    
    # 额外处理tokenizer 加载
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_type == 'bloom':
        args.use_fast_tokenizer = True
    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "use_fast": args.use_fast_tokenizer,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_name_or_path = args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        #tokenizer_name_or_path = args.model_name_or_path
        raise ValueError("直接使用peft adpater当做model path 时必须额外传入tokenizer path，因为adapter中没有config.json 不能直接加载tokenizer！")
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # set as the <unk> token

    ## STEP 1 从本地加载train/eval 数据集 这次换个直接的写法
    data_files = {}
    if args.train_file_dir is not None and os.path.exists(args.train_file_dir):
        train_data_files = glob(f'{args.train_file_dir}/**/*.json', recursive=True) + glob(f'{args.train_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"train files: {', '.join(train_data_files)}")
        data_files["train"] = train_data_files

        # 这个validation 数据集一般要填写
    if args.validation_file_dir is not None and os.path.exists(args.validation_file_dir):
        eval_data_files = glob(f'{args.validation_file_dir}/**/*.json', recursive=True) + glob( f'{args.validation_file_dir}/**/*.jsonl', recursive=True)
        logger.info(f"eval files: {', '.join(eval_data_files)}")
        data_files["validation"] = eval_data_files
            
    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=args.cache_dir,
        )
        # 没有验证集就自动切分一部分数据作为验证集. 当然一般情况下 我们建议直接传一个path 
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[:{args.validation_split_percentage}%]",  # 注意train[:1%] 写法是支持的！参见https://www.tensorflow.org/datasets/splits
            cache_dir=args.cache_dir,
            )
        raw_datasets["train"] = load_dataset(
            'json',
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            cache_dir=args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets  这里之所以保留 max_source_length 以及求和 而不直接使用max_length 是因为DPOTrainer要用到max prompt length，也就是这里的source_length（估计是为了区分q和a）
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length
    full_max_length = max_source_length + max_target_length

    # Preprocess the dataset
    train_dataset = None
    max_train_samples = 0
    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if args.max_train_samples is not None and args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), args.max_train_samples)
            train_dataset = train_dataset.shuffle().select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        tokenized_dataset = train_dataset.shuffle().map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset = tokenized_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num train_samples: {len(train_dataset)}")
        logger.debug("First train example:")
        logger.debug(train_dataset[0]['prompt'] + train_dataset[0]['chosen'])

    eval_dataset = None
    max_eval_samples = 0
    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        max_eval_samples = len(eval_dataset)
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
            eval_dataset = eval_dataset.shuffle().select(range(max_eval_samples))
        logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = eval_dataset.filter(
            lambda x: 0 < len(x['prompt'] + x['chosen']) <= full_max_length
                      and 0 < len(x['prompt'] + x['rejected']) <= full_max_length
        )
        logger.debug(f"Num eval_samples: {len(eval_dataset)}")
        logger.debug("First eval example:")
        logger.debug(eval_dataset[0]['prompt'] + eval_dataset[0]['chosen'])

    ## STEP 2  定义 data collator  本次使用trl库 不需要自定义 因为DPOTrainer发现 data collator为空时 会自动创建
    
    ## STEP 3  load model
    if args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")
    logger.error(f"args.qlora={args.qlora}")
    
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        #config=config,
        #low_cpu_mem_usage=True,
        torch_dtype=_compute_dtype_map[args.compute_type],#torch.float16,
        #load_in_4bit=True,
        device_map= {"":0},#'auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        ) if args.qlora else None,
        trust_remote_code = True,      
    )
    # 在使用普通loramodel当作训练模型时 ref_model=None 避免手动copy model 制造ref_model导致oom
    model_ref=copy.deepcopy(model).to("cuda:1") if args.use_ref_model ==True else None

    logger.error(f"id(model)={id(model)}")
    logger.error(f"id(model_ref)={id(model_ref)}")
  
    # Initialize our Trainer
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    ## 建议add this line for training a model which params are frozen ,like lora.
    model.enable_input_require_grads()

    ## STEP4 将model转化为peftModel 准备loRA微调 此时已经是个peftmodel 了 所以这里不用再写
    # 构造参数 target_modules 这些modules是需要进行lora微调的
    target_modules = args.target_modules.split(',') if args.target_modules else None #可以写 dense,qkv... 逗号分隔 也可以直接写all
    if target_modules and 'all' in target_modules:
        target_modules = find_all_linear_names(model, int4=args.load_in_4bit, int8=args.load_in_8bit)
    logger.info(f"Peft target_modules: {target_modules}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    logger.info(model) # 此时传入的这个model 是peft model 才对

    ## STEP 5 定义trainer
    trainer = DPOTrainer(
        model,
        ref_model = model_ref,  # 在使用普通loramodel当作训练模型时 ref_model=None 避免手动copy model 制造ref_model导致oom
        args=hf_train_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer = tokenizer, #　add to save tokenizer in every ckpt during training. 
        peft_config=peft_config if args.use_peft else None,
        max_prompt_length=args.max_source_length,
        max_length=full_max_length,
    )
    print_trainable_parameters(trainer.model)
    logger.debug(f"训练样本量: {len(train_dataset)}")
    logger.debug(f"验证样本量: {len(eval_dataset)}")

    ## STEP 6 
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.model.save_pretrained(os.path.join(args.output_dir,"base_model"))


# 执行流程
if __name__ =="__main__" :    
    train()

