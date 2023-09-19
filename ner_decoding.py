import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import string


punctuation = string.punctuation



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def remove_duplicates(lst):
    lst = [list(dict.fromkeys(sub_lst)) for sub_lst in lst]
    seen = set()
    lst = [x for x in lst if tuple(x) not in seen and not seen.add(tuple(x))]
    return lst

def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = str(sys.argv[1]), #"./lora-alpaca" #"tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.0,
        top_p=1.0,
        top_k=65536,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
#        prompt = prompter.generate_prompt(instruction, input)
        prompt = instruction
#        print("prompt:")
#        print(prompt)
#        print("input:")
#        print(input)
        inputs = tokenizer(prompt, return_tensors="pt")

#        input_ = [remove_duplicates(tokenizer([item for item in input.split(" ")] + [" " + item for item in input.split(" ")] + [" <<< >>> <im_end>"], add_special_tokens=False).input_ids)]
        constraint_words = []
        input_ = tokenizer([item for item in input.split(" ")] + [" " + item for item in input.split(" ")] + [" <<< >>> <im_end>"], add_special_tokens=False).input_ids
        for item in input_:
            constraint_words += item



        print("input_")
        print(input_)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        force_triggers_ids = input_
        #[remove_duplicates(tokenizer([i for i in input.split() if i not in punctuation],
     #                                                 add_special_tokens=False).input_ids)]
#        print(force_triggers_ids)
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
            "prefix_allowed_tokens_fn": lambda batch_id, sent: constraint_words
        }
        generation_output = ""
#        print("pre_start")

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
#            print("start")
#            print("generator:")
#            print(generator)
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)
#                print("decoded_output:")
#                print(decoded_output)
                if len(decoded_output) >= 8 and decoded_output[-8:]=='<im_end>':
                    generation_output = decoded_output
                    break

#        with torch.no_grad():
#            generation_output = model.generate(
#                input_ids=input_ids,
#                generation_config=generation_config,
#                return_dict_in_generate=True,
#                output_scores=True,
#                max_new_tokens=max_new_tokens,
#            )

#        s = generation_output.sequences[0]
#        output = tokenizer.decode(s)
#        print("output:", generation_output)
        output = generation_output
        return output

    # Old testing code follows.
    # testing code for readme
    import json
    fi = open("test_v3.json")
    p = 0
    r = 0
    pa = 0
    ra = 0

#a="Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease extract the entity of quantity in the input sentence given below , the entity of quantity refers to the entity that represents a specific quantity or unit of measurement in the input sentence .\n\n### Input:\nIt has a population of about 750,000 , and is the largest city in the Yucatán Peninsula .\n\n### Response:\n<im_start>"

    for line in fi:
        obj = json.loads(line.strip())
#        instruction = obj["instruction"] + " " + obj["input"] + " <im_start>"
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n" + obj["instruction"] + "\n\n### Input:\n" + obj["input"] + "\n\n### Response:\n<im_start> i can extract entities for you, the extracted entities are"
        out = obj["output"].split("<im_end>")[0].lower()
        print("------------------------------")

        print(obj["output"].split("<im_end>")[0])
        print("----------------------end---------")
        arr = out.split("<<<")
        ents = [item.split(">>>")[0] for item in arr[1:]]
#        z = evaluate(instruction)
        response = evaluate(instruction, input=obj["input"] + " <im_end>").lower()
        print("response:")
        print(response)
        arr = response.split("<<<")[1:]
        out_ents = []
        for item in arr:
            try:
                if item.split(">>>")[0].strip() in obj["input"].lower():
                    out_ents.append(item.split(">>>")[0])
            except:
                pass
        out_ents = list (set (out_ents))
        print("out_ents:")
        print(out_ents)
        print("ents")
        print(ents)
        ra += len(out_ents)
        pa += len(ents)
        for item in ents:
            if item in out_ents:
                p += 1
        for item in out_ents:
            if item in ents:
                r += 1
        if pa > 0 and ra > 0:
            print(p / pa, r / ra)


#        out_ents = [item.split(">>>")[0] for item in arr]

#


if __name__ == "__main__":
    main()
