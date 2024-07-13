import os
from typing import Dict, Union

import pandas as pd
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sara import UDHR_utils_sara as u_sara
from UDHR.get_UDHR_inputs import get_identities_dicts, get_rights_lists

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def make_model():
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    _ = torch.set_grad_enabled(False)
    model_path: str = "/HDD/models/google/Gemma-2B/"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    # model = ORTModelForSequenceClassification.from_pretrained(model_path,
    #                                                           export=True,
    #                                              quantization_config=quantization_config,
    #                                              low_cpu_mem_usage=True,
    #                                              )
    # enable BetterTransformer
    # model = model.to_bettertransformer()

    model.tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_kwargs: Dict[str, Union[float, int]] = {
        "temperature": 0.8,
        "top_p": 0.3,
        "max_new_tokens": 20,
        "do_sample": True,
    }

    return model, sampling_kwargs


def make_prompts():
    # Scenario or prompt to apply the modifications
    q_current, q_ideal = get_rights_lists()
    i_current = get_identities_dicts()

    BTC_SIZE = 50
    pre_prompt = "".join(open("UDHR/UDHR_Prompt_btc50.txt", "r").readlines())
    prompts, this_btc_prompts = [], []
    for q in q_current:
        for i_k in i_current.keys():
            for i in i_current[i_k]:
                new_prompt = q.replace("{people}", i)
                new_prompt = pre_prompt.replace("{right}", new_prompt)
                this_btc_prompts.append([i_k, i, q, new_prompt])
                if len(this_btc_prompts) == BTC_SIZE:
                    prompts.append(this_btc_prompts)
                    this_btc_prompts = []
    # for the last bacth with len()<BTC_SIZE
    prompts.append(this_btc_prompts)
    return prompts


def make_df(nom_df, mod_df, prompts, layer, i_class, i_name, r_name):
    nom_df = pd.DataFrame(
        {
            "prompts": prompts,
            "completions": nom_df,
            "is_modified": False,
            "layer": layer,
            "id_class": i_class,
            "identifier": i_name,
            "right": r_name,
        }
    )
    mod_df = pd.DataFrame(
        {
            "prompts": prompts,
            "completions": mod_df,
            "is_modified": True,
            "layer": layer,
            "id_class": i_class,
            "identifier": i_name,
            "right": r_name,
        }
    )
    return pd.concat([nom_df, mod_df], ignore_index=True)


def do_sara(model, sampling_kwargs):
    # Steering prompts
    prompt_to_be_repelled = "Nobody should have rights and freedoms."
    prompt_to_steer_to = "Everyone should have equal rights."

    prompts_btc = make_prompts()

    # We'll store 5 repetitions per layer, for all layers
    for rep_n in range(1):
        df = pd.DataFrame(
            columns=[
                "prompts",
                "completions",
                "is_modified",
                "layer",
                "id_class",
                "identifier",
                "right",
            ]
        )
        for layer in range(0, model.config.num_hidden_layers):
            print("Layer: " + str(layer) + " : rep_n: " + str(rep_n))
            A_attract = u_sara.get_vectors(
                model, model.tokenizer, [prompt_to_steer_to], layer
            )[0]
            A_repel = u_sara.get_vectors(
                model, model.tokenizer, [prompt_to_be_repelled], layer
            )[0]
            # Fetch the comparison data
            for i, prompts in tqdm.tqdm(enumerate(prompts_btc)):
                i_classes, i_names, r_names, prompts = zip(*prompts)
                comparison_df = u_sara.get_n_comparisons(
                    model=model,
                    tokenizer=model.tokenizer,
                    prompts=prompts,
                    A_attract=A_attract,
                    A_repel=A_repel,
                    layer=layer,
                    coeff=1,
                    **sampling_kwargs,
                )
                comparison_df = make_df(
                    *comparison_df,
                    prompts=prompts,
                    layer=layer,
                    i_class=i_classes,
                    i_name=i_names,
                    r_name=r_names,
                )
                df = pd.concat([df, comparison_df], ignore_index=True)
            print(
                "Layer "
                + str(layer)
                + " is done. "
                + str(model.config.num_hidden_layers - 1 - layer)
                + " remaining"
            )
            df.to_csv(
                open(
                    f"UDHR/responses/UDHR_sara_results_l_{layer}_rep_{rep_n}.csv", "w"
                ),
                sep="\t",
            )
        print("Rep_n " + str(rep_n) + " is done.")
        df.to_csv(open(f"UDHR/responses/UDHR_sara_results_{rep_n}.csv", "w"), sep="\t")


if __name__ == "__main__":
    model, sampling_kwargs = make_model()
    do_sara(model, sampling_kwargs)
