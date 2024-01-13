from typing import List, Optional, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,LlamaTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger
from vllm.transformers_utils.tokenizers import *
from .tokenization_baichuan import BaichuanTokenizer
from .tokenization_chatglm import ChatGLMTokenizer
from .tokenization_qwen import QWenTokenizer
import sentencepiece
logger = init_logger(__name__)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if ("llama" in tokenizer_name.lower() and kwargs.get("use_fast", True)
            and tokenizer_name != _FAST_LLAMA_TOKENIZER):
        logger.info(
            "For some LLaMA V1 models, initializing the fast tokenizer may "
            "take a long time. To reduce the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
    try:
        logger.info(f"{tokenizer_name.lower()}")
        if "llama" in tokenizer_name.lower() or "congrong_chat" in tokenizer_name.lower() or "cwcr" in tokenizer_name.lower():
            logger.info("use own tokenizer")
            # tokenizer_path = "/workspace/engine-llm-inference/tokenizer/llama"
            if "2.0" in tokenizer_name.lower():
                tokenizer = sentencepiece.SentencePieceProcessor()
                tokenizer.load(tokenizer_name+"/tokenizer.model")
                tokenizer.pad_token_id = 0
                tokenizer.eos_token_id = 2
                tokenizer.all_special_tokens = []
            # # elif "ielts" in tokenizer_name.lower():
            # #     tokenizer_path = tokenizer_name
            # #     tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=False,add_eos_token=False)
            # #     tokenizer.pad_token_id = tokenizer.eos_token_id
            # else:
            tokenizer_path = tokenizer_name
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=False)
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "baichuan" in tokenizer_name.lower() :
            logger.info("use baichuanTokenizer tokenizer")
            tokenizer_path = tokenizer_name
            tokenizer = BaichuanTokenizer.from_pretrained(tokenizer_path, legacy=False)
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "chatglm" in tokenizer_name.lower() :
            logger.info("use chatglm tokenizer")
            tokenizer_path = tokenizer_name
            logger.info(f"tokenizer_path:{tokenizer_path}")
            tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path, legacy=False)
        elif "qwen" in tokenizer_name.lower() :
            logger.info("use qwen tokenizer")
            tokenizer_path = tokenizer_name
            logger.info(f"tokenizer_path:{tokenizer_path}")
            tokenizer = QWenTokenizer.from_pretrained(tokenizer_path, legacy=False)
        else:
            logger.info("use AutoTokenizer tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        # tokenizer = AutoTokenizer.from_pretrained(
        #     tokenizer_name,
        #     *args,
        #     trust_remote_code=trust_remote_code,
        #     tokenizer_revision=tokenizer_revision,
        #     **kwargs)
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer.")
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if (not trust_remote_code and
            ("does not exist or is not currently imported." in str(e)
             or "requires you to execute the tokenizer file" in str(e))):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    except AttributeError as e:
        if "BaichuanTokenizer" in str(e):
            # This is for the error "'BaichuanTokenizer' object has no
            # attribute 'sp_model'".
            tokenizer = BaichuanTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead.")
    return tokenizer


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    output_tokens: List[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if hasattr(tokenizer,'get_added_vocab') and token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        if hasattr(tokenizer,'convert_tokens_to_string'):
            sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        else:
            sub_text = tokenizer.decode_ids(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int = 0,
    read_offset: int = 0,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    if prev_tokens is None:
        if hasattr(tokenizer,'convert_ids_to_tokens'):
            new_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids, skip_special_tokens=skip_special_tokens)
        else:
            new_tokens = [tokenizer.IdToPiece(id_) for id_ in all_input_ids]
        output_tokens = new_tokens
        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_offset = max(len(output_tokens) - 6, 0)
        # If the first new token is a special token, we can't skip 1 extra token
        if skip_special_tokens and hasattr(tokenizer,'all_special_ids') and new_token_id in tokenizer.all_special_ids:
            read_offset = max(len(output_tokens), 0)
        else:
            read_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        if hasattr(tokenizer,'convert_ids_to_tokens'):
            new_tokens = tokenizer.convert_ids_to_tokens(
                [new_token_id], skip_special_tokens=skip_special_tokens)
        else:
            new_tokens = [tokenizer.IdToPiece(id_) for id_ in [new_token_id]]
        output_tokens = prev_tokens + new_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if (hasattr(tokenizer,'is_fast') and tokenizer.is_fast) or (hasattr(tokenizer,'get_added_vocab') and not tokenizer.get_added_vocab()):
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text):]
        return new_tokens, new_text, read_offset, len(output_tokens)
    else:
        return new_tokens, "", prefix_offset, read_offset
