from typing import List, Optional, Union
from unittest.mock import MagicMock

import pytest
from transformers import PreTrainedTokenizer

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob, Sequence, SequenceStatus


def create_running_sequence(
    text_wo_last_token: str,
    last_token: str,
    last_token_id: int,
    is_last_token_eos: bool,
) -> Sequence:
    """
    Create a dummy Sequence that ends with an specified token.
    """

    seq = Sequence(
        seq_id=0,
        inputs={
            "prompt": "",
            "prompt_token_ids": [],
            "multi_modal_data": None,
        },
        block_size=16,
        eos_token_id=last_token_id if is_last_token_eos else None,
    )
    seq.output_text = text_wo_last_token + last_token

    offset = last_token_id + 1
    for i in range(offset, len(text_wo_last_token) + offset):
        seq.append_token_id(token_id=i, logprobs={i: Logprob(0.0)})
    seq.append_token_id(token_id=last_token_id,
                        logprobs={last_token_id: Logprob(0.0)})

    seq.status = SequenceStatus.RUNNING

    return seq


@pytest.fixture
def stop_checker():
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    get_tokenizer_for_seq = MagicMock(return_value=tokenizer)
    stop_checker = StopChecker(max_model_len=1024,
                               get_tokenizer_for_seq=get_tokenizer_for_seq)
    return stop_checker


@pytest.mark.parametrize(["text_wo_eos", "eos_token", "eos_token_id"], [
    ("This text ends with EOS token", "</s>", 2),
])
@pytest.mark.skip_global_cleanup
def test_min_tokens_not_met(stop_checker: StopChecker, text_wo_eos: str,
                            eos_token: str, eos_token_id: int):
    """
    Test that the sequence does not stop when the number of generated tokens is
    less than min_tokens even if the text ends with EOS token.
    """
    seq = create_running_sequence(
        text_wo_last_token=text_wo_eos,
        last_token=eos_token,
        last_token_id=eos_token_id,
        is_last_token_eos=True,
    )
    new_char_count = len(eos_token)

    min_tokens = len(seq.output_text) + 1
    max_tokens = min_tokens + 1

    sampling_params = SamplingParams(min_tokens=min_tokens,
                                     max_tokens=max_tokens)

    stop_checker.maybe_stop_sequence(
        seq=seq,
        new_char_count=new_char_count,
        sampling_params=sampling_params,
    )

    assert seq.status == SequenceStatus.RUNNING
    assert seq.output_text == text_wo_eos + eos_token


@pytest.mark.parametrize(["text_wo_eos", "eos_token", "eos_token_id"], [
    ("This text ends with EOS token", "</s>", 2),
])
@pytest.mark.parametrize("ignore_eos", [True, False, None])
@pytest.mark.parametrize("include_stop_str_in_output", [True, False, None])
@pytest.mark.skip_global_cleanup
def test_on_eos_token(stop_checker: StopChecker, text_wo_eos: str,
                      eos_token: str, eos_token_id: int,
                      ignore_eos: Optional[bool],
                      include_stop_str_in_output: Optional[bool]):
    """
    Test the behavior of the StopChecker's maybe_stop_sequence method
    when an EOS token is encountered.

    This test covers:
    - When the EOS token should stop the sequence and be removed from the output
    - When the EOS token should stop the sequence and be included in the output
    - When the EOS token should be ignored, and the sequence continues
    """

    seq = create_running_sequence(
        text_wo_last_token=text_wo_eos,
        last_token=eos_token,
        last_token_id=eos_token_id,
        is_last_token_eos=True,
    )
    new_char_count = len(eos_token)

    # Note that `stop` and `stop_token_ids` are not specified
    sampling_params = SamplingParams(
        min_tokens=1,
        max_tokens=1024,
        ignore_eos=ignore_eos,
        include_stop_str_in_output=include_stop_str_in_output)

    stop_checker.maybe_stop_sequence(
        seq=seq,
        new_char_count=new_char_count,
        sampling_params=sampling_params,
    )

    if ignore_eos:
        assert seq.status == SequenceStatus.RUNNING
        assert seq.output_text == text_wo_eos + eos_token
    elif include_stop_str_in_output:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_eos + eos_token
    else:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_eos


@pytest.mark.parametrize(["text_wo_last_token", "last_token", "last_token_id"],
                         [
                             ("This text ends with a non-EOS token", ".", 100),
                         ])
@pytest.mark.parametrize("stop_token_ids", [
    [100, 101, 102],
    [200, 201, 202],
    None,
])
@pytest.mark.parametrize("include_stop_str_in_output", [True, False, None])
@pytest.mark.skip_global_cleanup
def test_on_stop_token_id(stop_checker: StopChecker, text_wo_last_token: str,
                          last_token: str, last_token_id: int,
                          stop_token_ids: Optional[List[str]],
                          include_stop_str_in_output: Optional[bool]):
    """
    Test the behavior of the StopChecker's maybe_stop_sequence method
    when an token whose id is in the stop_token_ids is encountered.

    This test covers:
    - When the token should stop the sequence and be removed from the output
    - When the token should stop the sequence and be included in the output
    - When the token should be ignored, and the sequence continues
    """

    seq = create_running_sequence(text_wo_last_token=text_wo_last_token,
                                  last_token=last_token,
                                  last_token_id=last_token_id,
                                  is_last_token_eos=False)
    new_char_count = len(last_token)

    sampling_params = SamplingParams(
        min_tokens=1,
        max_tokens=1024,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=include_stop_str_in_output)

    stop_checker.maybe_stop_sequence(
        seq=seq,
        new_char_count=new_char_count,
        sampling_params=sampling_params,
    )

    if stop_token_ids is None or last_token_id not in stop_token_ids:
        assert seq.status == SequenceStatus.RUNNING
        assert seq.output_text == text_wo_last_token + last_token
    elif include_stop_str_in_output:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_last_token + last_token
    else:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_last_token


@pytest.mark.parametrize(["text_wo_last_token", "last_token", "last_token_id"],
                         [
                             ("This text ends with a non-EOS token", ".", 100),
                         ])
@pytest.mark.parametrize("stop", [
    ".",
    ",",
    [".", ","],
    ["!", "?"],
    None,
])
@pytest.mark.parametrize("include_stop_str_in_output", [True, False, None])
@pytest.mark.skip_global_cleanup
def test_on_stop_token(stop_checker: StopChecker, text_wo_last_token: str,
                       last_token: str, last_token_id: int,
                       stop: Optional[Union[str, List[str]]],
                       include_stop_str_in_output: Optional[bool]):
    """
    Test the behavior of the StopChecker's maybe_stop_sequence method
    when an stop token is encountered.

    This test covers:
    - When the token should stop the sequence and be removed from the output
    - When the token should stop the sequence and be included in the output
    - When the token should be ignored, and the sequence continues
    """

    seq = create_running_sequence(text_wo_last_token=text_wo_last_token,
                                  last_token=last_token,
                                  last_token_id=last_token_id,
                                  is_last_token_eos=False)
    new_char_count = len(last_token)

    sampling_params = SamplingParams(
        min_tokens=1,
        max_tokens=1024,
        stop=stop,
        include_stop_str_in_output=include_stop_str_in_output)

    stop_checker.maybe_stop_sequence(
        seq=seq,
        new_char_count=new_char_count,
        sampling_params=sampling_params,
    )

    if stop is None or (isinstance(stop, str) and stop != last_token) or (
            isinstance(stop, list) and last_token not in stop):
        assert seq.status == SequenceStatus.RUNNING
        assert seq.output_text == text_wo_last_token + last_token
    elif include_stop_str_in_output:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_last_token + last_token
    else:
        assert seq.status == SequenceStatus.FINISHED_STOPPED
        assert seq.output_text == text_wo_last_token
