from typing import List
import torch

from common import utils
from common.utils import OutputData, InputData
from torch import Tensor

def argmax_for_seq_len(inputs, seq_lens, padding_value=-100):
    packed_inputs = utils.pack_sequence(inputs, seq_lens)
    outputs = torch.argmax(packed_inputs, dim=-1, keepdim=True)
    return utils.unpack_sequence(outputs, seq_lens, padding_value).squeeze(-1)


def decode(output: OutputData,
           target: InputData = None,
           pred_type="slot",
           multi_threshold=0.5,
           ignore_index=-100,
           return_list=True,
           return_sentence_level=True,
           use_multi=False,
           use_crf=False,
           CRF=None) -> List or Tensor:
    """ decode output logits

    Args:
        output (OutputData): output logits data
        target (InputData, optional): input data with attention mask. Defaults to None.
        pred_type (str, optional): prediction type in ["slot", "intent", "token-level-intent"]. Defaults to "slot".
        multi_threshold (float, optional): multi intent decode threshold. Defaults to 0.5.
        ignore_index (int, optional): align and pad token with ignore index. Defaults to -100.
        return_list (bool, optional): if True return list else return torch Tensor. Defaults to True.
        return_sentence_level (bool, optional): if True decode sentence level intent else decode token level intent. Defaults to True.
        use_multi (bool, optional): whether to decode to multi intent. Defaults to False.
        use_crf (bool, optional): whether to use crf. Defaults to False.
        CRF (CRF, optional): CRF function. Defaults to None.

    Returns:
        List or Tensor: decoded sequence ids
    """
    if pred_type == "slot":
        inputs = output.slot_ids
    else:
        inputs = output.intent_ids

    if pred_type == "slot":
        if not use_multi:
            if use_crf:
                res = CRF.decode(inputs, mask=target.attention_mask)
            else:
                res = torch.argmax(inputs, dim=-1)
        else:
            raise NotImplementedError("Multi-slot prediction is not supported.")
    elif pred_type == "intent":
        if not use_multi:
            res = torch.argmax(inputs, dim=-1)
        else:
            res = (torch.sigmoid(inputs) > multi_threshold).nonzero()
            if return_list:
                res_index = res.detach().cpu().tolist()
                res_list = [[] for _ in range(len(target.seq_lens))]
                for item in res_index:
                    res_list[item[0]].append(item[1])
                return res_list
            else:
                return res
    elif pred_type == "token-level-intent":
        if not use_multi:
            res = torch.argmax(inputs, dim=-1)
            if not return_sentence_level:
                return res
            if return_list:
                res = res.detach().cpu().tolist()
            attention_mask = target.attention_mask
            for i in range(attention_mask.shape[0]):
                temp = []
                for j in range(attention_mask.shape[1]):
                    if attention_mask[i][j] == 1:
                        temp.append(res[i][j])
                    else:
                        break
                res[i] = temp
            return [max(it, key=lambda v: it.count(v)) for it in res]
        else:
            seq_lens = target.seq_lens

            if not return_sentence_level:
                token_res = torch.cat([
                    torch.sigmoid(inputs[i, 0:seq_lens[i], :]) > multi_threshold
                    for i in range(len(seq_lens))],
                    dim=0)
                return utils.unpack_sequence(token_res, seq_lens, padding_value=ignore_index)

            intent_index_sum = torch.cat([
                torch.sum(torch.sigmoid(inputs[i, 0:seq_lens[i], :]) > multi_threshold, dim=0).unsqueeze(0)
                for i in range(len(seq_lens))],
                dim=0)

            res = (intent_index_sum > torch.div(seq_lens, 2, rounding_mode='floor').unsqueeze(1)).nonzero()
            if return_list:
                res_index = res.detach().cpu().tolist()
                res_list = [[] for _ in range(len(seq_lens))]
                for item in res_index:
                    res_list[item[0]].append(item[1])
                return res_list
            else:
                return res
    else:
        raise NotImplementedError("Prediction mode except ['slot','intent','token-level-intent'] is not supported.")
    if return_list:
        res = res.detach().cpu().tolist()
    return res


def compute_loss(pred: OutputData,
                 target: InputData,
                 criterion_type="slot",
                 use_crf=False,
                 ignore_index=-100,
                 loss_fn=None,
                 use_multi=False,
                 CRF=None):
    """ compute loss

    Args:
        pred (OutputData): output logits data
        target (InputData): input golden data
        criterion_type (str, optional): criterion type in ["slot", "intent", "token-level-intent"]. Defaults to "slot".
        ignore_index (int, optional): compute loss with ignore index. Defaults to -100.
        loss_fn (_type_, optional): loss function. Defaults to None.
        use_crf (bool, optional): whether to use crf. Defaults to False.
        CRF (CRF, optional): CRF function. Defaults to None.

    Returns:
        Tensor: loss result
    """
    if criterion_type == "slot":
        if use_crf:
            return -1 * CRF(pred.slot_ids, target.slot, target.get_slot_mask(ignore_index).byte())
        else:
            pred_slot = utils.pack_sequence(pred.slot_ids, target.seq_lens)
            target_slot = utils.pack_sequence(target.slot, target.seq_lens)
            return loss_fn(pred_slot, target_slot)
    elif criterion_type == "token-level-intent":
        # TODO: Two decode function
        intent_target = target.intent.unsqueeze(1)
        if not use_multi:
            intent_target = intent_target.repeat(1, pred.intent_ids.shape[1])
        else:
            intent_target = intent_target.repeat(1, pred.intent_ids.shape[1], 1)
        intent_pred = utils.pack_sequence(pred.intent_ids, target.seq_lens)
        intent_target = utils.pack_sequence(intent_target, target.seq_lens)
        return loss_fn(intent_pred, intent_target)
    else:
        return loss_fn(pred.intent_ids, target.intent)
