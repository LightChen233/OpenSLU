'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-02-17 19:39:22
Description: Metric calculation class

'''
from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics import f1_score

from common.utils import InputData, OutputData


class Evaluator(object):
    """Evaluation metric funtions library class
        supported metric:
        - slot_f1
        - intent_acc
        - exactly_match_accuracy
        - intent_f1 (defult "macro_intent_f1")
            - macro_intent_f1
            - micro_intent_f1=
    """
    @staticmethod
    def exactly_match_accuracy(pred_slot: List[List[str or int]],
                               real_slot: List[List[str or int]],
                               pred_intent: List[List[str or int] or str or int],
                               real_intent: List[List[str or int] or str or int]) -> float:
        """Compute the accuracy based on the whole predictions of given sentence, including slot and intent.
            (both support str or int index as the representation of slot and intent)
        Args:
            pred_slot (List[List[str or int]]): predicted sequence of slot list
            real_slot (List[List[str or int]]): golden sequence of slot list.
            pred_intent (List[List[str or int] or str or int]): golden intent list / golden multi intent list.
            real_intent (List[List[str or int] or str or int]): predicted intent list / predicted multi intent list.

        Returns:
            float: exactly match accuracy score
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):
            if isinstance(p_intent, list):
                p_intent, r_intent = set(p_intent), set(r_intent)
            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count


    @staticmethod
    def intent_accuracy(pred_list: List, real_list: List) -> float:
        """Get  intent accuracy measured by predictions and ground-trues. Support both multi intent and single intent.

        Args:
            pred_list (List): predicted intent list
            real_list (List): golden intent list

        Returns:
            float: intent accuracy score
        """
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_list, real_list):
            if isinstance(p_intent, list):
                p_intent, r_intent = set(p_intent), set(r_intent)
            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def intent_f1(pred_list: List[List[int]], real_list: List[List[int]], num_intent: int, average='macro') -> float:
        """Get  intent accuracy measured by predictions and ground-trues. Support both multi intent and single intent.
        (Only support multi intent now, but you can use [[intent1], [intent2], ...] to compute intent f1 in single intent)
        Args:
            pred_list (List[List[int]]): predicted multi intent list.
            real_list (List[List[int]]): golden multi intent list.
            num_intent (int)
            average (str): support "micro" and "macro"

        Returns:
            float: intent accuracy score
        """
        return f1_score(Evaluator.__instance2onehot(num_intent, real_list),
                        Evaluator.__instance2onehot(num_intent, pred_list),
                        average=average,
                        zero_division=0)

    @staticmethod
    def __multilabel2one_hot(labels, nums):
        res = [0.] * nums
        if len(labels) == 0:
            return res
        if isinstance(labels[0], list):
            for label in labels[0]:
                res[label] = 1.
            return res
        for label in labels:
            res[label] = 1.
        return res

    @staticmethod
    def __instance2onehot(num_intent, data):
        res = []
        for intents in data:
            res.append(Evaluator.__multilabel2one_hot(intents, num_intent))
        return np.array(res)

    @staticmethod
    def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
        if prevTag == 'B' and tag == 'B':
            chunkStart = True
        if prevTag == 'I' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if prevTag == 'E' and tag == 'E':
            chunkStart = True
        if prevTag == 'E' and tag == 'I':
            chunkStart = True
        if prevTag == 'O' and tag == 'E':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if tag != 'O' and tag != '.' and prevTagType != tagType:
            chunkStart = True
        return chunkStart

    @staticmethod
    def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
        if prevTag == 'B' and tag == 'B':
            chunkEnd = True
        if prevTag == 'B' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'B':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag == 'E' and tag == 'E':
            chunkEnd = True
        if prevTag == 'E' and tag == 'I':
            chunkEnd = True
        if prevTag == 'E' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
            chunkEnd = True
        return chunkEnd

    @staticmethod
    def __splitTagType(tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tagType = ""
        else:
            tag = s[0]
            tagType = s[1]
        return tag, tagType

    @staticmethod
    def computeF1Score(correct_slots: List[List[str]], pred_slots: List[List[str]]) -> float:
        """compute f1 score is modified from conlleval.pl

        Args:
            correct_slots (List[List[str]]): golden slot string list
            pred_slots (List[List[str]]): predicted slot string list

        Returns:
            float: slot f1 score
        """
        correctChunk = {}
        correctChunkCnt = 0.0
        foundCorrect = {}
        foundCorrectCnt = 0.0
        foundPred = {}
        foundPredCnt = 0.0
        correctTags = 0.0
        tokenCount = 0.0
        for correct_slot, pred_slot in zip(correct_slots, pred_slots):
            inCorrect = False
            lastCorrectTag = 'O'
            lastCorrectType = ''
            lastPredTag = 'O'
            lastPredType = ''
            for c, p in zip(correct_slot, pred_slot):
                c = str(c)
                p = str(p)
                correctTag, correctType = Evaluator.__splitTagType(c)
                predTag, predType = Evaluator.__splitTagType(p)

                if inCorrect == True:
                    if Evaluator.__endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                            Evaluator.__endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                            (lastCorrectType == lastPredType):
                        inCorrect = False
                        correctChunkCnt += 1.0
                        if lastCorrectType in correctChunk:
                            correctChunk[lastCorrectType] += 1.0
                        else:
                            correctChunk[lastCorrectType] = 1.0
                    elif Evaluator.__endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                            Evaluator.__endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                            (correctType != predType):
                        inCorrect = False

                if Evaluator.__startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        Evaluator.__startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (correctType == predType):
                    inCorrect = True

                if Evaluator.__startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                    foundCorrectCnt += 1
                    if correctType in foundCorrect:
                        foundCorrect[correctType] += 1.0
                    else:
                        foundCorrect[correctType] = 1.0

                if Evaluator.__startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                    foundPredCnt += 1.0
                    if predType in foundPred:
                        foundPred[predType] += 1.0
                    else:
                        foundPred[predType] = 1.0

                if correctTag == predTag and correctType == predType:
                    correctTags += 1.0

                tokenCount += 1.0

                lastCorrectTag = correctTag
                lastCorrectType = correctType
                lastPredTag = predTag
                lastPredType = predType

            if inCorrect == True:
                correctChunkCnt += 1.0
                if lastCorrectType in correctChunk:
                    correctChunk[lastCorrectType] += 1.0
                else:
                    correctChunk[lastCorrectType] = 1.0

        if foundPredCnt > 0:
            precision = 1.0 * correctChunkCnt / foundPredCnt
        else:
            precision = 0

        if foundCorrectCnt > 0:
            recall = 1.0 * correctChunkCnt / foundCorrectCnt
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0

        return f1

    @staticmethod
    def max_freq_predict(sample):
        """Max frequency prediction.
        """
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def __token_map(indexes, token_label_map):
        return [[token_label_map[idx] if idx in token_label_map else -1 for idx in index] for index in indexes]

    @staticmethod
    def compute_all_metric(inps: InputData,
                           output: OutputData,
                           intent_label_map: dict = None,
                           metric_list: List=None)-> Dict:
        """Auto compute all metric mentioned in 'metric_list'
        
        Args:
            inps (InputData): input golden slot and intent labels
            output (OutputData): output predicted slot and intent labels
            intent_label_map (dict, Optional): dict like {"intent1": 0, "intent2": 1, ...},which aims to map intent string to index
            metric_list (List): support metrics in ["slot_f1", "intent_acc", "intent_f1", "macro_intent_f1", "micro_intent_f1", "EMA"]

        Returns:
            Dict: all metric mentioned in 'metric_list', like {'EMA': 0.7, ...}
            
        
        Example:
            if compute slot metric:
            
                inps.slot = [["slot1", "slot2", ...], ...]; output.slot_ids=[["slot1", "slot2", ...], ...];
                
            if compute intent metric:
            
                [Multi Intent] inps.intent = [["intent1", "intent2", ...], ...]; output.intent_ids = [["intent1", "intent2", ...], ...] 
                
                [Single Intent] inps.intent = ["intent1", ...]; [Single Intent] output.intent_ids = ["intent1", ...]
        """
        if not metric_list:
            metric_list = ["slot_f1", "intent_acc", "EMA"]
        res_dict = {}
        use_slot = output.slot_ids is not None and len(output.slot_ids) > 0
        use_intent = output.intent_ids is not None and len(
            output.intent_ids) > 0
        if use_slot and "slot_f1" in metric_list:
            
            res_dict["slot_f1"] = Evaluator.computeF1Score(
                output.slot_ids, inps.slot)
        if use_intent and "intent_acc" in metric_list:
            res_dict["intent_acc"] = Evaluator.intent_accuracy(
                output.intent_ids, inps.intent)
            if isinstance(output.intent_ids[0], list):
                if "intent_f1" in metric_list:
                    res_dict["intent_f1"] = Evaluator.intent_f1(Evaluator.__token_map(output.intent_ids, intent_label_map),
                                                                Evaluator.__token_map(
                                                                    inps.intent, intent_label_map),
                                                                len(intent_label_map.keys()))
                elif "macro_intent_f1" in metric_list:
                    res_dict["macro_intent_f1"] = Evaluator.intent_f1(Evaluator.__token_map(output.intent_ids, intent_label_map),
                                                                      Evaluator.__token_map(inps.intent, intent_label_map),
                                                                      len(intent_label_map.keys()), average="macro")
                if "micro_intent_f1" in metric_list:
                    res_dict["micro_intent_f1"] = Evaluator.intent_f1(Evaluator.__token_map(output.intent_ids, intent_label_map),
                                                                      Evaluator.__token_map(inps.intent, intent_label_map),
                                                                      len(intent_label_map.keys()), average="micro")

        if use_slot and use_intent and "EMA" in metric_list:
            res_dict["EMA"] = Evaluator.exactly_match_accuracy(output.slot_ids, inps.slot, output.intent_ids,
                                                               inps.intent)
        return res_dict
