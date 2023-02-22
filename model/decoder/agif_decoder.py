from common.utils import HiddenData, OutputData
from model.decoder.base_decoder import BaseDecoder


class AGIFDecoder(BaseDecoder):
    def forward(self, hidden: HiddenData, **kwargs):
        # hidden = self.interaction(hidden)
        pred_intent = self.intent_classifier(hidden)
        intent_index = self.intent_classifier.decode(OutputData(pred_intent, None),
                                                     return_list=False,
                                                     return_sentence_level=True)
        interact_args = {"intent_index": intent_index,
                         "batch_size": pred_intent.classifier_output.shape[0],
                         "intent_label_num": self.intent_classifier.config["intent_label_num"]}
        pred_slot = self.slot_classifier(hidden, internal_interaction=self.interaction, **interact_args)
        return OutputData(pred_intent, pred_slot)
