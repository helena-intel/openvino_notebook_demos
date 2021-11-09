import logging as log
import os
import sys

import numpy as np
from openvino.inference_engine import IECore

open_model_zoo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))
sys.path.append(os.path.join(open_model_zoo_path, "demos", "common", "python"))

from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs

INPUT_NAMES = {
    "bert-large-uncased-whole-word-masking-squad-0001": ["0", "1", "2"],
    "bert-large-uncased-whole-word-masking-squad-int8-0001": ["result.1", "result.2", "result.3"],
    "bert-small-uncased-whole-word-masking-squad-0001": ["input_ids", "attention_mask", "token_type_ids"],
    "bert-small-uncased-whole-word-masking-squad-0002": [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    ],
    "bert-small-uncased-whole-word-masking-squad-int8-0002": [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    ],
}

OUTPUT_NAMES = {
    "bert-large-uncased-whole-word-masking-squad-0001": ["3171", "3172"],
    "bert-large-uncased-whole-word-masking-squad-int8-0001": ["5211", "5212"],
    "bert-small-uncased-whole-word-masking-squad-0001": ["output_s", "output_e"],
    "bert-small-uncased-whole-word-masking-squad-0002": ["output_s", "output_e"],
    "bert-small-uncased-whole-word-masking-squad-int8-0002": ["output_s", "output_e"],
}


# return entire sentence as start-end positions for a given answer (within the sentence).
def find_sentence_range(context, s, e):
    # find start of sentence
    for c_s in range(s, max(-1, s - 200), -1):
        if context[c_s] in "\n.":
            c_s += 1
            break

    # find end of sentence
    for c_e in range(max(0, e - 1), min(len(context), e + 200), +1):
        if context[c_e] in "\n.":
            break

    return c_s, c_e


class BERT(object):
    def __init__(self, input_url, vocab_file, model_name, base_model_dir, reshape, device, model_squad_ver):
        self._input_url = input_url
        self.vocab_file = vocab_file
        self._model_name = model_name
        self.base_model_dir = base_model_dir
        self.reshape = reshape
        self.device = device
        self.model_squad_ver = model_squad_ver

        self.max_question_token_num = 8
        self.max_answer_token_num = 15

        self.load_context()
        self.load_model()

    @property
    def input_names(self):
        return INPUT_NAMES[self.model_name]

    @property
    def output_names(self):
        return OUTPUT_NAMES[self.model_name]

    @property
    def input_url(self):
        return self._input_url

    @property
    def model_name(self):
        return self._model_name

    def set_input_url(self, url):
        self._input_url = url
        self.load_context()
        if self.reshape:
            self.load_model()

    def set_model_name(self, model_name):
        if model_name not in INPUT_NAMES:
            raise ValueError(
                f"Model `{model_name}` is not supported. Supported models are: {', '.join(list(INPUT_NAMES.keys()))}."
            )
        self._model_name = model_name
        self.load_model()

    def load_context(self):
        # get context as a string (as we might need it's length for the sequence reshape)
        # for this demo we only accept one URL as input
        input_urls = [
            self._input_url,
        ]
        paragraphs = get_paragraphs(input_urls)
        context = "\n".join(paragraphs)
        self.context = context
        self.vocab = load_vocab_file(self.vocab_file)
        self.c_tokens_id, self.c_tokens_se = text_to_tokens(self.context.lower(), self.vocab)

    def load_model(self):
        ie = IECore()
        # read IR
        precision_str = "FP16-INT8" if "int8" in self.model_name else "FP16"
        base_model_path = os.path.join(self.base_model_dir, "intel", self.model_name, precision_str, self.model_name)
        # print(f"Loading model from {base_model_path}")
        model_xml = base_model_path + ".xml"
        model_bin = base_model_path + ".bin"
        ie_encoder = ie.read_network(model=model_xml, weights=model_bin)
        # load model to the device

        if self.reshape:
            first_input_layer = next(iter(ie_encoder.input_info))
            c = ie_encoder.input_info[first_input_layer].input_data.shape[1]
            # find the closest multiple of 64, if it is smaller than current network's sequence length, let' use that
            seq = min(c, int(np.ceil((len(self.c_tokens_id) + self.max_question_token_num) / 64) * 64))
            if seq < c:
                input_info = list(self.input_info)
                new_shapes = dict([])
                for i in input_info:
                    n, c = ie_encoder.input_info[i].input_data.shape
                    new_shapes[i] = [n, seq]
                    print(
                        "Reshaped input {} from {} to the {}".format(
                            i, ie_encoder.input_info[i].input_data.shape, new_shapes[i]
                        )
                    )
                print("Attempting to reshape the network to the modified inputs...")
                try:
                    ie_encoder.reshape(new_shapes)
                    print("Successful!")
                except RuntimeError:
                    print("Failed to reshape the network, please set the `reshape` setting to False")
            else:
                print(
                    "Skipping network reshaping,"
                    " as (context length + max question length) exceeds the current (input) network sequence length"
                )

        self.ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=self.device)

        # check input and output names
        if self.ie_encoder_exec.input_info.keys() != set(
            self.input_names
        ) or self.ie_encoder_exec.outputs.keys() != set(self.output_names):
            log.error("Input or Output names do not match")
            log.error(
                "    The demo expects input->output names: {}->{}. "
                "Please use the --input_names and --output_names to specify the right names "
                "(see actual values below)".format(self.input_names, self.output_names)
            )
            log.error(
                "    Actual network input->output names: {}->{}".format(
                    list(self.ie_encoder_exec.input_info.keys()), list(self.ie_encoder_exec.outputs.keys())
                )
            )
            raise Exception("Unexpected network input or output names")




    def ask(self, questions, show_context=True, show_answers=True):
        if isinstance(questions, str):
            questions = [
                questions,
            ]

        COLOR_BLUE = "\033[94m"
        COLOR_RED = "\033[91m"
        COLOR_RESET = "\033[0m"

        # loop over questions
        for question in questions:
            q_tokens_id, _ = text_to_tokens(question.lower(), self.vocab)

            # maximum number of tokens that can be processed by network at once
            max_length = self.ie_encoder_exec.input_info[self.input_names[0]].input_data.shape[1]

            # calculate number of tokens for context in each inference request.
            # reserve 3 positions for special tokens
            # [CLS] q_tokens [SEP] c_tokens [SEP]
            c_wnd_len = max_length - (len(q_tokens_id) + 3)

            # token num between two neighbour context windows
            # 1/2 means that context windows are overlapped by half
            c_stride = c_wnd_len // 2

            # array of answers from each window
            answers = []

            # init a window to iterate over context
            c_s, c_e = 0, min(c_wnd_len, len(self.c_tokens_id))

            # iterate while context window is not empty
            while c_e > c_s:
                # form the request
                tok_cls = self.vocab["[CLS]"]
                tok_sep = self.vocab["[SEP]"]
                input_ids = [tok_cls] + q_tokens_id + [tok_sep] + self.c_tokens_id[c_s:c_e] + [tok_sep]
                token_type_ids = [0] + [0] * len(q_tokens_id) + [0] + [1] * (c_e - c_s) + [0]
                attention_mask = [1] * len(input_ids)

                # pad the rest of the request
                pad_len = max_length - len(input_ids)
                input_ids += [0] * pad_len
                token_type_ids += [0] * pad_len
                attention_mask += [0] * pad_len

                # create numpy inputs for IE
                inputs = {
                    self.input_names[0]: np.array([input_ids], dtype=np.int32),
                    self.input_names[1]: np.array([attention_mask], dtype=np.int32),
                    self.input_names[2]: np.array([token_type_ids], dtype=np.int32),
                }
                if len(self.input_names) > 3:
                    inputs[self.input_names[3]] = np.arange(len(input_ids), dtype=np.int32)[None, :]

                # infer by IE
                res = self.ie_encoder_exec.infer(inputs=inputs)

                # get start-end scores for context
                def get_score(name):
                    out = np.exp(res[name].reshape((max_length,)))
                    return out / out.sum(axis=-1)

                score_s = get_score(self.output_names[0])
                score_e = get_score(self.output_names[1])

                # get 'no-answer' score (not valid if model has been fine-tuned on squad1.x)
                if self.model_squad_ver.split(".")[0] == "1":
                    score_na = 0
                else:
                    score_na = score_s[0] * score_e[0]

                # find product of all start-end combinations to find the best one
                c_s_idx = len(q_tokens_id) + 2  # index of first context token in tensor
                c_e_idx = max_length - (1 + pad_len)  # index of last+1 context token in tensor
                score_mat = np.matmul(
                    score_s[c_s_idx:c_e_idx].reshape((c_e - c_s, 1)), score_e[c_s_idx:c_e_idx].reshape((1, c_e - c_s))
                )
                # reset candidates with end before start
                score_mat = np.triu(score_mat)
                # reset long candidates (>max_answer_token_num)
                score_mat = np.tril(score_mat, self.max_answer_token_num - 1)
                # find the best start-end pair
                max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
                max_score = score_mat[max_s, max_e] * (1 - score_na)

                # convert to context text start-end index
                max_s = self.c_tokens_se[c_s + max_s][0]
                max_e = self.c_tokens_se[c_s + max_e][1]

                # check that answers list does not have duplicates (because of context windows overlapping)
                same = [i for i, a in enumerate(answers) if a[1] == max_s and a[2] == max_e]
                if same:
                    assert len(same) == 1
                    # update existing answer record
                    a = answers[same[0]]
                    answers[same[0]] = (max(max_score, a[0]), max_s, max_e)
                else:
                    # add new record
                    answers.append((max_score, max_s, max_e))

                # check that context window reached the end
                if c_e == len(self.c_tokens_id):
                    break

                # move to next window position
                c_s = min(c_s + c_stride, len(self.c_tokens_id))
                c_e = min(c_s + c_wnd_len, len(self.c_tokens_id))

            # print top 3 results
            if show_context or show_answers:
                answers = sorted(answers, key=lambda x: -x[0])
                print(COLOR_BLUE + question + COLOR_RESET)

                for score, s, e in answers[:3]:
                    if show_answers:
                        print(f"---answer (score: {score:.2f}): {self.context[s:e]}")
                    if show_context:
                        c_s, c_e = find_sentence_range(self.context, s, e)
                        print(
                            "   "
                            + self.context[c_s:s]
                            + COLOR_RED
                            + self.context[s:e]
                            + COLOR_RESET
                            + self.context[e:c_e]
                        )
                print()
