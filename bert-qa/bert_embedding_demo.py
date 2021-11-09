import logging as log
import os
import sys
import time

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
    "bert-small-uncased-whole-word-masking-squad-emb-int8-0001": [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    ],
    "bert-large-uncased-whole-word-masking-squad-emb-0001": [
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
    "bert-small-uncased-whole-word-masking-squad-emb-int8-0001": ["embedding"],
    "bert-large-uncased-whole-word-masking-squad-emb-0001": ["embedding"],
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
    def __init__(
        self, input_url, vocab_file, model_name_emb, model_name_qa, base_model_dir, reshape, device, model_squad_ver
    ):
        self.input_url = input_url
        self.vocab_file = vocab_file
        self._model_name = model_name_emb
        self._model_name_qa = model_name_qa
        self.base_model_dir = base_model_dir
        self.reshape = reshape
        #         self.input_names = input_names
        #         self.output_names = output_names
        self.device = device
        self.model_squad_ver = model_squad_ver

        self.max_question_token_num = 8
        self.max_answer_token_num = 15
        self.best_n = 10

        self.max_length_c = 384
        self.max_length_q = 32
        self.load_model()
        if model_name_qa is not None:
            self.load_model_qa()

        self.load_context()

    @property
    def input_names(self):
        return INPUT_NAMES[self.model_name]

    @property
    def output_names(self):
        return OUTPUT_NAMES[self.model_name]

    @property
    def input_names_qa(self):
        return INPUT_NAMES[self.model_name_qa]

    @property
    def output_names_qa(self):
        return OUTPUT_NAMES[self.model_name_qa]

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_name_qa(self):
        return self._model_name_qa

    def set_model_name_qa(self, model_name):
        self._model_name_qa = model_name
        self.load_model_qa()

    def set_model_name(self, model_name):
        self._model_name = model_name
        self.load_model()

    def set_input_url(self, url):
        self._input_url = url
        self.load_context()
        if self.reshape:
            self.load_model()

    def load_model(self):
        ie = IECore()
        # read IR
        precision_str = "FP16-INT8" if "int8" in self.model_name else "FP16"
        base_model_path = os.path.join(self.base_model_dir, "intel", self.model_name, precision_str, self.model_name)
        # print(f"Loading model from {base_model_path}")
        model_xml = base_model_path + ".xml"
        model_bin = base_model_path + ".bin"
        ie_encoder_emb = ie.read_network(model=model_xml, weights=model_bin)
        # load model to the device

        self.ie_encoder_exec_emb_dict = {}

        for length in [self.max_length_q, self.max_length_c]:
            new_shapes = {}
            for i, input_info in ie_encoder_emb.input_info.items():
                new_shapes[i] = [1, length]
                log.info("Reshaped input {} from {} to the {}".format(i, input_info.input_data.shape, new_shapes[i]))
            log.info("Attempting to reshape the context embedding network to the modified inputs...")

            try:
                ie_encoder_emb.reshape(new_shapes)
                log.info("Successful!")
            except RuntimeError:
                log.error("Failed to reshape the embedding network")
                raise

            # Loading model to the plugin
            log.info("Loading model to the plugin")
            self.ie_encoder_exec_emb_dict[length] = ie.load_network(network=ie_encoder_emb, device_name=self.device)

    def load_model_qa(self):
        ie = IECore()
        # read IR
        precision_str = "FP16-INT8" if "int8" in self.model_name_qa else "FP16"
        base_model_path = os.path.join(
            self.base_model_dir, "intel", self.model_name_qa, precision_str, self.model_name_qa
        )
        # print(f"Loading model from {base_model_path}")
        model_xml = base_model_path + ".xml"
        model_bin = base_model_path + ".bin"
        self.ie_encoder_qa = ie.read_network(model=model_xml, weights=model_bin)
        # load model to the device
        self.ie_encoder_exec_qa = ie.load_network(network=self.ie_encoder_qa, device_name=self.device)

    def load_context(self):

        # small class to store context as text and tokens and its embedding vector
        outerself = self

        class ContextData:
            def __init__(self, context, c_tokens_id, c_tokens_se):
                self.context = context
                self.c_tokens_id = c_tokens_id
                self.c_tokens_se = c_tokens_se

                max_length_c = 384
                self.c_emb = outerself.calc_emb(self.c_tokens_id, max_length_c)

        # load vocabulary file for all models
        log.info("Loading vocab file:\t{}".format(self.vocab_file))
        self.vocab = load_vocab_file(self.vocab_file)
        log.info("{} tokens loaded".format(len(self.vocab)))

        self.contexts_all = []

        input_urls = [
            self.input_url,
        ]
        paragraphs = get_paragraphs(input_urls)
        log.info("Indexing {} paragraphs...".format(len(paragraphs)))
        for par in paragraphs:
            c_tokens_id, c_tokens_se = text_to_tokens(par.lower(), self.vocab)
            if not c_tokens_id:
                continue

            # get context as string and then encode it into token id list
            # calculate number of tokens for context in each request.
            # reserve 3 positions for special tokens
            # [CLS] q_tokens [SEP] c_tokens [SEP]
            if self.model_name_qa is not None:
                # to make context be able to pass model_qa together with question
                self.max_length_qc = self.ie_encoder_qa.input_info[self.input_names_qa[0]].input_data.shape[1]
                c_wnd_len = self.max_length_qc - (self.max_length_q + 3)
            else:
                # to make context be able to pass model_emb without question
                c_wnd_len = self.max_length_c - 2

            # token num between 2 neighbours context windows
            # 1/2 means that context windows are interleaved by half
            c_stride = c_wnd_len // 2

            # init scan window
            c_s, c_e = 0, min(c_wnd_len, len(c_tokens_id))

            # iterate while context window is not empty
            while c_e > c_s:
                self.contexts_all.append(ContextData(par, c_tokens_id[c_s:c_e], c_tokens_se[c_s:c_e]))

                # check that context window reach the end
                if c_e == len(c_tokens_id):
                    break

                # move to next window position
                c_s, c_e = c_s + c_stride, c_e + c_stride

                shift_left = max(0, c_e - len(c_tokens_id))
                c_s, c_e = c_s - shift_left, c_e - shift_left
                assert (
                    c_s >= 0
                ), "start can be left of 0 only with window less than len but in this case we can not be here"

    # define function to infer embedding
    def calc_emb(self, tokens_id, max_length):
        num = min(max_length - 2, len(tokens_id))

        # forms the request
        pad_len = max_length - num - 2
        tok_cls = [self.vocab["[CLS]"]]
        tok_sep = [self.vocab["[SEP]"]]
        tok_pad = [self.vocab["[PAD]"]]

        dtype = np.int32
        inputs = {
            self.input_names[0]: np.array([tok_cls + tokens_id[:num] + tok_sep + tok_pad * pad_len], dtype=dtype),
            self.input_names[1]: np.array([[1] + [1] * num + [1] + [0] * pad_len], dtype=dtype),
            self.input_names[2]: np.array([[0] + [0] * num + [0] + tok_pad * pad_len], dtype=dtype),
            self.input_names[3]: np.arange(max_length, dtype=dtype)[None, :],
        }

        # calc embedding
        ie_encoder_exec_emb = self.ie_encoder_exec_emb_dict[max_length]

        t_start = time.perf_counter()
        res = ie_encoder_exec_emb.infer(inputs=inputs)
        t_end = time.perf_counter()
        log.info(
            "embedding calculated for sequence of length {} with {:0.2f} requests/sec ({:0.2} sec per request)".format(
                max_length, 1 / (t_end - t_start), t_end - t_start
            )
        )

        res = res[self.output_names[0]]
        return res.squeeze(0)

    def ask(self, questions, show_embeddings=True, show_context=True, show_answers=True):
        if isinstance(questions, str):
            questions = [
                questions,
            ]

        COLOR_BLUE = "\033[94m"
        COLOR_RED = "\033[91m"
        COLOR_MAGENTA = "\033[95m"
        COLOR_RESET = "\033[0m"

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.WARNING, stream=sys.stdout)

        # loop on user's or prepared questions
        for question in questions:
            if not question.strip():
                break

            if show_answers or show_embeddings or show_context:
                print(COLOR_BLUE + question + COLOR_RESET)

            q_tokens_id, _ = text_to_tokens(question.lower(), self.vocab)

            q_emb = self.calc_emb(q_tokens_id, self.max_length_q)
            distances = [(np.linalg.norm(c.c_emb - q_emb, 2), c) for c in self.contexts_all]
            distances.sort(key=lambda x: x[0])
            keep_num = min(self.best_n, len(distances))
            distances_filtered = distances[:keep_num]

            # print short list
            if show_embeddings:
                print(COLOR_MAGENTA + "Contexts from embeddings: " + COLOR_RESET)
                for i, (dist, c_data) in enumerate(distances_filtered):
                    print(f"#{i+1}: embedding distance {dist:.2f}")
                    print(f"{c_data.context}")

            # run model_qa if available to find exact answer to question in filtered in contexts
            if self.model_name_qa is not None:

                # array of answers from each context_data
                answers = []

                for dist, c_data in distances_filtered:
                    # forms the request
                    tok_cls = [self.vocab["[CLS]"]]
                    tok_sep = [self.vocab["[SEP]"]]
                    tok_pad = [self.vocab["[PAD]"]]
                    req_len = len(q_tokens_id) + len(c_data.c_tokens_id) + 3
                    pad_len = self.max_length_qc - req_len
                    assert pad_len >= 0

                    input_ids = tok_cls + q_tokens_id + tok_sep + c_data.c_tokens_id + tok_sep + tok_pad * pad_len
                    token_type_ids = (
                        [0] * (len(q_tokens_id) + 2) + [1] * (len(c_data.c_tokens_id) + 1) + tok_pad * pad_len
                    )
                    attention_mask = [1] * req_len + [0] * pad_len

                    # create numpy inputs for IE
                    inputs = {
                        self.input_names_qa[0]: np.array([input_ids], dtype=np.int32),
                        self.input_names_qa[1]: np.array([attention_mask], dtype=np.int32),
                        self.input_names_qa[2]: np.array([token_type_ids], dtype=np.int32),
                    }
                    if len(self.input_names_qa) > 3:
                        inputs["position_ids"] = np.arange(self.max_length_qc, dtype=np.int32)[None, :]

                    # infer by IE
                    t_start = time.perf_counter()
                    res = self.ie_encoder_exec_qa.infer(inputs=inputs)
                    t_end = time.perf_counter()
                    log.info(
                        "Exact answer calculated for sequence of length {} with {:0.2f} requests/sec ({:0.2} sec per request)".format(
                            self.max_length_qc, 1 / (t_end - t_start), t_end - t_start
                        )
                    )

                    # get start-end scores for context
                    def get_score(name):
                        out = np.exp(res[name].reshape((self.max_length_qc,)))
                        return out / out.sum(axis=-1)

                    score_s = get_score(self.output_names_qa[0])
                    score_e = get_score(self.output_names_qa[1])

                    # find product of all start-end combinations to find the best one
                    c_s_idx = len(q_tokens_id) + 2  # index of first context token in tensor
                    c_e_idx = self.max_length_qc - (1 + pad_len)  # index of last+1 context token in tensor
                    score_mat = np.matmul(
                        score_s[c_s_idx:c_e_idx].reshape((len(c_data.c_tokens_id), 1)),
                        score_e[c_s_idx:c_e_idx].reshape((1, len(c_data.c_tokens_id))),
                    )
                    # reset candidates with end before start
                    score_mat = np.triu(score_mat)
                    # reset long candidates (>max_answer_token_num)
                    score_mat = np.tril(score_mat, self.max_answer_token_num - 1)
                    # find the best start-end pair
                    max_s, max_e = divmod(score_mat.flatten().argmax(), score_mat.shape[1])
                    max_score = score_mat[max_s, max_e]

                    # convert to context text start-end index
                    max_s = c_data.c_tokens_se[max_s][0]
                    max_e = c_data.c_tokens_se[max_e][1]

                    # check that answers list does not have answer yet
                    # it could be because of context windows overlapping
                    same = [
                        i for i, a in enumerate(answers) if a[1] == max_s and a[2] == max_e and a[3] is c_data.context
                    ]
                    if same:
                        assert len(same) == 1
                        # update exist answer record
                        a = answers[same[0]]
                        answers[same[0]] = (max(max_score, a[0]), max_s, max_e, c_data.context)
                    else:
                        # add new record
                        answers.append((max_score, max_s, max_e, c_data.context))

                def mark(txt):
                    return "\033[91m" + txt + "\033[0m"

                # print top 3 results
                answers.sort(key=lambda x: -x[0])
                log.info("---Stage 3---Find best 3 answers from {} results of Stage 1".format(len(answers)))
                # if show_results:
                #    for score, s, e, context in answers[:3]:
                #        print("Answer (score: {:0.2f}): {}".format(score, mark(context[s:e])))
                #        print(context[:s] + mark(context[s:e]) + context[e:])

                if show_answers or show_context:
                    if show_embeddings:
                        # Embeddings can take up a bit of space. If they are shown, show the question again, and an "Answer" heading.
                        print(COLOR_BLUE + question + COLOR_RESET)
                        print(COLOR_MAGENTA + "Answers:" + COLOR_RESET)
                    for score, s, e, context in answers[:3]:
                        if show_answers:
                            print(f"---answer (score: {score:.2f}): {context[s:e]}")
                        if show_context:
                            c_s, c_e = find_sentence_range(context, s, e)
                            print("   " + context[c_s:s] + COLOR_RED + context[s:e] + COLOR_RESET + context[e:c_e])
                    print()
