import copy
import os
from collections import defaultdict
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str


char_dict = defaultdict(int)
base_dict = {'"': 61,
             '#': 145,
             '$': 43,
             '%': 106,
             '&': 149,
             "'": 133,
             '(': 22,
             ')': 27,
             '+': 118,
             ',': 12,
             '-': 65,
             '.': 31,
             '/': 96,
             '0': 45,
             '1': 44,
             '2': 48,
             '3': 60,
             '4': 49,
             '5': 30,
             '6': 25,
             '7': 56,
             '8': 23,
             '9': 24,
             ':': 75,
             ';': 154,
             '<PAD>': 0,
             '<UNK>': 1,
             '?': 104,
             'A': 76,
             'B': 108,
             'C': 101,
             'D': 111,
             'E': 107,
             'F': 98,
             'G': 129,
             'H': 123,
             'I': 109,
             'J': 122,
             'K': 138,
             'L': 119,
             'M': 80,
             'N': 67,
             'O': 134,
             'P': 102,
             'Q': 146,
             'R': 85,
             'S': 71,
             'T': 110,
             'U': 137,
             'V': 132,
             'W': 121,
             'X': 152,
             'Y': 125,
             'Z': 143,
             '_': 136,
             'a': 73,
             'b': 99,
             'c': 82,
             'd': 70,
             'e': 86,
             'f': 84,
             'g': 120,
             'h': 94,
             'i': 81,
             'j': 144,
             'k': 100,
             'l': 103,
             'm': 77,
             'n': 79,
             'o': 68,
             'p': 95,
             'q': 148,
             'r': 69,
             's': 83,
             't': 72,
             'u': 87,
             'v': 112,
             'w': 97,
             'x': 130,
             'y': 116,
             'z': 78,
             '«': 90,
             '\xad': 140,
             '»': 91,
             '×': 141,
             'А': 63,
             'Б': 58,
             'В': 29,
             'Г': 105,
             'Д': 117,
             'Е': 53,
             'Ж': 135,
             'З': 66,
             'И': 32,
             'Й': 155,
             'К': 74,
             'Л': 89,
             'М': 57,
             'Н': 62,
             'О': 88,
             'П': 64,
             'Р': 54,
             'С': 15,
             'Т': 55,
             'У': 93,
             'Ф': 92,
             'Х': 115,
             'Ц': 124,
             'Ч': 18,
             'Ш': 113,
             'Щ': 142,
             'Ъ': 139,
             'Ь': 131,
             'Э': 2,
             'Ю': 126,
             'Я': 114,
             'а': 21,
             'б': 39,
             'в': 20,
             'г': 8,
             'д': 38,
             'е': 6,
             'ж': 52,
             'з': 19,
             'и': 9,
             'й': 34,
             'к': 35,
             'л': 26,
             'м': 7,
             'н': 5,
             'о': 4,
             'п': 28,
             'р': 16,
             'с': 33,
             'т': 3,
             'у': 14,
             'ф': 42,
             'х': 40,
             'ц': 50,
             'ч': 13,
             'ш': 11,
             'щ': 47,
             'ъ': 51,
             'ы': 41,
             'ь': 10,
             'э': 36,
             'ю': 46,
             'я': 17,
             'ё': 151,
             '‑': 153,
             '–': 37,
             '—': 59,
             '“': 127,
             '”': 128,
             '€': 150,
             '№': 147}
char_dict.update(base_dict)


class NERDictionaryAgent(DictionaryAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(NERDictionaryAgent),
            help='Sets the dictionary\'s class'
        )

    def __init__(self, opt, shared=None):
        child_opt = copy.deepcopy(opt)
        # child_opt['model_file'] += '.labels'
        child_opt['dict_file'] = os.path.splitext(child_opt['dict_file'])[0] + '.labels.dict'
        self.labels_dict = DictionaryAgent(child_opt, shared)
        self.char_dict = char_dict
        super().__init__(opt, shared)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        labels_observation = copy.deepcopy(observation)
        labels_observation['text'] = None
        observation['labels'] = None
        self.labels_dict.observe(labels_observation)
        return super().observe(observation)

    def act(self):
        self.labels_dict.act()
        super().act()
        return {'id': 'NERDictionary'}

    def save(self, filename=None, append=False, sort=True):
        filename = self.opt['model_file'] if filename is None else filename
        self.labels_dict.save(os.path.splitext(filename)[0] + '.labels.dict')
        return super().save(filename, append, sort)

    def tokenize(self, text, building=False):
        return text.split(' ') if text else []


