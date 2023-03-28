import ast
import os

import json
import requests
import uuid
from typing import Union

from abc import ABC, abstractmethod
from typing import List, Union
import yaml
from vyper import v
from easydict import EasyDict as edict


class TranslationClient(ABC):
    def __init__(self, config=None, config_path="utils/config.yaml"):
        if config is None:
            self.config = edict(yaml.full_load(open(config_path)))
        else:
            self.config = config

        self.chunk_limit = self.config.csTranslateClient.chunkLimit

    def translate(self, text_list: Union[str, List[str]], source_language: str = None, target_language: str = "en") -> Union[str, List[str]]:
        translated_transcript = []
        if isinstance(text_list, str):
            data = [text_list]
        elif isinstance(text_list, list):
            data = text_list
        else:
            raise Exception("Invalid input type. Expected a string or a list of strings")

        chunks = list(self.__partition__(data, self.chunk_limit))
        for _, chunk in enumerate(chunks):
            body = []
            for i, text in enumerate(chunk):
                body.append({'text': text})

            #detected_languages = self.do_detect(body)
            translation = self.do_translate(body, source_language=source_language, target_language=target_language)
            for i, _ in enumerate(chunk):
                translated_text = translation[i]['translations'][0]['text']
                translated_transcript.append(translated_text)

        if isinstance(text_list, str):
            translated_transcript = translated_transcript[0]

        return translated_transcript

    @abstractmethod
    def do_translate(self, text_list: Union[str, List[str]], source_language: str, target_language: str) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def do_detect(self, text_list: Union[str, List[str]]):
        pass

    @staticmethod
    def __partition__(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


class CSTranslationClient(TranslationClient):
    def __init__(self, config=None, config_path="utils/config.yaml"):
        super().__init__(config=config, config_path=config_path)
        #self.config = edict(yaml.full_load(open(config_path)))
        #self.chunk_limit = v.get(self.config.csTranslateClient.chunk_limit)

        endpoint = v.get(self.config.csTranslateClient.endpointAddress)
        subscription_key = v.get(self.config.csTranslateClient.subscriptionKey)
        location = v.get(self.config.csTranslateClient.location)

        if endpoint is None or subscription_key is None or location is None:
            raise Exception("Missing configurations for endpoint or subscription_key or location")

        self.translate_url = endpoint + '/translate'
        self.detect_url = endpoint + '/detect'
        self.api_version = v.get(self.config.csTranslateClient.apiVersion)

        self.headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def do_translate(self, text_list: Union[str, List[str]], source_language: str = None, target_language: str = "en") -> Union[str, List[str]]:
        params = {
            'api-version': self.api_version,
            'to': [target_language]
        }
        if source_language is not None:
            params['from'] = source_language

        response = requests.post(self.translate_url, params=params, headers=self.headers, json=text_list)
        response = response.json()

        return ast.literal_eval(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))

    def do_detect(self, texts: Union[str, List[str]]):
        text_list = texts
        if isinstance(texts, str):
            text_list = [texts]

        body = [{'text': t} for t in text_list]

        params = {
            'api-version': self.api_version
        }
        response = requests.post(self.detect_url, params=params, headers=self.headers, json=body)
        response = response.json()

        if 'error' in response:
            raise Exception(f"error {response['error']['code']}: {response['error']['message']}")

        detected_languages = [{'language': j['language'], 'score': j['score']} for j in response]
        if isinstance(texts, str):
            detected_languages = detected_languages[0]

        return detected_languages


from transformers import AutoTokenizer, M2M100ForConditionalGeneration
class HFTranslationClient(TranslationClient):
    def __init__(self, config_path="utils/config.yaml"):
        self.config = edict(yaml.full_load(open(config_path)))
        self.chunk_limit = v.get(self.config.csTranslateClient.chunk_limit)

    def translate(self, text_list: Union[str, List[str]], source_language: str, target_language: str) -> Union[str, List[str]]:
        model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

        prefix = f"translate French to English: "
        model_inputs = tokenizer(prefix + text_list, return_tensors="pt")

        token_id = tokenizer.get_lang_id(target_language)
        gen_tokens = model.generate(**model_inputs, forced_bos_token_id=token_id)
        translated_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return translated_texts

    def do_detect(self, text_list: Union[str, List[str]]):
        pass


if __name__ == "__main__":
    v.automatic_env()

    translator = CSTranslationClient(config_path=r"C:\Users\saaamar\repos\CRM.Sales.ML.Pipelines\src\email-suggested-reply\utils\config.yaml")
    french = translator.translate(["this is a test"], "en", "es")
    print(french)

    french = translator.translate("this is a test", "en", "es")
    print(french)

