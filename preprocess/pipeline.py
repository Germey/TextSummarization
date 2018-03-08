#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from preprocess import config
import jieba


class Pipeline():
    def process_all(self, data):
        """
        process all data
        :param data: array of text
        :return: array of text
        """
        results = []
        for text in data:
            result = self.process_text(text)
            results.append(result)
        return results
    
    def process_text(self, text):
        """
        process text
        :param text: text
        :return: text
        """
        raise NotImplementedError


class StripPipeline(Pipeline):
    def process_text(self, text):
        """
        strip text
        :param text:
        :return:
        """
        return text.strip()


class UrlPipeline(Pipeline):
    def process_text(self, text):
        """
        change url to placeholder
        :param text: text containing url
        :return: text containing url placeholder
        """
        return re.sub(config.URL_REGEX, config.URL_PLACEHOLDER, text, flags=re.S)


class RemovePipeline(Pipeline):
    def process_text(self, text):
        """
        remote content from text
        :param text: text before remove
        :return: text after remove
        """
        for pattern in config.REMOVE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.S)
        return text


class ReplacePipeline(Pipeline):
    def process_text(self, text):
        """
        replace content from text
        :param text: text before replacement
        :return: text after replacement
        """
        for pattern in config.REPLACE_PATTERNS:
            text = re.sub(pattern[0], pattern[1], text, flags=re.S)
        return text


class PhonePipeline(Pipeline):
    def process_text(self, text):
        """
        change phone to placeholder
        :param text: text containing phone
        :return: text containing phone placeholder
        """
        return re.sub(config.PHONE_REGEX, config.PHONE_PLACEHOLDER, text, flags=re.S)


class EmailPipeline(Pipeline):
    def process_text(self, text):
        """
        change email to placeholder
        :param text: text containing email
        :return: text containing email placeholder
        """
        return re.sub(config.EMAIL_REGEX, config.EMAIL_PLACEHOLDER, text, flags=re.S)


class SegmentPipeline(Pipeline):
    def __init__(self):
        """
        add user dict
        """
        for word in config.SEGMENT_WORDS:
            jieba.add_word(word)
    
    def process_text(self, text):
        """
        segment cut
        :param text: text before segment cut
        :return: text joined with flag after segment
        """
        
        return config.SEGMENT_JOIN_FLAG.join(jieba.cut(text))
