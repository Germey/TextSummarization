#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import config
import jieba


class Pipeline(object):
    def __str__(self):
        """
        get class name
        :return:
        """
        return self.__class__.__name__
    
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
    def __init__(self, regex=config.URL_REGEX, placeholder=config.URL_PLACEHOLDER):
        self.regex = regex
        self.placeholder = placeholder
    
    def process_text(self, text):
        """
        change url to placeholder
        :param text: text containing url
        :return: text containing url placeholder
        """
        return re.sub(self.regex, self.placeholder, text, flags=re.S)


class RemovePipeline(Pipeline):
    def __init__(self, patterns=config.REMOVE_PATTERNS):
        self.patterns = patterns
    
    def process_text(self, text):
        """
        remote content from text
        :param text: text before remove
        :return: text after remove
        """
        for pattern in self.patterns:
            text = re.sub(pattern, '', text, flags=re.S)
        return text


class ReplacePipeline(Pipeline):
    def __init__(self, patterns=config.REPLACE_PATTERNS):
        self.patterns = patterns
    
    def process_text(self, text):
        """
        replace content from text
        :param text: text before replacement
        :return: text after replacement
        """
        for pattern in self.patterns:
            text = re.sub(pattern[0], pattern[1], text, flags=re.S)
        return text


class PhonePipeline(Pipeline):
    def __init__(self, regex=config.PHONE_REGEX, placeholder=config.PHONE_PLACEHOLDER):
        self.regex = regex
        self.placeholder = placeholder
    
    def process_text(self, text):
        """
        change phone to placeholder
        :param text: text containing phone
        :return: text containing phone placeholder
        """
        return re.sub(self.regex, self.placeholder, text, flags=re.S)


class EmailPipeline(Pipeline):
    def process_text(self, text):
        """
        change email to placeholder
        :param text: text containing email
        :return: text containing email placeholder
        """
        return re.sub(config.EMAIL_REGEX, config.EMAIL_PLACEHOLDER, text, flags=re.S)


class JiebaPipeline(Pipeline):
    def __init__(self, join_flag=config.SEGMENT_JOIN_FLAG, words=config.SEGMENT_WORDS):
        """
        add user dict
        """
        self.join_flag = join_flag
        self.words = words
        for word in self.words:
            jieba.add_word(word)
    
    def process_text(self, text):
        """
        segment cut
        :param text: text before segment cut
        :return: text joined with flag after segment
        """
        return self.join_flag.join(jieba.cut(text))


class CharPipeline(Pipeline):
    def __init__(self, join_flag=config.SEGMENT_JOIN_FLAG):
        self.join_flag = join_flag
    
    def process_text(self, text):
        """
        segment cut
        :param text: text before segment cut
        :return: text joined with flag after segment
        """
        return self.join_flag.join(list(text))


class MaxPipeline(Pipeline):
    def __init__(self, max_length=config.MAX_LENGTH):
        self.max_length = max_length
    
    def process_text(self, text):
        """
        max cut
        :param text: text before cut
        :return: text after cut
        """
        return text[:self.max_length]


class HalfWidthPipeline(Pipeline):
    def f2h(self, f_str):
        """
        transfer full width to half width
        :param f_str:
        :return:
        """
        h_str = ''
        for uchar in f_str:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif inside_code >= 65281 and inside_code <= 65374:
                inside_code -= 65248
            h_str += chr(inside_code)
        return h_str
    
    def process_text(self, text):
        """
        transfer
        :param text: text contains half width
        :return: text contains full width
        """
        return self.f2h(text)


class FullWidthPipeline(Pipeline):
    def h2f(self, h_str):
        """
        transfer half width to full width
        :return:
        """
        f_str = ''
        for uchar in h_str:
            inside_code = ord(uchar)
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
            
            f_str += chr(inside_code)
        return f_str
    
    def process_text(self, text):
        """
        transfer
        :param text: text contains half width
        :return: text contains full width
        """
        return self.h2f(text)


class NumberLetterHalfPipeline(Pipeline):
    def process_text(self, text):
        """
        transfer number letter to half width
        :param text:
        :return:
        """
        result = ''
        for c in text:
            inside_code = ord(c)
            if 65296 <= inside_code <= 65305 or 65345 <= inside_code <= 65370 or 65313 <= inside_code <= 65338:
                inside_code = inside_code - 65248
                c = chr(inside_code)
            result += c
        return result


class NumberLetterFullPipeline(Pipeline):
    def process_text(self, text):
        """
        transfer number letter to full width
        :param text:
        :return:
        """
        result = ''
        for c in text:
            inside_code = ord(c)
            if 48 <= inside_code <= 57 or 97 <= inside_code <= 122 or 65 <= inside_code <= 90:
                inside_code = inside_code + 65248
                c = chr(inside_code)
            result += c
        return result


class LowerPipeline(Pipeline):
    def process_text(self, text):
        """
        transfer to lower text
        :param text:
        :return:
        """
        return text.lower()


class UpperPipeline(Pipeline):
    def process_text(self, text):
        """
        transfer to upper text
        :param text:
        :return:
        """
        return text.upper()
