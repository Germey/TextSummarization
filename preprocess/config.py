from preprocess.pipeline import *

__all__ = ['*']

# pipelines enabled
ENABLE_PIPELINES = [
    StripPipeline,
    PhonePipeline,
    EmailPipeline,
    UrlPipeline,
    RemovePipeline,
    HalfWidthPipeline,
    LowerPipeline,
    ReplacePipeline,
    SegmentPipeline
]

# configs
URL_PLACEHOLDER = 'url'

URL_REGEX = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'

REMOVE_PATTERNS = [
    '\s*',
    '<Paragraph>',
    '{!--.*?--}',
    '您的浏览器不支持video标签。',
    '【.*?】',
    '（.*?）',
    '\(.*?\)',
    '\[.*?\]',
    '…*',
]

REPLACE_PATTERNS = [
    ('“', '"'),
    ('”', '"'),
    (',', '，'),
    ('!', '！'),
    (';', '；'),
]

PHONE_REGEX = '13[0123456789]{1}\d{8}|15[012356789]\d{8}|18[0123456789]\d{8}|17[678]\d{8}|14[57]\d{8}'
PHONE_PLACEHOLDER = 'phone'

EMAIL_REGEX = '\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*'
EMAIL_PLACEHOLDER = 'email'

SEGMENT_JOIN_FLAG = '\t'

SEGMENT_WORDS = [
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    URL_PLACEHOLDER
]
