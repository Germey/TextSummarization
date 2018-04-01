from os.path import join

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

DATE_PATTERNS = [
    '\d{2,4}年\d{1,2}月\d{1,2}日',
    '\d{1,2}月\d{1,2}日',
    '\d{2,4}年\d{1,2}月',
    # '\d{2,4}年',
    # '\d{1,2}日',
    '\d{2,4}-\d{1,2}-\d{1,2}',
    '\d{2,4}\/\d{1,2}\/\d{1,2}',
    # '\d{1,2}-\d{1,2}',
]

TIME_PATTERNS = [
    '\d{2}:\d{2}:\d{2}',
    '\d{2}:\d{2}',
    '\d{1,2}时\d{1,2}分',

]

PHONE_REGEX = '13[0123456789]{1}\d{8}|15[012356789]\d{8}|18[0123456789]\d{8}|17[678]\d{8}|14[57]\d{8}'
PHONE_PLACEHOLDER = 'phone'

EMAIL_REGEX = '\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*'
EMAIL_PLACEHOLDER = 'email'
DATE_PLACEHOLDER = 'date'
TIME_PLACEHOLDER = 'time'

SEGMENT_JOIN_FLAG = ' '

SEGMENT_WORDS = [
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    URL_PLACEHOLDER,
    DATE_PLACEHOLDER,
    TIME_PLACEHOLDER
]

GO = 'GO'
EOS = 'EOS'  # also function as PAD
UNK = 'UNK'

MAX_LENGTH = 1000
