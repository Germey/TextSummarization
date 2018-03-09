import re

a = '09azAZ'
for i in a:
    print('='*10)
    print(i)
    print(ord(i))
    print(ord(i) + 65248)
    print(chr(ord(i) + 65248))

    
    if 65296<= i <= 65305 and 65345 <= i <= 65370 and 65313 <= i <= 65338:
        print(True)