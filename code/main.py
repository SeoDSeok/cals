
import sys
import os

print(sys.argv)
file_path = sys.argv[1]
file_type = sys.argv[2]
kakao_api = sys.argv[3]
kakao_url = sys.argv[4]

input_path = os.path.abspath(file_path)
print(input_path)
if file_type == '0':
    from table3 import table
    table(input_path, kakao_api, kakao_url)
elif file_type == '1':
    from document2 import document
    document(input_path, kakao_api, kakao_url)