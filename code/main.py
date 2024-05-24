
import sys

print(sys.argv)
file_path = sys.argv[1]
file_type = sys.argv[2]
kakao_api = sys.argv[3]
kakao_url = sys.argv[4]

if file_type == '0':
    from table import table
    table("/data/"+file_path,kakao_api,kakao_url)
elif file_type == '1':
    from document import document
    document("/data/"+file_path,kakao_api,kakao_url)