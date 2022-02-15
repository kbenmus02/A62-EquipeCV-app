import pickle
#import gzip
import bz2

def pickle_save(object_to_save: object, file_info: str):
    compressed_file = bz2.open(file_info,"wb")
    pickle.dump(object_to_save, compressed_file)
    compressed_file.close()

def pickle_read(file_info: str) -> object:
    compressed_file = bz2.open(file_info,"rb")
    result=pickle.load(compressed_file)
    compressed_file.close()
    return result

def split_file_info(file_info: str) -> (str, str, str):
    file_info_temp  = file_info.replace("/", "\\")
    info_token_list = file_info_temp.split("\\")
    file_token_list = info_token_list[-1].split(".")
    file_name       = file_token_list[0]
    file_extension  = file_token_list[1]
    path = "/".join(info_token_list[:-1])
    return path, file_name, file_extension