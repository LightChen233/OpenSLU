'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-02-12 14:35:37
LastEditTime: 2023-02-12 14:37:40
Description: 

'''
def _init():
    global _global_dict
    _global_dict = {}
 
 
def set_value(key, value):
    # set gobal value to object pool
    _global_dict[key] = value
 
 
def get_value(key):
    # get gobal value from object pool
    try:
        return _global_dict[key]
    except:
        print('读取' + key + '失败\r\n')

 