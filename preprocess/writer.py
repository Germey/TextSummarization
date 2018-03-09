import json
import pickle
from os.path import join


class Writer(object):
    
    def __init__(self, folder='dataset'):
        """
        init folder
        :param folder:
        """
        self.folder = folder
    
    def write_to_txt(self, data, file_name):
        """
        write to txt line by line
        :param data: data of array
        :param file_name: target_file
        :return:
        """
        print('Write %d items to %s' % (len(data), file_name))
        with open(join(self.folder, file_name), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item)
                f.write('\n')
    
    def write_to_json(self, data, file_name, ensure_ascii=False):
        """
        write to json
        :param data: data
        :param file_name: target_file
        :param ensure_ascii: ensure ascii
        :return:
        """
        print('Write %d items to %s' % (len(data), file_name))
        with open(join(self.folder, file_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=ensure_ascii))
    
    def write_to_pickle(self, data, file_name):
        """
        output data to pickle file
        :param data: data
        :param file_name: pickle
        :return:
        """
        print('Write %d items to %s' % (len(data), file_name))
        with open(join(self.folder, file_name), 'wb') as f:
            pickle.dump(data, f)
