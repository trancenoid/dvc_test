import json
import warnings
import sys


class Settings():
    
    '''
        Helper class to parse all the config files and expose as dictionary objects for easy access

        To add later : Versioning, auto fetch by version tags like "latest", "best"

    '''

    def __init__(self, 
    conn_json_path, 
    train_data_json_path,
    filter_json_path,
    train_json_path,
    test_json_path
     ) -> None:

        try :

            with open(train_data_json_path, 'rb') as f:
                self.train_data_config = json.load(f)

        
            with open(conn_json_path, 'rb') as f:
                self.conn_config = json.load(f)

    
            with open(filter_json_path, 'rb') as f:
                self.filters_config = json.load(f)

        
            with open(train_json_path, 'rb') as f:
                self.train_config = json.load(f)
        
        except FileNotFoundError as e:
            print("Some paths are wrong or files are missing", e)
            raise

        try :
            with open(test_json_path, 'rb') as f:
                self.test_config = json.load(f) 
            
        except FileNotFoundError as e:
            warnings.warn("TEST CONFIG NOT FOUND !!, Proceeding to train without testing", e)
            self.test_config = None






 

