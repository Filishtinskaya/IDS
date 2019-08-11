from cryptography.fernet import Fernet
import pickle
import pandas as pd

#41Sj7FIz7JxHygT6OM0kGWWbGX32NlepAYqIPN7Rtl0= - model
#2lhPuSxL9WRPEacvVyv1PLTc6V2iQIwTTiTKuawxfQk=

MODEL_PATH = 'model.bin'
OUTPUT_PATH = "labels.bin"

def serializeModel(model):
    model = pickle.dumps(model)
    key = Fernet.generate_key()
    f = Fernet(key)
    print("Writing model to file with key: {} \n Please note that, otherwise you won't be able to use the model."
          .format(key))
    model = f.encrypt(model)
    model_file = open(MODEL_PATH, mode='wb')
    model_file.write(model)
    model_file.close()

def deserializeModel(key):
    model_file = open(MODEL_PATH, mode='rb')
    try:
        #print(key)
        key = Fernet(key)
        model = model_file.read()
        model = key.decrypt(model)
        print(model)
        model = pickle.loads(model)
        #print(model)
        return model
    except Exception as e:
        return None
    finally:
        model_file.close()

def process_packet(fe, anomDetector):
    x = fe.get_next_vector()
    return anomDetector.process(x)

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def write(self, row):
        self.data.append(row)

    def __del__(self):
        key = Fernet.generate_key()
        f = Fernet(key)
        file = open(self.filename, 'wb')
        self.data = pickle.dumps(self.data)
        self.data = f.encrypt(self.data)
        file.write(self.data)
        print(
            'Encryption key for file "{}" is {} . Please note it, otherwise you won\'t be able to read file later.'.format(
                self.filename, key))
        file.close()

def labelName(x):
    if (x == -1):
        return "anomaly"
    else:
        return "normal"

def getLabels(key):
    labels_file = open(OUTPUT_PATH, mode='rb')
    try:
        f = Fernet(key)
        labels = labels_file.read()
        labels = f.decrypt(labels)
        labels = pickle.loads(labels)
        labels = pd.DataFrame(labels)
        labels.reset_index(inplace=True)
        labels.columns = ['num', 'val']
        labels['val'] = labels['val'].apply(labelName)
        return labels
    except Exception as e:
        return None
    finally:
        labels_file.close()

