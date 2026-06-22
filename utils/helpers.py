import pandas as pd
from plyer import notification
def pre_process(data):
    data = data.copy()

    # ==== Encoding object dtype columns to numbers ==== 
    
    # galaxy_population encoding
    data['galaxy_population'] = data['galaxy_population'].map({'Red_Sequence': 0, 'Blue_Cloud': 1}) # Should be float?
    
    # Spectral Type OHE
    data = pd.get_dummies(data, columns=['spectral_type'], dtype=int)
    
    # "Class" encoding.
    if 'class' in data.columns:
        data['class'] = data['class'].map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})

    # ====

    # 

    return data


def get_x_y(data):
    x = data.drop(columns=['id', 'class'])
    y = data['class']

    return x, y

def train_finish_notification():
    notification.notify(title='Training Done', message='TRAINING DONE ٩(^‿^)۶', timeout=5)


def gen_submission(y_pred, filename):
    test_dt = pd.read_csv('../data/test.csv')

    labels = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
    submission = pd.DataFrame({
        'id': test_dt['id'],
        'class': [labels[p] for p in y_pred]
    })
    submission.to_csv(f'../data/submissions/{filename}.csv', index=False)
    print(submission['class'].value_counts())