# ====   PATHS ===================

PATH_TO_DATASET = "../data/raw/industrial_safety_and_health_database_with_accidents_description.csv"
OUTPUT_LABELENCODER_PATH = './preprocess/labelencoder.pkl'
OUTPUT_ORDINALENCODER_PATH = './preprocess/ordinalencoder.pkl'
OUTPUT_SCALER_PATH = './preprocess/scaler.pkl'
TOKENIZER_PATH = './preprocess/tokenizer.pkl'

WORD2VEC_MODEL='../embeddings/custom_word2vec_200d.pkl'
GLOVE_FILE='../embeddings/custom_glove_200d.txt'
EMBEDDINGMATRIX_MODEL='./preprocess/embedding_matrix.pkl'

OUTPUT_MODEL_PATH = './model/word2vec_dev_model.h5'

# ======= PARAMETERS ===============

SEED=100

#Train test Split 
TEST_SIZE=.1

NLP_MAX_LENGTH=100
NLP_VOCAB_SIZE=3000
NLP_EMBEDDING_SIZE=200
UPSAMPLE_COUNT=150
MODEL_EPOCHS=50

# selected features for training
INPUT_FEATURES = ['Critical Risk', 'Local', 'Countries', 'Employee or Third Party', 'Industry Sector', 'Genre',
            # this one is only to calculate temporal variable:
            'Data', 'Description', 'Potential Accident Level']


TARGET_MAPPINGS = {'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'VI':5}
    
# imputation parameters
REF_DATE= '1994,01,01'


FREQUENT_LABELS = { 'CriticalRisk':['Pressed', 'Pressurized Systems', 'Manual Tools', 'Others',
       'Fall prevention (same level)', 'Chemical substances',
       'Liquid Metal', 'Electrical installation', 'Confined space',
       'Pressurized Systems / Chemical Substances',
       'Blocking and isolation of energies', 'Suspended Loads', 'Poll',
       'Cut', 'Fall', 'Bees', 'Fall prevention', '\nNot applicable',
       'Traffic', 'Projection', 'Venomous Animals', 'Plates',
       'Projection/Burning', 'remains of choco',
       'Vehicles and Mobile Equipment', 'Projection/Choco',
       'Machine Protection', 'Power lock', 'Burn',
       'Projection/Manual Tools', 'Individual protection equipment',
       'Electrical Shock', 'Projection of fragments']}


DATE_VARIABLE = 'Date'


# variables to encode
CATEGORICAL_ENCODE = ['Country', 'Local', 'IndustrySector',
                          'Gender', 'EmployeeType',
                          'Weekday']
                          
ORDINAL_ENCODE = ['CriticalRisk']


TEXT_FEATURE = 'Description'

DROP_FEATURES = ['Date']


TARGET = 'Potential Accident Level'
TARGET_LEVEL={0:'Level-1', 1:'Level-2', 2: 'Level-3', 3: 'Level-4', 4: 'Level-5' }
TARGET_LEVEL_DESC={0:'The risk is minimal.', 
                1:'The risk minor. However, keep a first aid kit with you.', 
                2: 'You are at moderate risk. Before undertaking this activity, ensure you have sufficient medical support in case of injuries or illness.', 
                3: 'You are at major risk. The accidents can cause temporary disability.', 
                4: 'You are at severe risk. This accident can be fatal or cause permanent disability.' }

