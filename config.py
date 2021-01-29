# ====   PATHS ===================

PATH_TO_DATASET = "./data/raw/industrial_safety_and_health_database_with_accidents_description.csv"
OUTPUT_LABELENCODER_PATH = './labelencoder.pkl'
OUTPUT_ORDINALENCODER_PATH = './ordinalencoder.pkl'
OUTPUT_SCALER_PATH = './scaler.pkl'
TOKENIZER_PATH = './tokenizer.pkl'

WORD2VEC_MODEL='.custom_word2vec_200d.pkl'
GLOVE_FILE='./custom_glove_200d.txt'
EMBEDDINGMATRIX_MODEL='./embedding_matrix.pkl'

OUTPUT_MODEL_PATH = './word2vec_dev_model.h5'

# ======= PARAMETERS ===============

SEED=1

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


FREQUENT_LABELS = { 'Critical Risk':['Pressed', 'Pressurized Systems', 'Manual Tools', 'Others',
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
CATEGORICAL_ENCODE = ['Country', 'Local', 'Industry Sector',
                          'Gender', 'Employee type',
                          'Weekday']
                          
ORDINAL_ENCODE = ['Critical Risk']


TEXT_FEATURE = 'Description'

DROP_FEATURES = ['Date']


TARGET = 'Potential Accident Level'
TARGET_LEVEL={0:'Level I: Minimal', 1:'Level II: Minor', 2: 'Level III: Moderate', 3: 'Level IV: Major', 4: 'Level V: Severe' }
TARGET_LEVEL_DESC={0:'The impact is too low. Or, employees slightly escaped from the accidents. However, it requires reporting and follow-up actions.', 
                1:'The impact is low. The employees might have faced a danger and recovered with the first-aid itself.', 
                2: 'The type of injuries or illness require medical attention. However, the employee can continue the job post medication. So, there is no time loss because of the injury.', 
                3: 'The injury can cause temporary disability to an employee that results in time loss. It can also involve one or many employees.', 
                4: 'This type of injury results in permanent disability or even more fatal' }
