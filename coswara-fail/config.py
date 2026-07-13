\
\
\
\
\
\


RANDOM_SEED = 42

PROVENANCE_SOURCE = "https://github.com/iiscleap/Coswara-Data"
DEFAULT_RAW_DATA_DIRNAME = "external_datasets/Coswara-Data"
DEFAULT_AUDIO_TYPES = ["vowel", "cough", "breathing"]

VALID_AUDIO_EXTENSIONS = {".wav"}
METADATA_EXTENSIONS = {".json", ".csv"}
MIN_PARTICIPANT_ID_LEN = 8


SICK_SYMPTOMS = {
    "cough",
    "fever",
    "cold",
    "loss_of_smell",
}
TRUE_VALUES = {"1", "true", "yes", "y", "positive", "pos", "present"}
FALSE_VALUES = {"0", "false", "no", "n", "negative", "neg", "absent"}

EXCLUSION_REASON_UNKNOWN = "unknown"
EXCLUSION_REASON_AMBIGUOUS_STATUS = "ambiguous_covid_status"
EXCLUSION_REASON_MISSING_OR_AMBIGUOUS = "missing_or_ambiguous_label_fields"
EXCLUSION_REASON_MISSING_PARTICIPANT = "missing_participant_id"
EXCLUSION_REASON_MISSING_AUDIO = "missing_audio_path"


HEALTHY_STATUS = "healthy"

SICK_STATUS_LABELS = {"infected"}
NOT_SICK_STATUS_LABELS = {"not_infected"}
BINARY_CLASS_NAMES = ["not_infected", "infected"]

METADATA_KEEP_COLUMNS = [
    "participant_id",
    "age",
    "gender",
    "covid_status",
    "test_status",
    "cough",
    "fever",
    "cold",
    "loss_of_smell",
    "sore_throat",
    "breathing_difficulty",
    "fatigue",
    "metadata_path",
]

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_TARGET_DURATION_SEC = 5
TRIM_TOP_DB = 20
PEAK_NORMALIZE_TARGET = 1.0

CLIPPING_THRESHOLD = 0.999
NOISE_RMS_THRESHOLD = 0.01

DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


DEFAULT_TARGET_SPEC_FRAMES = 216

MFCC_COUNT = 13
F0_MIN_NOTE = "C2"
F0_MAX_NOTE = "C7"
STE_FRAME_SIZE = 1024
STE_HOP_SIZE = 512

NON_FEATURE_COLUMNS = {
    "participant_id",
    "label",
    "is_sick",
    "audio_type",
    "processed_audio_path",
    "spectrogram_path",
}

MIN_GROUP_SAMPLE_SIZE = 3
SHAPIRO_MAX_N = 5000
SIGNIFICANCE_ALPHA = 0.05
TOP_DISCRIMINATIVE_FEATURES = 10

DEFAULT_TRAIN_SIZE = 0.70
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15
SPLIT_SEQUENCE = ["train", "val", "test"]


TARGETS = ["binary"]
TARGET_LABEL_COLUMNS = {"binary": "is_sick"}

SPEC_NORM_EPS = 1e-8
AUG_TIME_SHIFT_MIN = -10
AUG_TIME_SHIFT_MAX = 10
AUG_FREQ_MASK_START_MAX = 19
AUG_FREQ_MASK_WIDTH_MIN = 4
AUG_FREQ_MASK_WIDTH_MAX = 19
AUG_TIME_MASK_START_MAX = 29
AUG_TIME_MASK_WIDTH_MIN = 4
AUG_TIME_MASK_WIDTH_MAX = 19
AUG_NOISE_STD = 0.01

TABULAR_MODELS = ["random_forest", "gradient_boosting"]
RF_SCORING = "f1_macro"

RF_CV_SPLITS = 5
RF_RANDOM_SEARCH_ITER = 100
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 400, 600, 800, 1000, 1500],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", None],
}

GBM_CV_SPLITS = 5
GBM_RANDOM_SEARCH_ITER = 60
GBM_PARAM_DISTRIBUTIONS = {
    "max_iter": [200, 300, 500, 800],
    "max_leaf_nodes": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "l2_regularization": [0.0, 0.1, 1.0],
    "max_depth": [None, 6, 10, 20],
    "min_samples_leaf": [10, 20, 40],
}

CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV_FOLDS = 5

SENSITIVITY_CV_SPLITS = 3
SENSITIVITY_PARAM_GRID = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
}

DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EARLY_STOPPING_PATIENCE = 15
CNN_LR_WARMUP_EPOCHS = 3
DEFAULT_CNN_DROPOUT = 0.4
DEFAULT_CNN_LABEL_SMOOTHING = 0.05


CNN_USE_FOCAL_LOSS = True
CNN_FOCAL_GAMMA = 2.0
CNN_USE_BALANCED_SAMPLER = True


USE_BALANCED_RF = True


TUNE_BINARY_THRESHOLD = True
BINARY_THRESHOLD_METRIC = "balanced_accuracy"


CNN_SEARCH_TRIALS = 8
CNN_SEARCH_EPOCHS = 60
CNN_SEARCH_SPACE = {
    "learning_rate": [3e-5, 5e-5, 1e-4, 2e-4],
    "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "label_smoothing": [0.0, 0.05, 0.1],
}


TARGET_BALANCING = {"binary": "class_weight"}


EMBEDDING_MODEL = "WAV2VEC2_BASE"
EMBEDDING_MODEL_HUBERT = "HUBERT_BASE"
EMBEDDING_SAMPLE_RATE = 16000
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 16


PER_AUDIO_TYPE_TABULAR = True
PER_TYPE_RF_SEARCH_ITER = 30
PER_TYPE_GBM_SEARCH_ITER = 20


FINETUNE_UNFREEZE_LAYERS = 4
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 8
FINETUNE_BACKBONE_LR = 2e-5
FINETUNE_HEAD_LR = 1e-3
FINETUNE_WEIGHT_DECAY = 1e-4
FINETUNE_EARLY_STOPPING_PATIENCE = 5
FINETUNE_MAX_SAMPLES = 80000
FINETUNE_HEAD_DROPOUT = 0.3


STACK_META_LEARNER = "logistic"


TARGET_ENSEMBLE = {
    "binary": ["cnn", "embedding"],
}

ENSEMBLE_MODELS = [
    "random_forest",
    "gradient_boosting",
    "cnn",
    "embedding",
    "embedding_hubert",
    "finetune",
]
BOOTSTRAP_ITERATIONS = 2000
CALIBRATION_BINS = 10

TOP_RF_FEATURES_TO_PLOT = 20
MODELS = ["cnn", "embedding", "ensemble"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]