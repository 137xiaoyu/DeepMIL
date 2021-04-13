import numpy as np
import option
from utils import process_feat

if __name__ == "__main__":
    args = option.parser.parse_args()
    print('loading')

    rgb_list_file_train = args.rgb_list
    list_train = list(open(rgb_list_file_train))
    list_normal = list_train[810:]
    list_anomaly = list_train[:810]

    for i in range(len(list_normal)):
        features_normal = np.array(
            np.load(list_normal[i].strip('\n')), dtype=np.float32)
        features_normal = process_feat(features_normal, 32)
    for i in range(len(list_anomaly)):
        features_anomaly = np.array(
            np.load(list_anomaly[i].strip('\n')), dtype=np.float32)
        features_anomaly = process_feat(features_anomaly, 32)

    rgb_list_file_test = args.test_rgb_list
    list_test = list(open(rgb_list_file_test))
    for i in range(len(list_test)):
        features = np.array(
            np.load(list_test[i].strip('\n')), dtype=np.float32)

    gt = np.load(args.gt)
    print('load done')
