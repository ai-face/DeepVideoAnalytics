
import numpy as np
import os
import os.path

def get_paths(lfw_dir, pairs_filename, file_ext):
    # print("lfw_dir={}, pairs={}, file_ext={}".format(lfw_dir, pairs, file_ext))
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []

    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    pairs = np.array(pairs)

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1] ) + '_0.' +file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2] ) + '_0.' +file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1] ) + '_0.' +file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3] ) + '_0.' +file_ext)
            issame = False
        # print("path1={}".format(path1))
        # print("path0={}".format(path0))
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0 ,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs> 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list
