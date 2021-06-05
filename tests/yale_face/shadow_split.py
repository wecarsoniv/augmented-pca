# X_train = None
# X_test = None
# Y_train = None
# Y_test = None
# labels_id_train = None
# labels_id_test = None
# labels_shadow_train = None
# labels_shadow_test = None

# lt0_idx = np.where(Y < 0.0)
# gt0_idx = np.where(Y > 0.0)
# lteq0_idx = np.where(Y <= 0.0)
# gteq0_idx = np.where(Y >= 0.0)

# labels_id_unique = list(np.unique(labels_id))

# for label_id in labels_id_unique:
#     label_id_idx = np.where(labels_id==label_id)
    
#     shadow_switch = np.random.randint(low=0, high=2, dtype=int)
#     inclusive_switch = np.random.randint(low=0, high=2, dtype=int)
    
#     if shadow_switch:
#         if inclusive_switch:
#             comb_idx = np.intersect1d(label_id_idx, lteq0_idx)
#             diff_idx = np.setdiff1d(label_id_idx, lteq0_idx)
#         else:
#             comb_idx = np.intersect1d(label_id_idx, lt0_idx)
#             diff_idx = np.setdiff1d(label_id_idx, lt0_idx)
#     else:
#         if inclusive_switch:
#             comb_idx = np.intersect1d(label_id_idx, gteq0_idx)
#             diff_idx = np.setdiff1d(label_id_idx, gteq0_idx)
#         else:
#             comb_idx = np.intersect1d(label_id_idx, gt0_idx)
#             diff_idx = np.setdiff1d(label_id_idx, gt0_idx)
    
#     train_idx = comb_idx.copy()
#     test_idx = diff_idx.copy()
    
#     train_switch_idx = np.random.randint(0, len(train_idx), dtype=int)
#     test_switch_idx = np.random.randint(0, len(test_idx), dtype=int)
    
#     train_idx[train_switch_idx] = diff_idx[test_switch_idx]
#     test_idx[test_switch_idx] = comb_idx[train_switch_idx]

#     if X_train is None:
#         X_train = X[train_idx]
#         X_test = X[test_idx]
#         Y_train = Y[train_idx]
#         Y_test = Y[test_idx]
#         labels_id_train = labels_id[train_idx]
#         labels_id_test = labels_id[test_idx]
#         labels_shadow_train = labels_shadow[train_idx]
#         labels_shadow_test = labels_shadow[test_idx]
#     else:
#         X_train = np.concatenate((X_train, X[train_idx]), axis=0)
#         X_test = np.concatenate((X_test, X[test_idx]), axis=0)
#         Y_train = np.concatenate((Y_train, Y[train_idx]), axis=0)
#         Y_test = np.concatenate((Y_test, Y[test_idx]), axis=0)
#         labels_id_train = np.hstack((labels_id_train, labels_id[train_idx]))
#         labels_id_test = np.hstack((labels_id_test, labels_id[test_idx]))
#         labels_shadow_train = np.hstack((labels_shadow_train, labels_shadow[train_idx]))
#         labels_shadow_test = np.hstack((labels_shadow_test, labels_shadow[test_idx]))
    