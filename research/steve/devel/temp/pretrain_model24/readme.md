## File name descriptions (modifications tested)

| FOLDER NAME |  DESCRIPTION
|--|--
| output_train_orig_modAux_smaller (aka AUX-SMALL)  | AUX + batch size from 512 > 128 and model layers from [518,128] > [200,200]
| output_train_smaller_net_modPretrain          | AUX-SMALL + changed model / policy learning rates from 3e-4 > 3e-3 and model layers from [8,4] > [2,4] and policy layers from [4,4] to [1,2] and frames_before_learning / pretrain_n 10000 > 1000
| output_train_smaller_net_modPretrain2          | AUX-SMALL + changed model / policy learning rates from 3e-4 > 3e-3 and model layers from [8,4] > [2,4] and policy layers from [4,4] to [1,2] and frames_before_learning 1000 > 128, pretrain_n 10000 > 0
| output_train_smaller_net_modPretrain3          | AUX-SMALL + changed model / policy learning rates from 3e-4 > 3e-3 and model layers from [8,4] > [2,4] and policy layers from [4,4] to [1,2] and frames_before_learning / pretrain_n 10000 > 1000 and explore_chance 0.05 > 1
