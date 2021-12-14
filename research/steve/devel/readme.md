## File name descriptions (modifications tested)

|PATH | FOLDER NAME | AKA | DESCRIPTION
|--|--|--|--
| devel/H3 | output_master_orig                     |        | original code without modifications (multiprocessing)
| devel/H3 | output_train_orig                      |        | run sequentially rather than in separate threads (no multiprocessing)
| devel/H3 | output_train_orig_NoDone               |        | removed done function
| devel/H3 | output_train_orig_modAux_base          | AUX-3  | removed done function and aux inputs next_obs + obs + action > obs + action
| devel/H3 | output_train_orig_modAux_smaller       | SMALL-3| AUX-3 + changed batch size 512 > 128 and model nodes [518,128] > [200,200]
| devel/H3 | output_train_orig_modAux_MNL2          |        | AUX-3 + changed ensemble size 4 > 2
| devel/H3 | output_train_orig_modAux_MNL6          |        | AUX-3 + changed ensemble size 4 > 6
| devel/H3 | output_train_orig_modAux_SimDone       |        | AUX-3 + added early termination to HalfCheetah
| devel/H3 | output_train_orig_modAux_modRew        |        | AUX-3 + added in _lam* (1-dones)* next_val (transition([next_obs, next_action])) to better match hybrid learning model update
| devel/H3 | output_train_m12p12_modLR2             |        | SMALL-3 + changed model/policy/value learning rate 3e-4 > 3e-3 and model layers [8,4] > [1,2] and policy layers [4,4] > [1,2]
| devel/H3 | output_train_m22p12_modLR2_modPretrain1k  |     | SMALL-3 + changed model/policy/value learning rate 3e-4 > 3e-3 and model layers [8,4] > [2,2] and policy layers [4,4] > [1,2] and frames_before_learning / pretrain_n / full_random_n 10000 > 1000
| devel/H3 | output_train_m22p12_modLR2_modPretrain1k  |     | SMALL-3 + changed model/policy/value learning rate 3e-4 > 3e-3 and model layers [8,4] > [4,2] and policy layers [4,4] > [1,2] and frames_before_learning / pretrain_n / full_random_n 10000 > 1000
| devel/   | output_train_orig_modAux_base          | AUX    | removed done function and aux inputs next_obs + obs + action > obs + action and horizon 3 > 5 (Hopper) / 10 (HalfCheetah)
| devel/   | output_train_orig_modAux_smaller       | SMALL  | AUX + changed batch size 512 > 128 and model nodes [518,128] > [200,200]
| devel/   | output_train_smaller_net_m42p22        |        | SMALL + changed model/policy learning rates 3e-4 > 3e-3 and model layers [8,4] > [4,2] and policy layers [4,4] > [2,2]
| devel/   | output_train_smaller_net_m24p22        |        | SMALL + changed model/policy learning rates 3e-4 > 3e-3 and model layers [8,4] > [2,4] and policy layers [4,4] > [2,2]
| devel/   | output_train_smaller_net_m24p12        |        | SMALL + changed model/policy learning rates 3e-4 > 3e-3 and model layers [8,4] > [2,4] and policy layers [4,4] > [1,2]
| devel/   | output_train_smaller_net_m12p12        | M12P12 | SMALL + changed model/policy learning rates 3e-4 > 3e-3 and model layers [8,4] > [1,2] and policy layers [4,4] > [1,2]
| devel/   | output_train_smaller_net_m22p12        | M22P12 | SMALL + changed model/policy learning rates 3e-4 > 3e-3 and model layers [8,4] > [2,2] and policy layers [4,4] > [1,2]
| devel/ddpg_only | output_train_p12_modPretrain10k |        | SMALL + changed policy learning rates 3e-4 > 3e-3 and policy layers [4,4] > [1,2] (ddpg only)
| devel/ddpg_only | output_train_p12_modPretrain1k  |        | SMALL + changed policy learning rates 3e-4 > 3e-3 and policy layers [4,4] > [1,2] + frames_before_learning / pretrain_n / full_random_n 10000 > 1000 (ddpg only)
| devel/m12p12 | output_train_m12p12_modPretrain10k     |    | M12P12 (renamed for comparison plotting)
| devel/m12p12 | output_train_m12p12_modPretrain5k      |    | M12P12 + changed frames_before_learning / pretrain_n 10000 > 5000
| devel/m12p12 | output_train_m12p12_modPretrain1k      |BUFF| M12P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000
| devel/m12p12 | output_train_m12p12_modPretrain0.5k    |    | M12P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 500
| devel/m12p12 | output_train_m12p12_modPretrain0       |    | M12P12 + changed frames_before_learning 1000 > 128, pretrain_n / full_random_n 10000 > 0
| devel/m12p12 | output_train_m12p12_modPretrain1k_MNL2 |    | BUFF + changed ensemble size 4 > 2
| devel/m12p12 | output_train_m12p12_modPretrain1k_MNL4 |    | BUFF (renamed for comparison plotting)
| devel/m12p12 | output_train_m12p12_modPretrain1k_MNL6 |    | BUFF + changed ensemble size 4 > 6
| devel/m12p12 | output_train_m12p12_modPretrain1k_update500_modLR2 | | BUFF + changed value learning rate 3e-4 > 3e-3
| devel/m22p12 | output_train_m22p12_modPretrain1k_update500  |       | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000
| devel/m22p12 | output_train_m22p12_modPretrain1k_update250  |       | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000 and epoch_every_n 500 > 250
| devel/m22p12 | output_train_m22p12_modPretrain1k_update150  |       | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000 and epoch_every_n 500 > 125
| devel/m22p12 | output_train_m22p12_modPretrain1k_update500_done   | | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000 and learned done fn (hopper)
| devel/m22p12 | output_train_m22p12_modPretrain1k_update500_modLR  | | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000 and model/policy/value learning rates 3e-3 > 3e-4
| devel/m22p12 | output_train_m22p12_modPretrain1k_update500_modLR2 | | M22P12 + changed frames_before_learning / pretrain_n / full_random_n 10000 > 1000 and value learning rate 3e-4 > 3e-3
