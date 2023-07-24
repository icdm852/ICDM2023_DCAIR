def set_template(args):
    if args.template is None:
        return
    elif args.template.startswith('train_dcair'):
        args.mode = 'train'
        args.use_xbm = True
        args.min_rating = 0 
        args.min_sc = 0
        args.min_uc = 10
        args.split = 'leave_one_out'

        args.dataloader_code = 'dcair'
        batch = args.batch
        args.train_batch_size = batch
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampling_seed = 98765
        args.test_negative_sample_size = 1000

        args.trainer_code = 'dcair'
        args.optimizer = 'Adam'
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.metric_ks = [5, 10, 20]
        args.model_code = 'dcair'
        args.model_init_seed = 0
        args.head_num = 76661 

        # if args.head_num_code =='num_85':
        #     if args.min_uc == 5:
        #         if args.dataset_code == 'beauty':
        #             args.head_num = 215
        # elif args.head_num_code =='num_80':
        #     if args.min_uc == 10:
        #         if args.dataset_code == 'music':
        #                 args.head_num = 76661 

