import time
import sys
import torch
import copy


def training_model(start_epoch, end_epoch, batch_size, patience, trainer, evaluator,
                   train_set, valid_set, test_set, model_path, log_path):

    valid_previous = sys.maxsize
    p_count = 0
    min_epoch = None

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # for p in trainer.net.encoder.parameters():
    #     p.requires_grad = False

    # for p in trainer.net.feature_net.parameters():
    #     p.requires_grad = False

    for i in range(start_epoch, end_epoch):
        start_time = time.time()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=False, **kwargs)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, **kwargs)

        # print('started training')
        trainer.one_epoch(train_loader)
        # print('finished training')

        if i % 10 == 0:
            train_img_xy, train_img, train_xy, train_dis = evaluator.all_in_one(trainer.net, train_loader)
            valid_img_xy, valid_img, valid_xy, valid_dis = evaluator.all_in_one(trainer.net, valid_loader)
            test_img_xy, test_img, test_xy, test_dis = evaluator.all_in_one(trainer.net, test_loader)

            interval = (time.time() - start_time) / 60

            message = 'epoch {}, train {:.8f} {:.8f} {:.8f} {:.8f},' \
                      ' valid {:.8f} {:.8f} {:.8f} {:.8f},' \
                      ' test {:.8f} {:.8f} {:.8f} {:.8f}, time {:.2f}'.format(
                        i,
                        train_img_xy, train_img, train_xy, train_dis,
                        valid_img_xy, valid_img, valid_xy, valid_dis,
                        test_img_xy, test_img, test_xy, test_dis,
                        interval)
            print(message)
            with open(log_path, 'a') as log:
                log.write(message + '\n')
        else:
            # train_img_xy, train_img, train_xy, train_dis = evaluator.all_in_one(trainer.net, train_loader)
            valid_img_xy, valid_img, valid_xy, valid_dis = evaluator.all_in_one(trainer.net, valid_loader)
            test_img_xy, test_img, test_xy, test_dis = evaluator.all_in_one(trainer.net, test_loader)

            interval = (time.time() - start_time) / 60

            message = 'epoch {}, train NA NA NA NA,' \
                      ' valid {:.8f} {:.8f} {:.8f} {:.8f},' \
                      ' test {:.8f} {:.8f} {:.8f} {:.8f}, time {:.2f}'.format(
                        i,
                        valid_img_xy, valid_img, valid_xy, valid_dis,
                        test_img_xy, test_img, test_xy, test_dis,
                        interval)
            print(message)
            with open(log_path, 'a') as log:
                log.write(message + '\n')

        # check !!! Only save those whose validation score is lower at that iter.
        # print(valid_previous)

        if valid_dis < valid_previous:
            min_epoch = i
            p_count = 0
            print('saving weights...')
            torch.save({'epoch': i,
                        'net_state_dict': trainer.net.state_dict(),
                        'optimizer_state_dict': trainer.opt.state_dict()
                        }, model_path)
            valid_previous = valid_dis




        else:
            p_count += 1

        # early stopping if valid_mean_acc plateau for 100 epochs
        if p_count > patience:
            print('val_loss did not decrease for {} epochs consequently.'.format(patience))
            break

    with open(log_path, 'a') as log:
        log.write('Min epoch: {}'.format(min_epoch) + '\n')

    print('training ended.')
    return
