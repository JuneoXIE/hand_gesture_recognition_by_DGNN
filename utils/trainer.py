# @Author:Xie Ningwei
# @Date:2021-11-24 14:28:11
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-11-26 15:54:12
import os
import shutil
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch
import hiddenlayer as hl
from tqdm import tqdm


class AveTracker:
    def __init__(self):
        self.average = 0
        self.sum = 0
        self.counter = 0

    def update(self, value, n):
        self.sum += value * n
        self.counter += n
        self.average = self.sum / self.counter


class DGNN_Trainer():
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, train_loader, valid_loader, test_loader,
                 start_epoch, end_epoch, logger, model_save_dir,
                 save_interval, param_groups, freeze_graph_until):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model_save_dir = model_save_dir
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.logger = logger
        self.save_interval = save_interval
        self.param_groups = param_groups
        self.freeze_graph_until = freeze_graph_until

    def train(self):
        """Trains the model for epochs"""
        # record best loss and accuracy
        best_valid_loss = float('inf')
        best_valid_acc = 0
        best_acc_epoch = 0

        # use hiddenlayer module to visualize training procedure
        history = hl.History()
        c_loss = hl.Canvas()
        c_metrics = hl.Canvas()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            self.logger.info('epoch {}'.format(epoch))
            
            start_time = time.time()

            # Training
            self._update_graph_freeze(epoch)

            train_acc, train_precision, train_recall, train_f1, train_loss = self._train_epoch(epoch)
            self.logger.info('train_accuracy {:.5f} train_precision {:.5f}, train_recall {:.5f}, train_f1_score {:.5f}, train_loss {:.5f}'.format(
                train_acc, train_precision, train_recall, train_f1, train_loss
            ))

            # Validation
            valid_acc, valid_precision, valid_recall, valid_f1, valid_loss = self._valid_epoch(epoch)
            self.logger.info('valid_accuracy {:.5f} valid_precision {:.5f}, valid_recall {:.5f}, valid_f1_score {:.5f}, valid_loss {:.5f}'.format(
                valid_acc, valid_precision, valid_recall, valid_f1, valid_loss
            ))

            self.scheduler.step(valid_loss)

            # Compute running time
            elapsed = time.time() - start_time
            self.logger.info('Took {} seconds'.format(elapsed))

            # hiddenlayer update
            history.log(
                epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_acc = valid_acc,
                valid_precision=valid_precision,
                valid_recall=valid_recall,
                valid_f1_score=valid_f1
            )
            
            c_loss.draw_plot([history['train_loss'], history['valid_loss']])
            c_metrics.draw_plot([history['valid_acc'], history['valid_precision'], history['valid_recall'], history['valid_f1_score']])
            c_loss.save(os.path.join(self.model_save_dir, "training_process_loss.png"))
            c_metrics.save(os.path.join(self.model_save_dir, "training_process_metrics.png"))

            # Updata best valid loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                is_best = True
            else:
                is_best = False

            # Update best accuracy
            best_acc_epoch = epoch if valid_acc > best_valid_acc else best_acc_epoch
            best_valid_acc = max(valid_acc, best_valid_acc)

            # Create a checkpoint
            if (epoch % self.save_interval == 0 and epoch >= self.save_interval) or is_best:
                state = {'epoch': epoch,
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()
                         }
                self._save_checkpoint(
                    state,
                    is_best,
                    epoch
                )

            self.logger.info("Current best validation accuracy {} in epoch {}.".format(best_valid_acc, best_acc_epoch))

    def _train_epoch(self, epoch:int):
        """Trains the model for one epoch"""
        total_loss = AveTracker()
        acc = AveTracker()
        precision = AveTracker()
        recall = AveTracker()
        f1_sc = AveTracker()

        self.model.train()
        process = tqdm(self.train_loader)
        for step, (joint_data, bone_data, label, index) in enumerate(process):
            with torch.no_grad():
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_data = joint_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_data = bone_data.float().to(self.device)
                # batch_size * 1
                label = label.reshape(-1)
                label = label.long().to(self.device)
                # real batch size
                n = len(joint_data)
            # Clear gradients
            self.optimizer.zero_grad()
            # forward
            output = self.model(joint_data, bone_data)
            softmax_func = torch.nn.Softmax(dim=1)
            soft_output = softmax_func(output)
            value, predict_label = torch.max(soft_output, 1)
            # loss
            loss = self.criterion(output, label) / float(n)
            loss.backward()
            # optimizer step
            self.optimizer.step()

            a, pre, rec, f1 = self._evaluate_step(predict_label, label)
            total_loss.update(loss.item(), n)
            acc.update(a, n)
            precision.update(pre, n)
            recall.update(rec, n)
            f1_sc.update(f1, n)


        return acc.average, precision.average, recall.average, f1_sc.average, total_loss.average

    def _valid_epoch(self, epoch, phase = 'valid'):
        """Runs validation phase on either the validation or test set"""
        total_loss = AveTracker()
        acc = AveTracker()
        precision = AveTracker()
        recall = AveTracker()
        f1_sc = AveTracker()
        self.model.eval()

        if phase == 'test':
            self.logger.info('Running test set inference...')

        loader = self.valid_loader if phase == 'valid' else self.test_loader
        process = tqdm(loader)
        for step, (joint_data, bone_data, label, index) in enumerate(process):
            with torch.no_grad():
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_data = joint_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_data = bone_data.float().to(self.device)
                # batch_size * 1
                label = label.reshape(-1)
                label = label.long().to(self.device)
                # real batch size
                n = len(joint_data)
                # forward
                output = self.model(joint_data, bone_data)
                softmax_func = torch.nn.Softmax(dim=1)
                soft_output = softmax_func(output)
                value, predict_label = torch.max(soft_output, 1)
                # loss
                loss = self.criterion(output, label) / float(n)

                a, pre, rec, f1 = self._evaluate_step(predict_label, label)
                total_loss.update(loss.item(), n)
                acc.update(a, n)
                precision.update(pre, n)
                recall.update(rec, n)
                f1_sc.update(f1, n)
                
                if phase == 'test':
                    print(predict_label.shape)
                    print(label.shape)
                    np.save('label_pred.npy',predict_label.cpu())
                    np.save('label_true.npy',label.cpu())

        return acc.average, precision.average, recall.average, f1_sc.average, total_loss.average

    def validate(self):
        """Runs inference on test set to get the final performance metrics"""
        # Load the best performing model first
        best_state = torch.load(os.path.join(self.model_save_dir, "model_best.pth"))

        self.model.load_state_dict(
            best_state['model']
        )

        # Run validation on test set
        test_acc, test_precision, test_recall, test_f1_score, _ = self._valid_epoch(
            epoch=-1,
            phase='test',
        )
        self.logger.info('test_accuracy {:.5f} test_precision {:.5f}, test_recall {:.5f}, test_f1_score {:5f}'.format(
            test_acc,
            test_precision,
            test_recall,
            test_f1_score
            )
        )

    def _update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.freeze_graph_until)
        print('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad

    def _evaluate_step(self, predict_label, label):
        predict_label = predict_label.cpu()
        label = label.cpu()
        with torch.no_grad():
            acc = accuracy_score(label, predict_label)
            precision = precision_score(label, predict_label, average='macro')
            recall = recall_score(label, predict_label, average='macro')
            f1_sc = f1_score(label, predict_label, average='macro')

            return acc, precision, recall, f1_sc

    def _save_checkpoint(self, state_dict, is_best, epoch):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        filename = os.path.join(self.model_save_dir, 'checkpoint_ep{}.pth'.format(epoch))
        torch.save(state_dict, filename)

        if is_best:
            best_filename = os.path.join(self.model_save_dir, 'model_best.pth')
            shutil.copyfile(filename, best_filename)


class TwoStreamDGNN_Trainer():
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, train_loader, valid_loader, test_loader,
                 start_epoch, end_epoch, logger, model_save_dir,
                 save_interval, param_groups, freeze_graph_until, class_num):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model_save_dir = model_save_dir
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.logger = logger
        self.save_interval = save_interval
        self.param_groups = param_groups
        self.freeze_graph_until = freeze_graph_until
        self.class_num = class_num

    def train(self):
        """Trains the model for epochs"""
        # record best loss and accuracy
        best_valid_loss = float('inf')
        best_valid_acc = 0
        best_acc_epoch = 0

        # use hiddenlayer module to visualize training procedure
        history = hl.History()
        c_loss = hl.Canvas()
        c_metrics = hl.Canvas()

        for epoch in range(self.start_epoch, self.end_epoch + 1):
            self.logger.info('epoch {}'.format(epoch))
            
            start_time = time.time()

            # Training
            self._update_graph_freeze(epoch)

            train_acc, train_precision, train_recall, train_f1, train_loss = self._train_epoch(epoch)
            self.logger.info('train_accuracy {:.5f} train_precision {:.5f}, train_recall {:.5f}, train_f1_score {:.5f}, train_loss {:.5f}'.format(
                train_acc, train_precision, train_recall, train_f1, train_loss
            ))

            # Validation
            valid_acc, valid_precision, valid_recall, valid_f1, valid_loss = self._valid_epoch(epoch)
            self.logger.info('valid_accuracy {:.5f} valid_precision {:.5f}, valid_recall {:.5f}, valid_f1_score {:.5f}, valid_loss {:.5f}'.format(
                valid_acc, valid_precision, valid_recall, valid_f1, valid_loss
            ))

            self.scheduler.step(valid_loss)

            # Compute running time
            elapsed = time.time() - start_time
            self.logger.info('Took {} seconds'.format(elapsed))

            # hiddenlayer update
            history.log(
                epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_acc = valid_acc,
                valid_precision=valid_precision,
                valid_recall=valid_recall,
                valid_f1_score=valid_f1
            )
            
            c_loss.draw_plot([history['train_loss'], history['valid_loss']])
            c_metrics.draw_plot([history['valid_acc'], history['valid_precision'], history['valid_recall'], history['valid_f1_score']])
            c_loss.save(os.path.join(self.model_save_dir, "training_process_loss.png"))
            c_metrics.save(os.path.join(self.model_save_dir, "training_process_metrics.png"))

            # Updata best valid loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                is_best = True
            else:
                is_best = False

            # Update best accuracy
            best_acc_epoch = epoch if valid_acc > best_valid_acc else best_acc_epoch
            best_valid_acc = max(valid_acc, best_valid_acc)

            # Create a checkpoint
            if (epoch % self.save_interval == 0 and epoch >= self.save_interval) or is_best:
                state = {'epoch': epoch,
                         'model': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict()
                         }
                self._save_checkpoint(
                    state,
                    is_best,
                    epoch
                )

            self.logger.info("Current best validation accuracy {} in epoch {}.".format(best_valid_acc, best_acc_epoch))

    def _train_epoch(self, epoch:int):
        """Trains the model for one epoch"""
        total_loss = AveTracker()
        acc = AveTracker()
        precision = AveTracker()
        recall = AveTracker()
        f1_sc = AveTracker()

        self.model.train()
        process = tqdm(self.train_loader)
        for step, (joint_data, bone_data, joint_motion_data, bone_motion_data, label, index) in enumerate(process):
            with torch.no_grad():
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_data = joint_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_data = bone_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_motion_data = joint_motion_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_motion_data = bone_motion_data.float().to(self.device)
                # batch_size * 1
                label = label.reshape(-1)
                label = label.long().to(self.device)
                # real batch size
                n = len(joint_data)
            # Clear gradients
            self.optimizer.zero_grad()
            # forward
            output = self.model(joint_data, bone_data, joint_motion_data, bone_motion_data)
            softmax_func = torch.nn.Softmax(dim=1)
            soft_output = softmax_func(output)
            value, predict_label = torch.max(soft_output, 1)
            # loss
            loss = self.criterion(output, label) / float(n)
            loss.backward()
            # optimizer step
            self.optimizer.step()

            a, pre, rec, f1 = self._evaluate_step(predict_label, label)
            total_loss.update(loss.item(), n)
            acc.update(a, n)
            precision.update(pre, n)
            recall.update(rec, n)
            f1_sc.update(f1, n)


        return acc.average, precision.average, recall.average, f1_sc.average, total_loss.average

    def _valid_epoch(self, epoch, phase = 'valid'):
        """Runs validation phase on either the validation or test set"""
        total_loss = AveTracker()
        acc = AveTracker()
        precision = AveTracker()
        recall = AveTracker()
        f1_sc = AveTracker()
        self.model.eval()

        if phase == 'test':
            self.logger.info('Running test set inference...')

        loader = self.valid_loader if phase == 'valid' else self.test_loader
        process = tqdm(loader)
        for step, (joint_data, bone_data, joint_motion_data, bone_motion_data, label, index) in enumerate(process):
            with torch.no_grad():
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_data = joint_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_data = bone_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                joint_motion_data = joint_motion_data.float().to(self.device)
                # batch_size * # channels * # frames * # nodes * # persons = 32 * 3 * 20 * 22 * 1
                bone_motion_data = bone_motion_data.float().to(self.device)
                # batch_size * 1
                label = label.reshape(-1)
                label = label.long().to(self.device)
                # real batch size
                n = len(joint_data)
                # forward
                output = self.model(joint_data, bone_data, joint_motion_data, bone_motion_data)
                softmax_func = torch.nn.Softmax(dim=1)
                soft_output = softmax_func(output)
                value, predict_label = torch.max(soft_output, 1)
                # loss
                loss = self.criterion(output, label) / float(n)

                a, pre, rec, f1 = self._evaluate_step(predict_label, label)
                total_loss.update(loss.item(), n)
                acc.update(a, n)
                precision.update(pre, n)
                recall.update(rec, n)
                f1_sc.update(f1, n)

                if phase == 'test':
                    import matplotlib.pyplot as plt
                    predict_label = predict_label.cpu()
                    label = label.cpu()
                    cm = confusion_matrix(label, predict_label)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] *100
                    np.set_printoptions(precision=1)
                    print(cm_normalized)
                    # draw confusion matrix
                    if self.class_num == 14:
                        label_text = ['Grab','Tap','Expand','Pinch','Rotation CW','Rotation CCW','Swipe Right',
                                      'Swipe Left','Swipe Up','Swipe Down','Swipe X','Swipe V','Swipe +','Shake']
                    else:
                        label_text = ['Grab(1)', 'Grab(2)','Tap(1)','Tap(2)','Expand(1)','Expand(2)',
                                      'Pinch(1)','Pinch(2)', 'Rotation CW(1)','Rotation CW(2)',
                                      'Rotation CCW(1)','Rotation CCW(2)','Swipe Right(1)','Swipe Right(2)',
                                      'Swipe Left(1)','Swipe Left(2)','Swipe Up(1)','Swipe Up(2)',
                                      'Swipe Down(1)','Swipe Down(2)','Swipe X(1)','Swipe X(2)',
                                      'Swipe V(1)','Swipe V(2)','Swipe +(1)','Swipe +(2)', 'Shake(1)','Shake(2)']
                    tick_marks = np.array(range(len(label_text)))
                    plt.gca().set_xticks(tick_marks, minor=True)
                    plt.gca().set_yticks(tick_marks, minor=True)
                    plt.gca().xaxis.set_ticks_position('none')
                    plt.gca().yaxis.set_ticks_position('none')
                    plt.grid(True, which='minor', linestyle='-')
                    

                    if self.class_num == 14:
                        plt.figure(figsize=(14, 7), dpi=480)
                    else:
                        plt.figure(figsize=(25, 14), dpi=480)

                    plt.matshow(cm_normalized, fignum=0, cmap=plt.cm.Blues)#, interpolation='nearest')

                    ind_array = np.arange(len(label_text))
                    x, y = np.meshgrid(ind_array, ind_array)
                    
                    for x_val, y_val in zip(x.flatten(), y.flatten()):
                        c = cm_normalized[y_val][x_val]
                        if 0.0 <= c < 50.0 :
                            plt.text(x_val, y_val, "%0.1f" % (c,), color='black', fontsize=14, va='center', ha='center', font={'style':'normal', 'weight':'semibold'})
                        elif 50.0 <= c < 100.0:
                            plt.text(x_val, y_val, "%0.1f" % (c,), color='white', fontsize=14, va='center', ha='center', font={'style':'normal', 'weight':'semibold'})
                        else:
                            plt.text(x_val, y_val, "100", color='white', fontsize=10, va='center', ha='center', font={'style':'normal', 'weight':'semibold'})
                    
                    if self.class_num == 14:
                        plt.title("Confusion Matrix for DHG 14 Dataset",fontsize= 20)
                    else:
                        plt.title("Confusion Matrix for DHG 28 Dataset")
                    plt.colorbar()
                    xlocations = np.array(range(len(label_text)))
                    plt.xticks(xlocations, label_text, rotation=45)
                    plt.yticks(xlocations, label_text)
                    plt.tick_params(labelsize=10)
                    plt.ylabel('True label',fontsize= 13)
                    plt.xlabel('Predicted label',fontsize= 13)
                    
                    if self.class_num == 14:
                        plt.savefig('confusion_matrix_DHG14.png', format='png')
                    else:
                        plt.savefig('confusion_matrix_DHG28.png', format='png')

        return acc.average, precision.average, recall.average, f1_sc.average, total_loss.average

    def validate(self):
        """Runs inference on test set to get the final performance metrics"""
        # Load the best performing model first
        best_state = torch.load(os.path.join(self.model_save_dir, "model_best.pth"))

        self.model.load_state_dict(
            best_state['model']
        )

        # Run validation on test set
        test_acc, test_precision, test_recall, test_f1_score, _ = self._valid_epoch(
            epoch=-1,
            phase='test',
        )
        self.logger.info('test_accuracy {:.5f} test_precision {:.5f}, test_recall {:.5f}, test_f1_score {:5f}'.format(
            test_acc,
            test_precision,
            test_recall,
            test_f1_score
            )
        )

    def _update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.freeze_graph_until)
        print('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad

    def _evaluate_step(self, predict_label, label):
        predict_label = predict_label.cpu()
        label = label.cpu()
        with torch.no_grad():
            acc = accuracy_score(label, predict_label)
            precision = precision_score(label, predict_label, average='macro')
            recall = recall_score(label, predict_label, average='macro')
            f1_sc = f1_score(label, predict_label, average='macro')

            return acc, precision, recall, f1_sc

    def _save_checkpoint(self, state_dict, is_best, epoch):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        filename = os.path.join(self.model_save_dir, 'checkpoint_ep{}.pth'.format(epoch))
        torch.save(state_dict, filename)

        if is_best:
            best_filename = os.path.join(self.model_save_dir, 'model_best.pth')
            shutil.copyfile(filename, best_filename)