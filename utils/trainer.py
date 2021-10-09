# @Author:Xie Ningwei
# @Date:2021-10-09 16:03:50
# @LastModifiedBy:Xie Ningwei
# @Last Modified time:2021-10-09 16:05:49
import os
import shutil
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

            self.scheduler.step(epoch)

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
            best_acc_epoch = epoch if valid_acc > best_acc else best_acc_epoch
            best_valid_acc = max(valid_acc, best_valid_acc)

            # Create a checkpoint
            if epoch % self.save_interval == 0 or is_best:
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
            value, predict_label = torch.max(output, 1)
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
                value, predict_label = torch.max(output, 1)
                # loss
                loss = self.criterion(output, label) / float(n)

                a, pre, rec, f1 = self._evaluate_step(predict_label, label)
                total_loss.update(loss.item(), n)
                acc.update(a, n)
                precision.update(pre, n)
                recall.update(rec, n)
                f1_sc.update(f1, n)

        return acc.average, precision.average, recall.average, f1_sc.average, total_loss.average

    def validate(self, model_path):
        """Runs inference on test set to get the final performance metrics"""
        # Load the best performing model first
        best_state = torch.load(model_path)

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