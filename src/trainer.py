import time
import torch
import matplotlib.pyplot as plt

from src.util import get_device, make_bold
from src.evaluation import test_with_dataloader_points


class Trainer:
    def __init__(self, inside_data_loader, outside_data_loader_train, outside_data_loader_test, loss, lambda_factor_increase, lambda_factor_decrease, outside_data_loader_val=None):
        self.train_dataloader_inside  = inside_data_loader
        self.train_dataloader_outside = outside_data_loader_train
        self.test_dataloader_outside  = outside_data_loader_test
        self.val_dataloader_outside   = outside_data_loader_val
        self.loss_fn = loss
        self.lambda_factor_increase = lambda_factor_increase
        self.lambda_factor_decrease = lambda_factor_decrease

        self.device = get_device()
        self.total_losses = []
        self.lambdas = []
        self.output = dict()
        self.last_lr = 0
        self.lbda = 1.0
    
    def train(self, model, start_lr, epochs_per_lr, lr_reduction_factor, lr_reduction_depth, gammas):   
        self.start_time = time.time()

        last_model_path = "last_model.pt"
        best_model_path = "best_model.pt"

        lrs, num_epochs = self.get_lrs_and_num_epochs(start_lr, lr_reduction_factor, lr_reduction_depth, epochs_per_lr)
        
        if gammas is None:
            gammas = [1.0 for _ in range(len(lrs))]
        print("Gammas:", gammas)

        # lr is set in the loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)
        
        max_radius_difference = model.determine_max_radius_difference(self.train_dataloader_inside)
        result = max_radius_difference <= 0.0
        print("All radii are correct initially:", result, "Max radius difference:", max_radius_difference)
        assert result, "Radii are not correct initially"

        best_test_result, _, test_time_outside = test_with_dataloader_points(model, self.test_dataloader_outside, inside=False)
        print("Initial outside accuracy test:", best_test_result, "%. Time: {:.2f} s,".format(test_time_outside))

        if self.val_dataloader_outside is not None:
            initial_val_result_outside, _, val_time_outside = test_with_dataloader_points(model, self.val_dataloader_outside, inside=False)
            print("Initial outside accuracy val:", initial_val_result_outside, "%. Time: {:.2f} s,".format(val_time_outside))

        torch.save(model.state_dict(), last_model_path)
        torch.save(model.state_dict(), best_model_path)

        epoch = 0
        self.total_losses = []
        self.lambdas = []
        for lr, epochs, gamma in zip(lrs, num_epochs, gammas):

            # set learn rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # train the model the given number of epochs with the given learn rates
            for _ in range(epochs):       
                self.output = dict()

                epoch += 1
                
                self.output['epoch'] = (epoch, 0)
                self.output['lr'] = (lr, 6)

                success = False
                while not success:
                    train_start_time = time.time()
                    model.load_state_dict(torch.load(last_model_path, weights_only=False))
                    self._train_one_epoch_points(gamma, model, optimizer)
                    
                    max_difference = model.determine_max_radius_difference(self.train_dataloader_inside)
                    success = max_difference <= 0.0
                    
                    if not success:
                        new_lambda = self.lbda * self.lambda_factor_increase
                        self.lbda = new_lambda
                        print(f"Radii are not correct. Reloading model. New lambda:{new_lambda:.1f}, Max radius difference: {max_difference:.4f}")

                torch.save(model.state_dict(), last_model_path)
                
                train_time = time.time() - train_start_time
                self.output['train_time'] = (train_time, 3)

                test_result_outside, _, test_time_outside = test_with_dataloader_points(model, self.test_dataloader_outside, inside=False)
                self.output['test acc out'] = (test_result_outside, 2)
                self.output['test_time'] = (test_time_outside, 3)

                self.output['found new best'] = ("No", 0)
                if test_result_outside > best_test_result:
                    best_test_result = test_result_outside
                    torch.save(model.state_dict(), best_model_path)
                    self.output['found new best'] = (make_bold("Yes"), 0)

                if self.val_dataloader_outside is not None:
                    val_result_outside, _, val_time_outside = test_with_dataloader_points(model, self.val_dataloader_outside, inside=False)
                    self.output['Val acc out'] = (val_result_outside, 2)
                    self.output['val_time'] = (val_time_outside, 3)

                self.output["total_time"] = (time.time() - self.start_time, 3)
                self.last_lr = lr

                self.lambdas.append(self.lbda)
                self.output['lambda'] = (self.lbda, 2)

                new_lambda = self.lbda * self.lambda_factor_decrease

                self.lbda = new_lambda

                self.print_update_console()

        print("Total Time: ", (time.time() - self.start_time) / 60, "min") 
        print("Final lambda:", self.lbda)

        model.load_state_dict(torch.load(best_model_path, weights_only=False))

        return model
   
    def _train_one_epoch_points(self, gamma, model, optimizer): 
        model.train()
        model.to(self.device)

        epoch_loss = 0
        epoch_bce_or_hinge_loss = 0
        epoch_kr_loss = 0
                
        for (points_inside, radii_inside), (points_outside, radii_outside) in zip(self.train_dataloader_inside, self.train_dataloader_outside):

            optimizer.zero_grad()
                    
            points_inside = points_inside.to(self.device)
            output_inside = model.forward(points_inside)

            points_outside = points_outside.to(self.device)
            output_outside = model.forward(points_outside)

            radii_inside = radii_inside.to(self.device)
            if radii_outside is not None:
                radii_outside = radii_outside.to(self.device)

            loss, bce_or_hinge_loss, kr_loss = self.loss_fn(self.lbda, output_inside, output_outside, radii_inside, radii_outside, gamma)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce_or_hinge_loss += bce_or_hinge_loss.item()
            epoch_kr_loss += kr_loss.item()

        self.total_losses.append(epoch_loss)
        self.output['epoch_loss'] = (epoch_loss, 5)
        self.output['bce_loss/hinge_loss'] = (epoch_bce_or_hinge_loss, 5)   
        self.output['kr_loss'] = (epoch_kr_loss, 5)


    def get_lrs_and_num_epochs(self, start_lr, lr_reduction_factor, lr_reduction_depth, epochs_per_lr):
        lrs = []
        num_epochs = []
        run_lr = start_lr

        for _ in range(lr_reduction_depth):
            lrs.append(run_lr)
            num_epochs.append(epochs_per_lr)
            run_lr *= lr_reduction_factor

        print("Epochs:", num_epochs)
        print("lrs:", lrs)
        print("Total number of epochs", sum(num_epochs))

        return lrs, num_epochs
    
    def print_update_console(self):
        for key, (value, precision) in self.output.items():
            if isinstance(value, float):
                print(key, "{:.{precision}f}".format(value, precision=precision), end=" ")
            else:
                print(key, value, end=" ")
        print()

    def plot_training(self, start_epoch=None, end_epoch=None):
        for list, name in [(self.total_losses, "Losses"), (self.lambdas, "Lambdas")]:
            if start_epoch is None:
                start_epoch = 0
            if end_epoch is None:
                end_epoch = len(list)
            plt.plot(list[start_epoch:end_epoch])
            plt.xlabel('Epoch')
            plt.ylabel(name)
            plt.show()
